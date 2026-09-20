"""Canvas viewer: a chunky pixel-art bacterium in a pygame window.

Rendering only — all simulation and session logic lives in `sim`.
"""

import math
import random

import pygame

import aquarium as aqua
from sim import World
from sim import session
from sim.creature import Creature, Vec
from sim.world import WIDTH, HEIGHT, CreatureBrain

SCALE = 12  # pixels per world unit
FPS = 60
PIX = 4  # size of one art pixel on screen -> chunky look
ART_SIZE = 48  # art-surface side in art pixels

# palette: (body, body_dark, tail, accent) by mood
HAPPY = ((168, 100, 240), (120, 66, 180), (110, 60, 200), (255, 214, 90))
HUNGRY = ((214, 130, 84), (160, 92, 56), (170, 96, 60), (255, 170, 60))
STARVING = ((140, 140, 160), (100, 100, 120), (96, 96, 116), (200, 90, 90))
WATER_TOP = (12, 24, 48)
WATER_BOTTOM = (24, 60, 96)
FOOD_COLOR = (196, 92, 224)
FOOD_GLOW = (196, 92, 224, 40)
TEXT = (200, 210, 230)

SCREEN = (int(WIDTH * SCALE), int(HEIGHT * SCALE))
Palette = tuple[tuple[int, int, int], tuple[int, int, int],
                tuple[int, int, int], tuple[int, int, int]]


def mood_palette_hunger(hunger: float, starving: float) -> Palette:
    """Color by hunger: bright when fed, fades to gray while starving."""
    if starving <= 0.0:
        return HAPPY if hunger < 45 else HUNGRY
    base = HUNGRY if hunger < 85 else STARVING
    k = min(1.0, starving)
    return tuple(  # type: ignore[return-value]
        tuple(int(c + (128 - c) * k) for c in col) for col in base
    )


def mood_palette(world: World) -> Palette:
    """Color of the first creature (kept for single-creature callers)."""
    return mood_palette_hunger(world.hunger, world.starving)


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Blend two RGB colors."""
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


FLOOR_Y = SCREEN[1] - int(1.2 * SCALE)


def draw_background(screen: pygame.Surface, backdrop: pygame.Surface) -> None:
    """Static aquarium backdrop."""
    screen.blit(backdrop, (0, 0))


def draw_food(screen: pygame.Surface, world: World, t: float) -> None:
    """Glowing diamonds that pulse slowly."""
    for f in world.food:
        x, y = f.pos.x * SCALE, f.pos.y * SCALE
        pulse = 1.0 + 0.1 * math.sin(t * 1.5 + f.pos.x)
        r = max(PIX, int(4 * pulse))
        glow = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow, FOOD_GLOW, (r * 2, r * 2), r * 2)
        screen.blit(glow, (x - r * 2, y - r * 2))
        cx, cy = round(x / PIX) * PIX, round(y / PIX) * PIX
        pygame.draw.polygon(screen, FOOD_COLOR,
                            [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)])


def draw_trail(screen: pygame.Surface, trail: list[Vec],
               color: tuple[int, int, int]) -> None:
    """Soft fading squares behind one creature."""
    points = trail[::2]
    for i, p in enumerate(points):
        alpha = int(35 * (i + 1) / max(1, len(points)))
        square = pygame.Surface((PIX, PIX), pygame.SRCALPHA)
        square.fill((*color, alpha))
        screen.blit(square, (p.x * SCALE - PIX / 2, p.y * SCALE - PIX / 2))


def draw_body(art: pygame.Surface, center: tuple[float, float],
              heading: float, palette: Palette) -> None:
    """Pill-shaped body, dark rear rim, highlight and eyes."""
    body_c, body_dark, _, _ = palette
    ax, ay = center
    half_length = int(1.1 * SCALE / PIX)
    radius = 5

    def disc(cx: float, cy: float, r: float,
             color: tuple[int, ...]) -> None:
        """A filled pixel circle on the art surface."""
        for yy in range(int(cy - r) - 1, int(cy + r) + 2):
            for xx in range(int(cx - r) - 1, int(cx + r) + 2):
                if (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r and 0 <= xx < ART_SIZE and 0 <= yy < ART_SIZE:
                    art.set_at((int(xx), int(yy)), color)

    for s in range(-half_length, half_length + 1):
        rr = radius if s >= 0 else max(3, radius - (-s) // 3)  # teardrop tail end
        disc(ax + math.cos(heading) * s, ay + math.sin(heading) * s, rr, (*body_c, 255))
    for s in range(-half_length, 0):  # darker rim on the rear half, for volume
        rr = max(3, radius - (-s) // 3)
        disc(ax + math.cos(heading) * s + math.sin(heading) * rr * 0.6,
             ay + math.sin(heading) * s - math.cos(heading) * rr * 0.6,
             1.2, (*body_dark, 255))

    perp = heading + math.pi / 2
    for sign in (-1, 1):
        ex = ax + math.cos(heading) * 3 + math.cos(perp) * sign * 2
        ey = ay + math.sin(heading) * 3 + math.sin(perp) * sign * 2
        disc(ex, ey, 1.6, (255, 255, 255, 255))
        disc(ex + math.cos(heading) * 0.8, ey + math.sin(heading) * 0.8,
             0.8, (20, 20, 30, 255))


def draw_flagellum(art: pygame.Surface, center: tuple[float, float],
                   heading: float, t: float,
                   tail_color: tuple[int, int, int]) -> None:
    """Wavy tail: a sine whose amplitude grows toward the tip."""
    ax, ay = center
    body_edge = int(1.1 * SCALE / PIX) + 3
    tail_len = int(3.2 * SCALE / PIX)
    phase = t * 0.45 * 8.0

    def dot(px: float, py: float, r: int) -> None:
        for yy in range(int(py) - r, int(py) + r + 1):
            for xx in range(int(px) - r, int(px) + r + 1):
                if 0 <= xx < ART_SIZE and 0 <= yy < ART_SIZE:
                    art.set_at((xx, yy), (*tail_color, 255))

    for i in range(tail_len):
        u = i / tail_len
        d = body_edge + i
        bx = ax - math.cos(heading) * d
        by = ay - math.sin(heading) * d
        wave = math.sin(phase - u * 4.0) * (1.0 + 2.6 * u)
        bx += -math.sin(heading) * wave
        by += math.cos(heading) * wave
        dot(bx, by, max(1, 2 - i // (tail_len // 2)))


def draw_bacterium(screen: pygame.Surface, world: World, t: float,
                   cb: CreatureBrain) -> None:
    """One creature: trail + flagellum + body, upscaled with NEAREST."""
    creature = cb.body
    palette = mood_palette_hunger(creature.hunger, creature.starving)
    x, y = creature.pos.x * SCALE, creature.pos.y * SCALE
    heading = creature.heading

    draw_trail(screen, creature.trail, palette[0])

    art = pygame.Surface((ART_SIZE, ART_SIZE), pygame.SRCALPHA)
    center = (ART_SIZE / 2, ART_SIZE / 2)
    draw_flagellum(art, center, heading, t, palette[2])
    draw_body(art, center, heading, palette)

    big = pygame.transform.scale(art, (ART_SIZE * PIX, ART_SIZE * PIX))
    screen.blit(big, (round(x - big.get_width() / 2),
                      round(y - big.get_height() / 2)))


def draw_all(screen: pygame.Surface, world: World, t: float) -> None:
    """Every creature, oldest first (newborns on top)."""
    for cb in world.creatures:
        draw_bacterium(screen, world, t, cb)


def draw_light_shafts(screen: pygame.Surface, t: float) -> None:
    """Soft light shafts: wide translucent bands that drift and breathe."""
    shafts = pygame.Surface(SCREEN, pygame.SRCALPHA)
    for i in range(3):
        drift = math.sin(t * 0.05 + i * 2.1) * 40.0
        breathe = 0.5 + 0.5 * math.sin(t * 0.11 + i * 1.3)
        x0 = SCREEN[0] * (0.22 + 0.28 * i) + drift
        tilt = 0.45 + 0.1 * i
        top_half = 50.0 + 18.0 * i
        alpha = int(7 + 6 * breathe)
        pygame.draw.polygon(shafts, (210, 230, 255, alpha), [
            (x0 - top_half, 0), (x0 + top_half, 0),
            (x0 + top_half + tilt * SCREEN[1], SCREEN[1]),
            (x0 - top_half + tilt * SCREEN[1], SCREEN[1]),
        ])
    screen.blit(shafts, (0, 0))


def draw_stats(screen: pygame.Surface, font: pygame.font.Font,
               world: World, speed: int, paused: bool,
               monitor: session.ProgressMonitor | None) -> None:
    """HUD lines in the top-left corner."""
    lines = [
        f"hunger {world.hunger:5.1f}   food {world.food_eaten}   "
        f"bumps {world.wall_bumps}   starved {world.starvations}   "
        f"creatures {len(world.creatures)}",
        f"avg reward {world.recent_avg_reward:+.3f}   "
        f"curiosity {world.brain.config.epsilon:.3f}   "
        f"exp {world.brain.experience}",
    ]
    if monitor is not None:
        lines.append(monitor.hud_line())
    lines.append(f"speed {speed}x {'[PAUSED]' if paused else ''}   "
                 f"click spawn  +/- speed  space pause  s save  q quit")
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, TEXT), (10, 8 + i * 18))


def handle_key(key: int, speed: int, paused: bool) -> tuple[int, bool, bool]:
    """Apply one keypress. Returns (speed, paused, keep_running)."""
    if key in (pygame.K_q, pygame.K_ESCAPE):
        return speed, paused, False
    if key == pygame.K_SPACE:
        return speed, not paused, True
    if key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
        return min(500, speed * 2), paused, True
    if key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        return max(1, speed // 2), paused, True
    return speed, paused, True


def screen_to_world(mx: int, my: int) -> tuple[float, float]:
    """Convert a mouse click to world coordinates."""
    return mx / SCALE, my / SCALE


def run(world: World, speed: int,
        monitor: session.ProgressMonitor | None = None) -> None:
    """Main pygame loop: events, sim ticks, render, repeat."""
    pygame.init()
    screen = pygame.display.set_mode(SCREEN)
    pygame.display.set_caption("nz-octopus")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)
    backdrop = aqua.build_backdrop(SCREEN[0], SCREEN[1], PIX, FLOOR_Y,
                                   (WATER_TOP, WATER_BOTTOM), world.terrain)
    weeds = aqua.make_weeds(SCREEN[0], FLOOR_Y, PIX, count=7, seed=42,
                            terrain=world.terrain)
    rng = random.Random()
    bubbles: list[aqua.Bubble] = []

    paused = False
    t = 0.0
    running = True
    try:
        while running:
            t += 1.0 / FPS
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    speed, paused, running = handle_key(event.key, speed, paused)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    wx, wy = screen_to_world(*event.pos)
                    world.spawn_at(wx, wy)
            if not paused:
                for _ in range(speed):
                    world.step()
                    if monitor is not None:
                        monitor.step(world.food_eaten)
                aqua.step_bubbles(bubbles, -4.0, 1.0 / FPS, float(SCREEN[0]))
                if random.random() < 0.05:
                    aqua.spawn_bubble(bubbles, SCREEN[0], FLOOR_Y, rng)
            draw_background(screen, backdrop)
            draw_light_shafts(screen, t)
            aqua.draw_weeds(screen, weeds, t, FLOOR_Y, PIX, world.terrain)
            draw_food(screen, world, t)
            draw_all(screen, world, t)
            aqua.draw_bubbles(screen, bubbles, PIX)
            draw_stats(screen, font, world, speed, paused, monitor)
            pygame.display.flip()
            clock.tick(FPS)
    except KeyboardInterrupt:
        pass  # Ctrl-C exits cleanly; the brain is saved in main()
    finally:
        pygame.quit()


def main() -> None:
    """Entry point: parse args, load the brains, run, save the brains."""
    args = session.parse_args("Bacterium canvas", default_speed=1)
    brains = session.open_brains(args.fresh)
    world = World(creatures=[CreatureBrain(
        body=Creature(
            pos=Vec(WIDTH / 2, HEIGHT / 2),
            heading=random.uniform(0, 6.283185307179586),
            hunger=30.0,
            starving=0.0,
        ),
        brain=brain,
    ) for brain in brains])
    monitor = session.ProgressMonitor(world.brain)
    try:
        run(world, max(1, args.speed), monitor)
    finally:
        session.save_brains([cb.brain for cb in world.creatures])


if __name__ == "__main__":
    main()
