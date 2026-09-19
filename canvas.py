"""Canvas viewer: a chunky pixel-art bacterium in a pygame window.

Rendering only — all simulation and session logic lives in `sim`.
"""

import math

import pygame

from sim import World
from sim import session
from sim.world import WIDTH, HEIGHT

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
SAND = (58, 50, 40)
FOOD_COLOR = (120, 230, 120)
FOOD_GLOW = (120, 230, 120, 40)
TEXT = (200, 210, 230)

SCREEN = (int(WIDTH * SCALE), int(HEIGHT * SCALE))
Palette = tuple[tuple[int, int, int], tuple[int, int, int],
                tuple[int, int, int], tuple[int, int, int]]


def mood_palette(world: World) -> Palette:
    """Color by hunger: bright when fed, fades to gray while starving."""
    if world.starving <= 0.0:
        return HAPPY if world.hunger < 45 else HUNGRY
    base = HUNGRY if world.hunger < 85 else STARVING
    k = min(1.0, world.starving)
    return tuple(  # type: ignore[return-value]
        tuple(int(c + (128 - c) * k) for c in col) for col in base
    )


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Blend two RGB colors."""
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


def draw_background(screen: pygame.Surface) -> None:
    """Gradient water, a strip of sand and a few light rays."""
    for y in range(SCREEN[1]):
        t = y / SCREEN[1]
        pygame.draw.line(screen, lerp_color(WATER_TOP, WATER_BOTTOM, t),
                         (0, y), (SCREEN[0], y))
    sand_h = int(1.2 * SCALE)
    pygame.draw.rect(screen, SAND, (0, SCREEN[1] - sand_h, SCREEN[0], sand_h))
    rays = pygame.Surface(SCREEN, pygame.SRCALPHA)
    for i in range(4):
        x0 = int(SCREEN[0] * (0.15 + 0.2 * i))
        pygame.draw.polygon(rays, (255, 255, 255, 10),
                            [(x0, 0), (x0 + 60, 0), (x0 - 100, SCREEN[1]),
                             (x0 - 200, SCREEN[1])])
    screen.blit(rays, (0, 0))


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


def draw_trail(screen: pygame.Surface, world: World,
               color: tuple[int, int, int]) -> None:
    """Soft fading squares behind the creature."""
    points = world.trail[::2]
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


def draw_bacterium(screen: pygame.Surface, world: World, t: float) -> None:
    """The whole creature: trail + flagellum + body, upscaled with NEAREST."""
    palette = mood_palette(world)
    x, y = world.position.x * SCALE, world.position.y * SCALE
    heading = world.heading

    draw_trail(screen, world, palette[0])

    art = pygame.Surface((ART_SIZE, ART_SIZE), pygame.SRCALPHA)
    center = (ART_SIZE / 2, ART_SIZE / 2)
    draw_flagellum(art, center, heading, t, palette[2])
    draw_body(art, center, heading, palette)

    big = pygame.transform.scale(art, (ART_SIZE * PIX, ART_SIZE * PIX))
    screen.blit(big, (round(x - big.get_width() / 2),
                      round(y - big.get_height() / 2)))


def draw_stats(screen: pygame.Surface, font: pygame.font.Font,
               world: World, speed: int, paused: bool) -> None:
    """HUD lines in the top-left corner."""
    lines = [
        f"hunger {world.hunger:5.1f}   food {world.food_eaten}   "
        f"bumps {world.wall_bumps}   starved {world.starvations}",
        f"avg reward {world.recent_avg_reward:+.3f}   "
        f"curiosity {world.brain.config.epsilon:.3f}   "
        f"exp {world.brain.experience}",
        f"speed {speed}x {'[PAUSED]' if paused else ''}   "
        f"+/- speed  space pause  s save  q quit",
    ]
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


def run(world: World, speed: int) -> None:
    """Main pygame loop: events, sim ticks, render, repeat."""
    pygame.init()
    screen = pygame.display.set_mode(SCREEN)
    pygame.display.set_caption("nz-octopus")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

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
            if not paused:
                for _ in range(speed):
                    world.step()
            draw_background(screen)
            draw_food(screen, world, t)
            draw_bacterium(screen, world, t)
            draw_stats(screen, font, world, speed, paused)
            pygame.display.flip()
            clock.tick(FPS)
    except KeyboardInterrupt:
        pass  # Ctrl-C exits cleanly; the brain is saved in main()
    finally:
        pygame.quit()


def main() -> None:
    """Entry point: parse args, load the brain, run, save the brain."""
    args = session.parse_args("Bacterium canvas", default_speed=1)
    brain = session.open_brain(args.fresh)
    world = World(brain=brain)
    try:
        run(world, max(1, args.speed))
    finally:
        session.save_brain(brain)


if __name__ == "__main__":
    main()
