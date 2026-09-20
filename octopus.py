"""Canvas viewer: a chunky pixel-art octopus driven by the same brain.

Rendering only — the simulation, brains and save file are shared with
canvas.py; this is just a different body drawn over the same world.
"""

import math
import random

import pygame

import aquarium as aqua
from sim import World
from sim import session
from sim.creature import Creature, SWIM_SPEED, Vec
from sim.jet import JetAnimator
from sim.world import WIDTH, HEIGHT, CreatureBrain

SCALE = 12  # pixels per world unit
FPS = 60
PIX = 4  # size of one art pixel on screen -> chunky look
ART_SIZE = 64  # art-surface side in art pixels (bigger body than the bacterium)

# palette: (mantle, mantle_dark, arm, accent) by mood
HAPPY = ((168, 100, 240), (120, 66, 180), (110, 60, 200), (255, 214, 90))
HUNGRY = ((214, 130, 84), (160, 92, 56), (170, 96, 60), (255, 170, 60))
STARVING = ((140, 140, 160), (100, 100, 120), (96, 96, 116), (200, 90, 90))
WATER_TOP = (12, 24, 48)
WATER_BOTTOM = (24, 60, 96)
FOOD_COLOR = (196, 92, 224)
FOOD_GLOW = (196, 92, 224, 40)
TEXT = (200, 210, 230)

SCREEN = (int(WIDTH * SCALE), int(HEIGHT * SCALE))
FLOOR_Y = SCREEN[1] - int(1.2 * SCALE)
Palette = tuple[tuple[int, int, int], tuple[int, int, int],
                tuple[int, int, int], tuple[int, int, int]]

N_ARMS = 8
ARM_LEN = 14  # art pixels per arm
EYE_WHITE = (255, 255, 255, 255)
EYE_PUPIL = (20, 20, 30, 255)


def mood_palette_hunger(hunger: float, starving: float) -> Palette:
    """Color by hunger: bright when fed, fades to gray while starving."""
    if starving <= 0.0:
        return HAPPY if hunger < 45 else HUNGRY
    base = HUNGRY if hunger < 85 else STARVING
    k = min(1.0, starving)
    return tuple(  # type: ignore[return-value]
        tuple(int(c + (128 - c) * k) for c in col) for col in base
    )


def draw_disc(art: pygame.Surface, cx: float, cy: float, r: float,
              color: tuple[int, ...]) -> None:
    """A filled pixel circle on the art surface."""
    for yy in range(int(cy - r) - 1, int(cy + r) + 2):
        for xx in range(int(cx - r) - 1, int(cx + r) + 2):
            if (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r and 0 <= xx < ART_SIZE and 0 <= yy < ART_SIZE:
                art.set_at((int(xx), int(yy)), color)


def draw_mantle(art: pygame.Surface, center: tuple[float, float],
                heading: float, bend: float, squeeze: float,
                palette: Palette) -> None:
    """Head dome along a bowed spine: fat front, tapering rear, eyes.

    `squeeze` is 0..1 of mantle contraction during a jet pulse: the dome
    squashes along its length and widens slightly, like a real pulse.
    """
    mantle, mantle_dark, _, accent = palette
    ax, ay = center
    half_length = 9 * (1.0 - 0.25 * squeeze)  # dome shortens when firing
    fat = 8.0 * (1.0 + 0.15 * squeeze)  # ...and bulges sideways
    segments = 8

    def spine_h(u: float) -> float:
        """Heading along the spine: nose heading, easing to bent rear."""
        return heading - bend * u

    def spine_at(u: float) -> tuple[float, float]:
        d = half_length - 2 * half_length * u
        h = spine_h(u)
        return ax + math.cos(h) * d, ay + math.sin(h) * d

    def radius_at(u: float) -> float:
        """Fat dome over the front half, tapering to the rear tip."""
        if u <= 0.45:
            return fat
        return max(2.5, fat - (u - 0.45) * 13.0)

    for i in range(segments + 1):
        u = i / segments
        sx, sy = spine_at(u)
        draw_disc(art, sx, sy, radius_at(u), (*mantle, 255))
    for i in range(segments // 2, segments + 1):  # shaded rear half
        u = i / segments
        sx, sy = spine_at(u)
        h = spine_h(u)
        rr = radius_at(u)
        draw_disc(art, sx + math.sin(h) * rr * 0.6,
                  sy - math.cos(h) * rr * 0.6, 1.4, (*mantle_dark, 255))

    # eyes: big, forward-looking, on the dome
    ex_f, ey_f = spine_at(0.18)
    perp = heading + math.pi / 2
    for sign in (-1, 1):
        ex = ex_f + math.cos(perp) * sign * 3.2
        ey = ey_f + math.sin(perp) * sign * 3.2
        draw_disc(art, ex, ey, 2.4, EYE_WHITE)
        draw_disc(art, ex + math.cos(heading) * 1.0,
                  ey + math.sin(heading) * 1.0, 1.2, EYE_PUPIL)
    # a small accent spot on the mantle for character
    sx, sy = spine_at(0.35)
    draw_disc(art, sx + math.cos(perp) * 4.5, sy + math.sin(perp) * 4.5,
              1.0, (*accent, 255))


def draw_arms(art: pygame.Surface, center: tuple[float, float],
              heading: float, bend: float, t: float, speed: float,
              thrusting: bool, effort: float,
              arm_color: tuple[int, int, int]) -> None:
    """Eight reactive arms: stream behind in a thrust, relax into a fan
    during the glide, gather under the body at rest.
    """
    ax, ay = center
    rear_h = heading - math.pi - bend  # where the arms attach
    phase = t * 5.0 * min(1.0, max(0.2, speed))
    # during a thrust the arms trail straight back and bunch together;
    # while gliding they relax outward; at rest they droop into a fan
    stream = effort if thrusting else 0.0
    relax = (1.0 - effort) if thrusting else (0.35 + 0.4 * (1.0 - speed))

    def arm_dot(px: float, py: float, r: int) -> None:
        for yy in range(int(py) - r, int(py) + r + 1):
            for xx in range(int(px) - r, int(px) + r + 1):
                if 0 <= xx < ART_SIZE and 0 <= yy < ART_SIZE:
                    art.set_at((xx, yy), (*arm_color, 255))

    for arm in range(N_ARMS):
        # fan width: squeezed shut while streaming, wide while relaxed
        spread = (arm / (N_ARMS - 1) - 0.5) * math.tau * (0.55 * relax + 0.08)
        base_h = rear_h + spread
        # streaming arms curve less (they trail); relaxed arms curl outward
        curl = (0.9 * (arm / (N_ARMS - 1) - 0.5)) * (1.0 - stream)
        for i in range(ARM_LEN):
            u = i / ARM_LEN
            d = 4 + i
            h = base_h + curl * u
            px = ax + math.cos(h) * d
            py = ay + math.sin(h) * d
            # wave: lively in a thrust, lazy in a glide, slow at rest
            wave = math.sin(phase * (1.0 + stream) - u * 3.0 + arm * 0.7)
            wave *= (0.3 + 1.6 * u * u) * (1.0 - 0.5 * stream)
            px += -math.sin(h) * wave
            py += math.cos(h) * wave
            arm_dot(px, py, max(1, 2 - i // (ARM_LEN // 2)))
            if i % 3 == 2:  # sucker bumps on the underside
                sx = px + math.cos(h + math.pi / 2) * 1.2
                sy = py + math.sin(h + math.pi / 2) * 1.2
                arm_dot(sx, sy, 1)


def creature_speed(creature: Creature) -> float:
    """Distance covered last tick, normalized to full swim speed."""
    if len(creature.trail) < 2:
        return 0.0
    a, b = creature.trail[-1], creature.trail[-2]
    return math.hypot(a.x - b.x, a.y - b.y) / SWIM_SPEED


def draw_octopus(screen: pygame.Surface, t: float,
                 cb: CreatureBrain, anim: JetAnimator) -> None:
    """One octopus: arms behind, mantle in front, upscaled with NEAREST."""
    creature = cb.body
    palette = mood_palette_hunger(creature.hunger, creature.starving)
    x, y = anim.pos.x * SCALE, anim.pos.y * SCALE
    heading = anim.heading
    speed = creature_speed(creature)
    # mantle contraction peaks mid-thrust
    squeeze = anim.effort * (1.0 - anim.pulse_phase) if anim.thrusting else 0.0

    art = pygame.Surface((ART_SIZE, ART_SIZE), pygame.SRCALPHA)
    center = (ART_SIZE / 2, ART_SIZE / 2)
    draw_arms(art, center, heading, anim.bend, t, speed,
              anim.thrusting, anim.effort, palette[2])
    draw_mantle(art, center, heading, anim.bend, squeeze, palette)

    big = pygame.transform.scale(art, (ART_SIZE * PIX, ART_SIZE * PIX))
    screen.blit(big, (round(x - big.get_width() / 2),
                      round(y - big.get_height() / 2)))


def draw_all(screen: pygame.Surface, world: World, t: float,
             animators: dict[int, JetAnimator]) -> None:
    """Every octopus, oldest first (newborns on top)."""
    for cb in world.creatures:
        anim = animators.setdefault(cb.brain.id, JetAnimator())
        draw_octopus(screen, t, cb, anim)


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
    animators: dict[int, JetAnimator] = {}

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
                for cb in world.creatures:
                    animators.setdefault(cb.brain.id, JetAnimator()).update(
                        cb.body, 1.0 / FPS,
                    )
                aqua.step_bubbles(bubbles, -4.0, 1.0 / FPS, float(SCREEN[0]))
                if random.random() < 0.05:
                    aqua.spawn_bubble(bubbles, SCREEN[0], FLOOR_Y, rng)
            screen.blit(backdrop, (0, 0))
            draw_light_shafts(screen, t)
            aqua.draw_weeds(screen, weeds, t, FLOOR_Y, PIX, world.terrain)
            draw_food(screen, world, t)
            draw_all(screen, world, t, animators)
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
    args = session.parse_args("Octopus canvas", default_speed=1)
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
