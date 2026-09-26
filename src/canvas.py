"""Canvas viewer: a chunky pixel-art bacterium in a pygame window.

Rendering only — all simulation and session logic lives in `sim`.
"""

import math
import random

import pygame

from src import aquarium as aqua
from src.sim import World
from src.sim import session
from src.sim.animator import Animator
from src.sim.creature import Creature, SWIM_SPEED, Vec
from src.sim.world import WIDTH, HEIGHT, CreatureBrain

SCALE = 12  # pixels per world unit
FPS = 60
PIX = 4  # size of one art pixel on screen -> chunky look
ART_SIZE = 48  # art-surface side in art pixels
BODY_RADIUS_PX = 5 * PIX  # draw_body's disc radius, on-screen pixels

# palette: (body, body_dark, tail, accent) by mood
HAPPY = ((168, 100, 240), (120, 66, 180), (110, 60, 200), (255, 214, 90))
HUNGRY = ((214, 130, 84), (160, 92, 56), (170, 96, 60), (255, 170, 60))
STARVING = ((140, 140, 160), (100, 100, 120), (96, 96, 116), (200, 90, 90))
WATER_TOP = (12, 24, 48)
WATER_BOTTOM = (24, 60, 96)
FOOD_COLOR = (196, 92, 224)
FOOD_GLOW = (196, 92, 224, 40)
CONJUGATION_COLOR = (255, 214, 90)
STUFFED = (255, 170, 190)  # rosy tint a bloated ball blushes toward
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


def stuffed_palette(palette: Palette, bloat: float) -> Palette:
    """Blend the mood palette toward a rosy stuffed color as bloat grows."""
    k = 0.55 * bloat
    return tuple(lerp_color(col, STUFFED, k) for col in palette)  # type: ignore[return-value]


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Blend two RGB colors."""
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


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


def body_spine(center: tuple[float, float], heading: float, bend: float,
              segments: int, bloat: float = 0.0) -> list[tuple[float, float]]:
    """Spine points from nose to tail-end, bowed by the bend angle.

    Bloat shortens the spine: the discs pile up and the body reads as a ball.
    """
    ax, ay = center
    half_length = int(1.1 * SCALE / PIX * (1.0 - 0.75 * bloat))
    points: list[tuple[float, float]] = []
    for s in range(segments + 1):
        u = s / segments
        # heading eases from nose heading to heading - bend at the tail end
        h = heading - bend * u
        d = half_length - 2 * half_length * u  # +half_length nose .. -half_length tail
        points.append((ax + math.cos(h) * d, ay + math.sin(h) * d))
    return points


def draw_body(art: pygame.Surface, center: tuple[float, float],
              heading: float, bend: float, palette: Palette,
              bloat: float = 0.0) -> None:
    """Soft body: discs along a bowed spine, dark rear rim, eyes at the nose.

    Bloat fattens every disc and shortens the spine, so a stuffed creature
    swells into a near-sphere while keeping its eyes and rear rim.
    """
    body_c, body_dark, _, _ = palette
    segments = 6
    radius = 5 + 4 * bloat
    # a stuffed ball goes jelly-like: half transparent at full bloat
    alpha = int(255 * (1.0 - 0.5 * bloat))
    spine = body_spine(center, heading, bend, segments, bloat)

    def disc(cx: float, cy: float, r: float,
             color: tuple[int, ...]) -> None:
        """A filled pixel circle on the art surface."""
        for yy in range(int(cy - r) - 1, int(cy + r) + 2):
            for xx in range(int(cx - r) - 1, int(cx + r) + 2):
                if (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r and 0 <= xx < ART_SIZE and 0 <= yy < ART_SIZE:
                    art.set_at((int(xx), int(yy)), color)

    for i, (sx, sy) in enumerate(spine):
        u = i / segments
        # teardrop tail end; bloat keeps the rear fat so the shape stays round
        rr = radius if u <= 0.5 else max(radius * 0.6, radius - int((u - 0.5) * 6))
        disc(sx, sy, rr, (*body_c, alpha))
    for i in range(segments // 2, segments + 1):  # darker rim on the rear half
        u = i / segments
        rr = max(radius * 0.6, radius - int((u - 0.5) * 6)) if u > 0.5 else radius
        sx, sy = spine[i]
        h = heading - bend * u
        disc(sx + math.sin(h) * rr * 0.6, sy - math.cos(h) * rr * 0.6,
             1.2, (*body_dark, alpha))

    nx, ny = spine[0]  # eyes ride the nose, not the body center
    perp = heading + math.pi / 2
    for sign in (-1, 1):
        ex = nx + math.cos(perp) * sign * radius * 0.4
        ey = ny + math.sin(perp) * sign * radius * 0.4
        disc(ex, ey, 1.6, (255, 255, 255, 255))
        disc(ex + math.cos(heading) * 0.8, ey + math.sin(heading) * 0.8,
             0.8, (20, 20, 30, 255))


def draw_flagellum(art: pygame.Surface, center: tuple[float, float],
                   heading: float, t: float, speed: float, bend: float,
                   tail_color: tuple[int, int, int],
                   bloat: float = 0.0) -> None:
    """Swimming-flagellum beat: one slow travelling wave along the tail.

    A real flagellum sends a single low-frequency wave from base to tip;
    fast small ripples read as vibration, not swimming. The wave here is
    slow (~1.2 Hz at full speed), the root barely moves, and the bend
    grows smoothly toward the tip. The root follows the body's tail-end
    bend, so tail and body stay connected when the body bows. Bloat
    withers the tail: a stuffed ball is too heavy to whip it around.
    """
    ax, ay = center
    body_edge = int(1.1 * SCALE / PIX * (1.0 - 0.75 * bloat)) + 3
    tail_len = int(3.2 * SCALE / PIX * (1.0 - 0.6 * bloat))
    tail_alpha = int(255 * (1.0 - 0.5 * bloat))
    phase = t * 7.5 * min(1.0, max(0.15, speed))  # ~1.2 Hz at full swim
    root_heading = heading - bend  # the tail grows out of the bowed rear

    def dot(px: float, py: float, r: int) -> None:
        for yy in range(int(py) - r, int(py) + r + 1):
            for xx in range(int(px) - r, int(px) + r + 1):
                if 0 <= xx < ART_SIZE and 0 <= yy < ART_SIZE:
                    art.set_at((xx, yy), (*tail_color, tail_alpha))

    for i in range(tail_len):
        u = i / tail_len
        d = body_edge + i
        bx = ax - math.cos(root_heading) * d
        by = ay - math.sin(root_heading) * d
        # one long lazy wave: the tip swings ~4 screen px, the root ~1 px,
        # so the tail trails behind the body instead of whipping around
        wave = math.sin(phase - u * 2.2) * (0.3 + 1.1 * u * u)
        bx += -math.sin(root_heading) * wave
        by += math.cos(root_heading) * wave
        dot(bx, by, max(1, 2 - i // (tail_len // 2)))


def creature_speed(creature: Creature) -> float:
    """Distance covered last tick, normalized to full swim speed."""
    if len(creature.trail) < 2:
        return 0.0
    a, b = creature.trail[-1], creature.trail[-2]
    return math.hypot(a.x - b.x, a.y - b.y) / SWIM_SPEED


def draw_bacterium(screen: pygame.Surface, world: World, t: float,
                   cb: CreatureBrain, anim: Animator) -> None:
    """One creature: trail + flagellum + body, upscaled with NEAREST.

    Position, heading and body bend come from the animator's smoothed
    state, so the body glides and bows instead of snapping every tick.
    """
    creature = cb.body
    palette = stuffed_palette(
        mood_palette_hunger(creature.hunger, creature.starving), creature.bloat,
    )
    x, y = anim.pos.x * SCALE, anim.pos.y * SCALE
    # a stuffed ball bobs gently: it is lighter than water and just hangs there
    y += math.sin(t * 0.9 + anim.heading) * 2.5 * creature.bloat
    heading = anim.heading

    draw_trail(screen, creature.trail, palette[0])

    art = pygame.Surface((ART_SIZE, ART_SIZE), pygame.SRCALPHA)
    center = (ART_SIZE / 2, ART_SIZE / 2)
    draw_flagellum(art, center, heading, t, creature_speed(creature),
                   anim.bend, palette[2], creature.bloat)
    draw_body(art, center, heading, anim.bend, palette, creature.bloat)

    big = pygame.transform.scale(art, (ART_SIZE * PIX, ART_SIZE * PIX))
    screen.blit(big, (round(x - big.get_width() / 2),
                      round(y - big.get_height() / 2)))


def draw_all(screen: pygame.Surface, world: World, t: float,
             animators: dict[int, Animator]) -> None:
    """Every creature, oldest first (newborns on top)."""
    for cb in world.creatures:
        anim = animators.setdefault(cb.brain.id, Animator())
        draw_bacterium(screen, world, t, cb, anim)


def draw_conjugations(screen: pygame.Surface, world: World,
                      animators: dict[int, Animator], t: float) -> None:
    """A soft halo under each pair of creatures whose bodies touch.

    Gated on the sim's own contact state, not on smoothed render
    positions: the animator lags the truth, so a distance check here
    kept drawing the link after the pair had already bounced apart —
    phantom dashes in open water. The halo sits under the bodies as one
    glow covering the pair: dots or lines over the sprites read as
    scratches, a merged halo reads as exchange.
    """
    cbs = world.creatures
    for i, a in enumerate(cbs):
        for b in cbs[i + 1:]:
            if not world.touching.get((a.brain.id, b.brain.id), False):
                continue
            aa, ba = animators.get(a.brain.id), animators.get(b.brain.id)
            if aa is None or ba is None:
                continue
            ax, ay = aa.pos.x * SCALE, aa.pos.y * SCALE
            bx, by = ba.pos.x * SCALE, ba.pos.y * SCALE
            mx, my = (ax + bx) / 2, (ay + by) / 2
            r = math.hypot(bx - ax, by - ay) / 2 + BODY_RADIUS_PX
            pulse = 0.5 + 0.5 * math.sin(t * 2.5)
            glow = pygame.Surface((int(r) * 2 + PIX, int(r) * 2 + PIX),
                                  pygame.SRCALPHA)
            cx = cy = glow.get_width() // 2
            for k in range(3):  # soft edge: three fading rings
                alpha = int((12 + 16 * pulse) / (k + 1))
                pygame.draw.circle(glow, (*CONJUGATION_COLOR, alpha),
                                   (cx, cy), int(r) - k * PIX)
            screen.blit(glow, (mx - cx, my - cy))


def draw_light_shafts(screen: pygame.Surface, t: float) -> None:
    """Soft light shafts: wide translucent bands that drift and breathe."""
    win_w, win_h = screen.get_size()
    shafts = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
    for i in range(3):
        drift = math.sin(t * 0.05 + i * 2.1) * 40.0
        breathe = 0.5 + 0.5 * math.sin(t * 0.11 + i * 1.3)
        x0 = win_w * (0.22 + 0.28 * i) + drift
        tilt = 0.45 + 0.1 * i
        top_half = 50.0 + 18.0 * i
        alpha = int(7 + 6 * breathe)
        pygame.draw.polygon(shafts, (210, 230, 255, alpha), [
            (x0 - top_half, 0), (x0 + top_half, 0),
            (x0 + top_half + tilt * win_h, win_h),
            (x0 - top_half + tilt * win_h, win_h),
        ])
    screen.blit(shafts, (0, 0))


def draw_stats(screen: pygame.Surface, font: pygame.font.Font,
               speed: int, paused: bool) -> None:
    """The only HUD line: controls, plus speed and pause state."""
    line = (f"speed {speed}x {'[PAUSED]' if paused else ''}   "
            "click spawn  f pour food  +/- speed  space pause  s save  "
            "f11 fullscreen  q quit")
    screen.blit(font.render(line, True, TEXT), (10, 8))


def set_display(fullscreen: bool) -> pygame.Surface:
    """(Re)create the display. The window is freely resizable (maximize
    button works); on every resize the world itself grows or shrinks to
    match at the same zoom, so the tank always fills the window."""
    flags = pygame.RESIZABLE
    if fullscreen:
        flags |= pygame.FULLSCREEN
    return pygame.display.set_mode(SCREEN, flags)


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


def floor_y_for(world: World) -> int:
    """Sand baseline on screen: the tank bottom minus a floor band."""
    return int(world.height * SCALE) - int(1.2 * SCALE)


def run(world: World, speed: int) -> None:
    """Main pygame loop: events, sim ticks, render, repeat."""
    pygame.init()
    fullscreen = False
    screen = set_display(fullscreen)
    pygame.display.set_caption("nz-octopus")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    def build_scenery() -> tuple[pygame.Surface, list[aqua.Weed], int]:
        """Build everything sized in pixels for the current tank dims."""
        win_w = int(world.width * SCALE)
        floor_y = floor_y_for(world)
        backdrop = aqua.build_backdrop(win_w, int(world.height * SCALE), PIX,
                                       floor_y, (WATER_TOP, WATER_BOTTOM),
                                       world.terrain)
        weeds = aqua.make_weeds(win_w, floor_y, PIX, count=7, seed=42,
                                terrain=world.terrain)
        return backdrop, weeds, floor_y

    backdrop, weeds, floor_y = build_scenery()
    rng = random.Random()
    bubbles: list[aqua.Bubble] = []
    animators: dict[int, Animator] = {}

    paused = False
    t = 0.0
    running = True
    try:
        while running:
            t += 1.0 / FPS
            scenery_dirty = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        fullscreen = not fullscreen
                        screen = set_display(fullscreen)
                    elif event.key == pygame.K_f:
                        world.pour_food()
                    elif event.key == pygame.K_s:
                        session.save_brains([cb.brain for cb in world.creatures])
                    else:
                        speed, paused, running = handle_key(event.key, speed, paused)
                elif event.type in (pygame.VIDEORESIZE, pygame.WINDOWRESIZED):
                    # the tank itself follows the window at a fixed zoom:
                    # more screen means more water, not a stretched picture
                    if event.type == pygame.WINDOWRESIZED:
                        ew, eh = event.x, event.y  # window events use x/y
                    else:
                        ew, eh = event.w, event.h  # legacy VIDEORESIZE uses w/h
                    world.resize(max(1.0, ew / SCALE), max(1.0, eh / SCALE))
                    scenery_dirty = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    wx, wy = screen_to_world(*event.pos)
                    world.spawn_at(wx, wy)
            if scenery_dirty:
                backdrop, weeds, floor_y = build_scenery()
            win_w, win_h = screen.get_size()
            if not paused:
                for _ in range(speed):
                    world.step()
                for cb in world.creatures:
                    animators.setdefault(cb.brain.id, Animator()).update(
                        cb.body, 1.0 / FPS,
                    )
                aqua.step_bubbles(bubbles, -4.0, 1.0 / FPS, float(win_w))
                if random.random() < 0.05:
                    aqua.spawn_bubble(bubbles, win_w, floor_y, rng)
            draw_background(screen, backdrop)
            draw_light_shafts(screen, t)
            aqua.draw_weeds(screen, weeds, t, floor_y, PIX, world.terrain)
            draw_food(screen, world, t)
            draw_conjugations(screen, world, animators, t)
            draw_all(screen, world, t, animators)
            aqua.draw_bubbles(screen, bubbles, PIX)
            draw_stats(screen, font, speed, paused)
            pygame.display.flip()
            clock.tick(FPS)
    except KeyboardInterrupt:
        pass  # Ctrl-C exits cleanly; the brain is saved by the entry point
    finally:
        pygame.quit()
