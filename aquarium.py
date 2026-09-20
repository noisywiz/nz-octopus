"""Aquarium decorations: sand, rocks, pebbles, seaweed, bubbles.

Rendering only — draws chunky pixel art directly on the pygame screen.
"""

import math
import random
from dataclasses import dataclass

import pygame

from sim import terrain as tn

Color = tuple[int, int, int]

SAND_TOP = (232, 204, 150)
SAND_MID = (196, 166, 112)
SAND_DEEP = (148, 118, 74)
SAND_DARK = (108, 84, 52)
ROCK_OUT = (28, 30, 42)
ROCK_DARK = (72, 74, 92)
ROCK_MID = (108, 112, 134)
ROCK_LIGHT = (158, 164, 188)
MOSS = (96, 156, 84)
MOSS_DARK = (60, 116, 62)
WEED_LIGHT = (60, 160, 90)
WEED_DARK = (38, 112, 66)
WEED_OLIVE = (96, 150, 60)
BUBBLE = (170, 215, 255)


@dataclass(frozen=True)
class Weed:
    """One seaweed anchored to the floor."""

    base_x: int
    height: int
    phase: float
    speed: float
    color: Color


@dataclass
class Bubble:
    """A bubble rising toward the surface."""

    x: float
    y: float
    radius: int
    phase: float


def _px(surface: pygame.Surface, x: int, y: int, size: int,
        color: Color) -> None:
    """Draw one chunky pixel."""
    surface.fill(color, (x, y, size, size))


# Hand-drawn sprites. Letters map to ROCK_PALETTE; '.' is transparent.
BIG_ROCK = (
    "..............gg......................",
    "...........gGgg....ll.................",
    ".........gGlllllllllll..ll............",
    ".......gGllmmmmmmmmmmmmmlll...........",
    "......Gllmmmmmmmmmmmmmmmmdll..........",
    ".....llmmmmmmmmmmmmmmmmdddddl.........",
    "....lmmmmmmmmmmmmmmdmmmddddddl........",
    "...lmmmmmmmdmmmmmmmmmmmdddddddl.......",
    "..lmmmmmmmmdddmmmmmmmddddddddddl......",
    ".lmmmmmmmdddddmmmmmmdddddddddddl......",
    "lmmmmmmmddddddmmmmmdddddddddddddl.....",
    "lmmmmmmddddddddmmmddddddddddddddl.....",
    "lmmdmmdddddddddddmmdddddddddddddl.....",
    "lmddddddddddddddmmdddddddddddddddl....",
    "lmddddddddddddddddddddddddddddddl.....",
    ".lddddddddddddddddddddddddddddddl.....",
    "..lodddddddddddddddddddddddddol.......",
    "...loodddddddddddddddddddloool........",
    ".....lloooolllllllllloooool...........",
    "..........lGGllllllll.................",
)
SMALL_ROCK = (
    ".......gg.............",
    "....gGlllll...........",
    "...Gllmmmmmmll........",
    "..llmmmmmmmmmmdl......",
    ".lmmmmmmmmmmddddl.....",
    "lmmmmmmmmmmmdddddl....",
    "lmmdmmmmmmmmddddddl...",
    "lmdddddddddddddddl....",
    ".lodddddddddddodl.....",
    "..looooolllloool......",
    ".......GGlll..........",
)

ROCK_PALETTE: dict[str, Color] = {
    "l": ROCK_LIGHT, "m": ROCK_MID, "d": ROCK_DARK,
    "o": ROCK_OUT, "g": MOSS, "G": MOSS_DARK,
}


def draw_sprite(surface: pygame.Surface, rows: tuple[str, ...],
                x: int, y: int, size: int,
                palette: dict[str, Color], flip: bool = False) -> None:
    """Stamp a letter-art sprite with its bottom-left at (x, y)."""
    for row_i, row in enumerate(rows):
        yy = y - (len(rows) - row_i) * size
        for col_i, ch in enumerate(row):
            if ch == ".":
                continue
            color = palette[ch]
            xx = x + ((len(row) - 1 - col_i) if flip else col_i) * size
            surface.fill(color, (xx, yy, size, size))


def _build_sand(surface: pygame.Surface, rng: random.Random, width: int,
                height: int, size: int, floor_y: int,
                terrain: list[int]) -> None:
    """Dune floor from the shared terrain heightmap, one column per art px."""
    for col, level in enumerate(terrain):
        x = col * size
        if x >= width:
            break
        top = floor_y - size - level * size
        _px(surface, x, top, size, SAND_TOP)
        _px(surface, x, top + size, size, SAND_MID)
        for y in range(top + 2 * size, height, size):
            depth = y - top
            if depth > 5 * size:
                _px(surface, x, y, size, SAND_DEEP)
            elif rng.random() < 0.02:
                _px(surface, x, y, size, SAND_DARK)
            else:
                _px(surface, x, y, size, SAND_MID)


def _pebble(surface: pygame.Surface, cx: int, cy: int, size: int,
            color: Color) -> None:
    """A tiny 2x1-art-pixel pebble."""
    _px(surface, cx, cy, size, color)
    _px(surface, cx + size, cy, size, color)


def build_backdrop(width: int, height: int, pix: int,
                   floor_y: int, water: tuple[Color, Color],
                   terrain: list[int] | None = None) -> pygame.Surface:
    """Static layer: water gradient, sand dunes, rocks, pebbles.

    `terrain` is the shared sim heightmap; when omitted, a render-only one
    is generated (for previews/tests that run without a World).
    """
    backdrop = pygame.Surface((width, height))
    rng = random.Random(20260920)
    if terrain is None:
        terrain = tn.build(width // pix)
    for y in range(height):
        t = y / height
        top, bottom = water
        color = tuple(
            int(a + (b - a) * t) for a, b in zip(top, bottom)
        )
        pygame.draw.line(backdrop, color, (0, y), (width, y))

    _build_sand(backdrop, rng, width, height, pix, floor_y, terrain)
    # park rocks on the local dune level so none floats above the sand
    placements = (
        (BIG_ROCK, 0.03, False),
        (SMALL_ROCK, 0.24, False),
        (SMALL_ROCK, 0.47, True),
        (BIG_ROCK, 0.68, True),
        (SMALL_ROCK, 0.94, True),
    )
    for i, (sprite, fx, flip) in enumerate(placements):
        base_x = int(fx * width) // pix * pix
        col = min(len(terrain) - 1, base_x // pix)
        level = terrain[col]
        sink = pix * (1 if i % 2 else 0)
        draw_sprite(backdrop, sprite, base_x,
                    floor_y - pix + level * pix + sink, pix,
                    ROCK_PALETTE, flip=flip)
    for _ in range(8):  # a few pebbles resting on the sand
        _pebble(backdrop, rng.randrange(pix, width - pix, pix),
                rng.randrange(floor_y - pix * 2, height - pix, pix), pix,
                rng.choice((ROCK_MID, SAND_DEEP, ROCK_DARK)))
    return backdrop


def make_weeds(width: int, floor_y: int, pix: int, count: int,
               seed: int, terrain: list[int] | None = None) -> list[Weed]:
    """Scatter seaweed along the dune surface, denser near the edges."""
    rng = random.Random(seed)
    if terrain is None:
        terrain = tn.build(width // pix)
    weeds: list[Weed] = []
    for i in range(count):
        slot = (i + rng.uniform(0.15, 0.85)) / count
        col = min(len(terrain) - 1, int(slot * (width - 4 * pix) + 2 * pix) // pix)
        base_x = col * pix
        weeds.append(Weed(
            base_x=base_x,
            height=rng.randrange(5, 13),
            phase=rng.uniform(0.0, math.tau),
            speed=rng.uniform(0.8, 1.4),
            color=rng.choice((WEED_LIGHT, WEED_DARK, WEED_OLIVE)),
        ))
    return weeds


def draw_weeds(surface: pygame.Surface, weeds: list[Weed], t: float,
               floor_y: int, pix: int, terrain: list[int] | None = None) -> None:
    """Swaying blades rooted on the local dune level."""
    if terrain is None:
        terrain = tn.build(1000)
    for weed in weeds:
        col = min(len(terrain) - 1, weed.base_x // pix)
        root_y = floor_y - terrain[col] * pix
        for i in range(weed.height):
            k = i / weed.height
            sway = math.sin(t * weed.speed + weed.phase + i * 0.45) * k * 3.0
            x = int(weed.base_x + sway * pix) // pix * pix
            y = root_y - (i + 1) * pix
            color = weed.color if i < weed.height - 1 else WEED_DARK
            _px(surface, x, y, pix, color)
            if i % 3 == 1:  # a side leaf for volume
                _px(surface, x + pix * (1 if sway >= 0 else -1), y, pix,
                    WEED_DARK)


def spawn_bubble(bubbles: list[Bubble], width: int, floor_y: int,
                 rng: random.Random) -> None:
    """Release one bubble from the floor or a random height."""
    bubbles.append(Bubble(
        x=float(rng.randrange(0, width)),
        y=float(floor_y - rng.randrange(0, 30)),
        radius=rng.choice((2, 2, 3)),
        phase=rng.uniform(0.0, math.tau),
    ))


def step_bubbles(bubbles: list[Bubble], surface_y: float, dt: float,
                 width: float) -> None:
    """Rise, wobble, pop at the surface."""
    alive: list[Bubble] = []
    for b in bubbles:
        b.y -= 26.0 * dt
        b.x += math.sin(b.y * 0.05 + b.phase) * 0.4
        b.x = min(max(b.x, 0.0), width)
        if b.y > surface_y:
            alive.append(b)
    bubbles[:] = alive


def draw_bubbles(surface: pygame.Surface, bubbles: list[Bubble],
                 pix: int) -> None:
    """Pale hollow-looking circles."""
    for b in bubbles:
        cx, cy = int(b.x) // pix * pix, int(b.y) // pix * pix
        pygame.draw.circle(surface, BUBBLE, (cx, cy), b.radius, 1)
        _px(surface, cx - b.radius, cy - b.radius, 1, (230, 245, 255))
