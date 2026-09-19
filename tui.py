"""TUI viewer: watch the creature learn in your terminal.

Rendering only — all simulation and session logic lives in `sim`.
"""

import time

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from sim import DIR_ARROW_BY_SECTOR, N_DIRECTIONS, World
from sim import session
from sim.geometry import angle_sector
from sim.world import WIDTH, HEIGHT

DEFAULT_CELL = "grey19"
WALL_CELL = "grey42"
FOOD_CELL = "color(46)"
HAPPY_CELL = "color(201)"
HUNGRY_CELL = "color(129)"
STARVING_CELL = "color(57)"


def mood_style(hunger: float) -> str:
    """Cell color by hunger level."""
    if hunger < 33:
        return HAPPY_CELL
    if hunger < 66:
        return HUNGRY_CELL
    return STARVING_CELL


def frame_grid(world: World, width: int, height: int) -> list[list[str]]:
    """Tank cells as characters: walls, food, trail and the creature."""
    grid = [[" "] * width for _ in range(height)]
    for x in range(width):
        grid[0][x] = grid[height - 1][x] = "─"
    for y in range(height):
        grid[y][0] = grid[y][width - 1] = "│"
    grid[0][0] = grid[height - 1][width - 1] = "└"
    grid[0][width - 1] = "┘"
    grid[height - 1][0] = "┌"
    grid[height - 1][width - 1] = "┐"

    for f in world.food:
        px, py = round(f.pos.x), round(f.pos.y)
        if 0 <= px < width and 0 <= py < height:
            grid[py][px] = "◆"
    for p in world.trail:
        px, py = round(p.x), round(p.y)
        if 0 <= px < width and 0 <= py < height and grid[py][px] == " ":
            grid[py][px] = "·"
    px, py = round(world.position.x), round(world.position.y)
    if 0 <= px < width and 0 <= py < height:
        sector = angle_sector(world.heading, N_DIRECTIONS)
        grid[py][px] = DIR_ARROW_BY_SECTOR[sector]
    return grid


def render_frame(world: World, width: int, height: int) -> Text:
    """Colorize the grid and flatten it into one rich Text."""
    grid = frame_grid(world, width, height)
    frame = Text()
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            style = cell_style(world, grid, x, y)
            frame.append(cell, style=style)
        frame.append("\n")
    return frame


def cell_style(world: World, grid: list[list[str]], x: int, y: int) -> str:
    """Style of one cell based on what it contains."""
    cell = grid[y][x]
    if cell in "┌┐└┘─│":
        return WALL_CELL
    if cell == "◆":
        return FOOD_CELL
    if cell == "·":
        return mood_style(world.hunger)
    if cell != " ":
        return mood_style(world.hunger)  # the creature itself
    return DEFAULT_CELL


def stats_panel(world: World) -> Text:
    """Hunger bar and lifetime counters under the tank."""
    bar_len = 20
    filled = int(world.hunger / 100 * bar_len)
    bar = "▰" * filled + "▱" * (bar_len - filled)
    txt = Text()
    txt.append(" hunger ", style="bold")
    txt.append(f"{bar} {world.hunger:5.1f}\n", style=mood_style(world.hunger))
    txt.append(f" ticks: {world.ticks}   food eaten: {world.food_eaten}   "
               f"wall bumps: {world.wall_bumps}   starvations: {world.starvations}\n")
    txt.append(f" avg reward ({len(world.recent_rewards)}t): "
               f"{world.recent_avg_reward:+.3f}   "
               f"experience: {world.brain.experience}   "
               f"known states: {world.brain.known_states}\n")
    txt.append(f" epsilon (curiosity): {world.brain.config.epsilon:.3f}")
    return txt


def main() -> None:
    """Entry point: parse args, load the brain, run the render loop, save."""
    args = session.parse_args("Creature TUI", default_speed=2)
    console = Console()
    brain = session.open_brain(args.fresh)
    world = World(brain=brain)

    speed = max(1, args.speed)
    frame_interval = 1.0 / 8.0
    try:
        with Live(console=console, refresh_per_second=8, screen=True) as live:
            while True:
                frame_start = time.monotonic()
                for _ in range(speed):
                    world.step()
                frame = render_frame(world, int(WIDTH), int(HEIGHT))
                live.update(Group(stats_panel(world), frame))
                elapsed = time.monotonic() - frame_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        session.save_brain(brain)
        console.print(f"[green]Brain saved to {session.BRAIN_PATH}[/green]")


if __name__ == "__main__":
    main()
