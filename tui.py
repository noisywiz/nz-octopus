"""TUI viewer: watch the octopus learn in your terminal."""

import argparse
import math
import time
from pathlib import Path

from sim import QBrain, World, DIRS

try:
    import rich.live
    import rich.table
    import rich.text
    from rich.console import Console
except ImportError as e:
    raise SystemExit("rich is required: pip install rich") from e

BRAIN_PATH = Path(__file__).resolve().parent / "brain.json"

DEFAULT_CELL = "grey19"
WALL_CELL = "grey42"
FOOD_CELL = "color(46)"
HAPPY_CELL = "color(201)"
HUNGRY_CELL = "color(129)"
STARVING_CELL = "color(57)"


def mood_style(hunger: float) -> str:
    if hunger < 33:
        return HAPPY_CELL
    if hunger < 66:
        return HUNGRY_CELL
    return STARVING_CELL


def render_frame(world: World, width: int, height: int) -> rich.text.Text:
    """Draw the tank 1:1: one world unit = one terminal cell."""
    grid = [[" " for _ in range(width)] for _ in range(height)]
    styles = [[DEFAULT_CELL for _ in range(width)] for _ in range(height)]

    # walls
    for x in range(width):
        for y in (0, height - 1):
            grid[y][x] = "─"
            styles[y][x] = WALL_CELL
    for y in range(height):
        for x in (0, width - 1):
            grid[y][x] = "│"
            styles[y][x] = WALL_CELL
    grid[0][0] = grid[0][width - 1] = "┌"
    grid[height - 1][0] = grid[height - 1][width - 1] = "└"
    styles[0][0] = styles[0][width - 1] = styles[height - 1][0] = styles[height - 1][width - 1] = WALL_CELL

    # food
    for f in world.foods:
        px, py = round(f.x), round(f.y)
        if 0 <= px < width and 0 <= py < height:
            grid[py][px] = "◆"
            styles[py][px] = FOOD_CELL

    # trail: fading path of recent positions
    for i, (tx, ty) in enumerate(world.trail):
        px, py = round(tx), round(ty)
        if 0 <= px < width and 0 <= py < height and grid[py][px] == " ":
            grid[py][px] = "·"
            styles[py][px] = mood_style(world.hunger)

    # octopus
    px, py = round(world.octopus_x), round(world.octopus_y)
    if 0 <= px < width and 0 <= py < height:
        sector = int((world.octopus_dir + math.pi / DIRS) // (2 * math.pi / DIRS)) % DIRS
        arrows_by_sector = ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"]
        grid[py][px] = arrows_by_sector[sector]
        styles[py][px] = mood_style(world.hunger)

    frame = rich.text.Text()
    for row_cells, row_styles in zip(grid, styles):
        for cell, style in zip(row_cells, row_styles):
            frame.append(cell, style=style)
        frame.append("\n")
    return frame


def stats_panel(world: World) -> rich.text.Text:
    hunger_bar_len = 20
    filled = int(world.hunger / 100 * hunger_bar_len)
    bar = "▰" * filled + "▱" * (hunger_bar_len - filled)
    txt = rich.text.Text()
    txt.append(" hunger ", style="bold")
    txt.append(f"{bar} {world.hunger:5.1f}\n", style=mood_style(world.hunger))
    txt.append(f" ticks: {world.ticks}   food eaten: {world.food_eaten}   "
               f"wall bumps: {world.wall_bumps}\n")
    txt.append(f" avg reward (500t): {world.recent_avg_reward:+.3f}   "
               f"experience: {world.brain.experience}   "
               f"known states: {world.brain.known_states}\n")
    txt.append(f" epsilon (curiosity): {world.brain.epsilon:.3f}")
    return txt


def main():
    parser = argparse.ArgumentParser(description="Octopus TUI")
    parser.add_argument("--speed", type=int, default=2, help="sim ticks per frame")
    parser.add_argument("--fps", type=float, default=8.0, help="render fps")
    parser.add_argument("--fresh", action="store_true", help="start with empty brain")
    args = parser.parse_args()

    console = Console()
    if args.fresh and BRAIN_PATH.exists():
        BRAIN_PATH.unlink()
    brain = QBrain(n_states=0, n_actions=0, load_path=BRAIN_PATH)
    world = World(brain)

    frame_interval = 1.0 / args.fps
    with rich.live.Live(console=console, refresh_per_second=args.fps, screen=True) as live:
        try:
            while True:
                frame_start = time.monotonic()
                for _ in range(args.speed):
                    world.step()
                # tank is 80x30 world units -> draw 1:1, need a big enough terminal
                frame = render_frame(world, int(World.WIDTH), int(World.HEIGHT))
                live.update(rich.console.Group(stats_panel(world), frame))
                # actually respect fps: the loop would otherwise spin at max speed
                elapsed = time.monotonic() - frame_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        except KeyboardInterrupt:
            pass
        finally:
            brain.save(BRAIN_PATH)
            console.print(f"[green]Brain saved to {BRAIN_PATH}[/green]")


if __name__ == "__main__":
    main()
