"""Pure geometry helpers shared by the simulation and the renderers."""

import math

TWO_PI = 2.0 * math.pi


def clamp(value: float, low: float, high: float) -> float:
    """Restrict value to [low, high]."""
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    """Fold an angle into (-pi, pi]."""
    return (angle + math.pi) % TWO_PI - math.pi


def turn_toward(current: float, target: float, max_step: float) -> float:
    """Move current angle toward target by at most max_step radians."""
    diff = wrap_angle(target - current)
    return current + clamp(diff, -max_step, max_step)


def angle_sector(angle: float, n_sectors: int) -> int:
    """Map an angle (radians) to one of n_sectors around the circle."""
    return int((angle + math.pi / n_sectors) // (TWO_PI / n_sectors)) % n_sectors


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between two points."""
    return math.hypot(bx - ax, by - ay)


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: t=0 -> a, t=1 -> b."""
    return a + (b - a) * t
