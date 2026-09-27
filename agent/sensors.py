import math

from sim.geometry import cast_ray

RAY_OFFSETS = [math.radians(a) for a in (-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90)]  # relative to heading
OBSERVATION_SIZE = len(RAY_OFFSETS) + 3


def ray_distances(car, track, cfg):
    """Distance from the car's centre to the nearest road edge along each sensor ray, capped at RAY_MAX.

    Inputs:  car (Car); track (Track); cfg (Config)

    Outputs: list[float] - one distance per RAY_OFFSETS entry, in world units
    """
    pos = (car.x, car.y)
    return [cast_ray(pos, car.heading + offset, track.walls, cfg.ray_max) for offset in RAY_OFFSETS]


def angle_to(car, target):
    """Signed angle from the car's heading to a point, wrapped to [-pi, pi] (positive = to the right on screen).

    Inputs:  car (Car); target ((x, y))

    Outputs: float - radians
    """
    bearing = math.atan2(target[1] - car.y, target[0] - car.x)
    return (bearing - car.heading + math.pi) % (2 * math.pi) - math.pi


def observe(car, progress, track, cfg):
    """Build the agent's observation (spec 4): 9 rays, speed, direction to the next checkpoint, out-of-bounds flag.

    Inputs:  car (Car); progress (Progress) - next checkpoint and out-of-bounds state; track (Track); cfg (Config)

    Outputs: list[float] - OBSERVATION_SIZE values, each in [-1, 1]
    """
    (x1, y1), (x2, y2) = track.checkpoints[progress.next_checkpoint]
    rays = [d / cfg.ray_max for d in ray_distances(car, track, cfg)]
    return rays + [
        car.speed / cfg.max_speed,
        angle_to(car, ((x1 + x2) / 2, (y1 + y2) / 2)) / math.pi,
        # From outside the road the rays still just see an edge, so the agent needs this to know which side it's on.
        1.0 if progress.out_of_bounds else 0.0,
    ]
