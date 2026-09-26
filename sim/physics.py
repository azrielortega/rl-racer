import math
from dataclasses import dataclass


@dataclass
class Car:
    x: float
    y: float
    heading: float  # radians, 0 = +x, y points down (screen coordinates)
    speed: float = 0.0

    @classmethod
    def at(cls, pose):
        """Create a stationary car at a start pose.

        Inputs:  pose (Pose) - x, y, heading from the track

        Outputs: Car - car with speed 0
        """
        return cls(pose.x, pose.y, pose.heading)


def corners(car, cfg):
    """The four corners of the car's rectangle (CAR_LENGTH along the heading, CAR_WIDTH across).

    Inputs:  car (Car); cfg (Config)

    Outputs: list[(x, y)] - front-left, front-right, back-right, back-left (in screen terms)
    """
    fx, fy = math.cos(car.heading) * cfg.car_length / 2, math.sin(car.heading) * cfg.car_length / 2
    sx, sy = -math.sin(car.heading) * cfg.car_width / 2, math.cos(car.heading) * cfg.car_width / 2
    return [
        (car.x + fx - sx, car.y + fy - sy),
        (car.x + fx + sx, car.y + fy + sy),
        (car.x - fx + sx, car.y - fy + sy),
        (car.x - fx - sx, car.y - fy - sy),
    ]


def step(car, throttle, steer, cfg):
    """Advance the car by one fixed physics step (cfg.dt), in place.

    Inputs:  car (Car); throttle, steer (int) - each -1, 0 or +1; cfg (Config) - physics constants

    Outputs: None - updates car.x, car.y, car.heading, car.speed
    """
    dt = cfg.dt
    car.speed += throttle * cfg.accel * dt
    car.speed *= 1 - cfg.drag * dt
    car.speed = min(max(car.speed, -cfg.max_reverse), cfg.max_speed)
    # Turning circle: tight (steering lock) when slow, grip-limited (grows with speed²) when fast.
    # Turn rate = speed / radius, so a stopped car can't spin in place and reversing steers the other way.
    radius = max(cfg.min_turn_radius, car.speed * car.speed / cfg.grip)
    car.heading += steer * (car.speed / radius) * dt
    car.x += math.cos(car.heading) * car.speed * dt
    car.y += math.sin(car.heading) * car.speed * dt
