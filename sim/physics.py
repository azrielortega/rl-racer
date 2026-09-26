"""Kinematic car model (spec §1). Must stay identical to the TypeScript port."""

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


def step(car, throttle, steer, cfg):
    """Advance the car by one fixed physics step (cfg.dt), in place.

    Inputs:  car (Car); throttle, steer (int) - each -1, 0 or +1; cfg (Config) - physics constants

    Outputs: None - updates car.x, car.y, car.heading, car.speed
    """
    dt = cfg.dt
    car.speed += throttle * cfg.accel * dt
    car.speed *= 1 - cfg.drag * dt
    car.speed = min(max(car.speed, -cfg.max_reverse), cfg.max_speed)
    # Turning scales with speed so a stopped car can't spin in place (an exploit the agent would find).
    car.heading += steer * cfg.turn_rate * (car.speed / cfg.max_speed) * dt
    car.x += math.cos(car.heading) * car.speed * dt
    car.y += math.sin(car.heading) * car.speed * dt
