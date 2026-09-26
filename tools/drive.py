"""Drive the car around the configured track with the keyboard (the road edges don't block; leaving the road is flagged).

Usage (from the repo root): python3 -m tools.drive
Controls: W/S or Up/Down throttle/brake-reverse, A/D or Left/Right steer, V toggle sensor rays, R restart, Esc quit.
"""

import math

import pygame

from sim.config import load_config, load_track
from sim.geometry import cast_ray
from sim.physics import Car, corners
from sim.race import Progress, race_step

MAX_WINDOW = (1400, 850)
MARGIN = 20
HUD_HEIGHT = 30
RAY_OFFSETS = [math.radians(a) for a in (-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90)]  # spec 4

BG, WALL, CP, CP_NEXT, FINISH = (43, 43, 43), (235, 235, 235), (40, 90, 70), (250, 200, 50), (220, 50, 50)
CAR, CAR_OUT, NOSE, RAY, HIT = (60, 150, 255), (230, 60, 60), (255, 255, 255), (120, 120, 160), (255, 110, 110)
TEXT, TEXT_OUT = (220, 220, 220), (255, 90, 90)


class View:
    """Maps world coordinates to screen pixels, scaling the track down if it doesn't fit MAX_WINDOW."""

    def __init__(self, walls):
        xs = [p[0] for w in walls for p in w]
        ys = [p[1] for w in walls for p in w]
        self.min_x, self.min_y = min(xs), min(ys)
        w, h = max(xs) - self.min_x, max(ys) - self.min_y
        self.scale = min(1.0, (MAX_WINDOW[0] - 2 * MARGIN) / w, (MAX_WINDOW[1] - 2 * MARGIN - HUD_HEIGHT) / h)
        self.size = (round(w * self.scale) + 2 * MARGIN, round(h * self.scale) + 2 * MARGIN + HUD_HEIGHT)

    def __call__(self, p):
        return (
            MARGIN + (p[0] - self.min_x) * self.scale,
            HUD_HEIGHT + MARGIN + (p[1] - self.min_y) * self.scale,
        )


def read_input(keys):
    """Map held keys to the spec 3 action pair.

    Inputs:  keys (pygame.key.ScancodeWrapper) - result of pygame.key.get_pressed()

    Outputs: (int, int) - throttle and steer, each -1, 0 or +1
    """
    throttle = (keys[pygame.K_w] or keys[pygame.K_UP]) - (keys[pygame.K_s] or keys[pygame.K_DOWN])
    # Heading grows clockwise on screen (y points down), so +1 steer turns right.
    steer = (keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT])
    return throttle, steer


def fmt_time(seconds):
    return "--" if seconds is None else f"{seconds:.2f}s"


def draw(screen, font, view, cfg, track, car, progress, stats, show_rays):
    """Render the track, car, optional sensor rays and the HUD.

    Inputs:  screen (Surface); font (Font); view (View); cfg, track, car, progress - sim state;
             stats (dict) - current/last/best lap times in seconds and out-of-bounds count; show_rays (bool)

    Outputs: None - draws onto screen
    """
    screen.fill(BG)
    last = len(track.checkpoints) - 1
    for i, (a, b) in enumerate(track.checkpoints):
        color = CP_NEXT if i == progress.next_checkpoint else FINISH if i == last else CP
        pygame.draw.line(screen, color, view(a), view(b), 3 if i == progress.next_checkpoint else 1)
    for a, b in track.walls:
        pygame.draw.line(screen, WALL, view(a), view(b), 2)

    pos = (car.x, car.y)
    if show_rays:
        for offset in RAY_OFFSETS:
            angle = car.heading + offset
            d = cast_ray(pos, angle, track.walls, cfg.ray_max)
            end = (car.x + math.cos(angle) * d, car.y + math.sin(angle) * d)
            pygame.draw.line(screen, RAY, view(pos), view(end), 1)
            if d < cfg.ray_max:
                pygame.draw.circle(screen, HIT, view(end), 3)

    body = [view(p) for p in corners(car, cfg)]
    pygame.draw.polygon(screen, CAR_OUT if progress.out_of_bounds else CAR, body)
    pygame.draw.line(screen, NOSE, body[0], body[1], 3)  # front edge

    hud = (
        f"speed {car.speed:6.1f} / {cfg.max_speed}   lap {progress.laps + 1}   "
        f"checkpoint {progress.next_checkpoint}/{len(track.checkpoints)}   "
        f"time {fmt_time(stats['current'])}   last {fmt_time(stats['last'])}   best {fmt_time(stats['best'])}   "
        f"out of bounds x{stats['outs']}   [V] rays  [R] restart"
    )
    screen.blit(font.render(hud, True, TEXT), (MARGIN, 8))
    if progress.out_of_bounds:
        screen.blit(font.render("OUT OF BOUNDS", True, TEXT_OUT), (MARGIN, HUD_HEIGHT + 4))


def main():
    cfg = load_config()
    track = load_track(cfg.track)
    view = View(track.walls)

    pygame.init()
    screen = pygame.display.set_mode(view.size)
    pygame.display.set_caption(f"RL Racer - {cfg.track}")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    def reset():
        return Car.at(track.start_poses[0]), Progress(), {"current": 0.0, "last": None, "best": None, "outs": 0}

    car, progress, stats = reset()
    show_rays, accumulator, running = True, 0.0, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                show_rays = not show_rays
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                car, progress, stats = reset()

        # Fixed-step accumulator (spec 2): physics always advances in exact cfg.dt steps, whatever the frame rate.
        accumulator += min(clock.tick(60) / 1000, 0.25)
        while accumulator >= cfg.dt:
            throttle, steer = read_input(pygame.key.get_pressed())
            result = race_step(car, progress, throttle, steer, cfg, track)
            stats["current"] += cfg.dt
            stats["outs"] += result.went_out
            if result.lap:
                stats["last"] = stats["current"]
                stats["best"] = min(stats["best"] or math.inf, stats["current"])
                stats["current"] = 0.0
            accumulator -= cfg.dt

        draw(screen, font, view, cfg, track, car, progress, stats, show_rays)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
