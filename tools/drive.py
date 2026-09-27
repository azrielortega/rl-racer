"""Drive the car around the configured track with the keyboard (the road edges don't block; leaving the road is flagged).

Usage (from the repo root): python3 -m tools.drive [--model runs/ppo/final.zip]
Controls: W/S or Up/Down throttle/brake-reverse, A/D or Left/Right steer, V toggle sensor rays, R restart, Esc quit.
With --model the trained agent drives instead, and each run ends like a training episode (lap, out of bounds, stall).
"""

import argparse
import math

import numpy as np
import pygame

from agent.env import STALL_STEPS, decode_action
from agent.sensors import RAY_OFFSETS, observe, ray_distances
from sim.config import load_config, load_track
from sim.physics import Car, corners
from sim.race import Progress, race_step

MAX_WINDOW = (1400, 850)
MARGIN = 20
HUD_HEIGHT = 30

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
        for offset, d in zip(RAY_OFFSETS, ray_distances(car, track, cfg)):
            angle = car.heading + offset
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
    if stats["ended"]:
        hud += f"   agent runs {stats['runs']}, last ended: {stats['ended']}"
    screen.blit(font.render(hud, True, TEXT), (MARGIN, 8))
    if progress.out_of_bounds:
        screen.blit(font.render("OUT OF BOUNDS", True, TEXT_OUT), (MARGIN, HUD_HEIGHT + 4))


def main():
    parser = argparse.ArgumentParser(description="Drive the car, or watch a trained agent drive it.")
    parser.add_argument("--model", help="trained PPO .zip; the agent drives instead of the keyboard")
    args = parser.parse_args()
    policy = None
    if args.model:
        from stable_baselines3 import PPO  # only needed to watch the agent; keeps keyboard mode free of torch

        policy = PPO.load(args.model, device="cpu")

    cfg = load_config()
    track = load_track(cfg.track)
    view = View(track.walls)

    pygame.init()
    screen = pygame.display.set_mode(view.size)
    pygame.display.set_caption(f"RL Racer - {cfg.track}")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    def reset():
        stats = {"current": 0.0, "last": None, "best": None, "outs": 0, "runs": 0, "ended": None}
        return Car.at(track.start_pose), Progress(), stats

    car, progress, stats = reset()
    show_rays, accumulator, running = True, 0.0, True
    # Agent mode: physics steps into the current run, agent steps since the last checkpoint, the held action.
    tick, since_checkpoint, held = 0, 0, (0, 0)

    def end_run(reason):
        """Start the agent's next run from the grid, keeping the lap stats (mirrors an env episode ending)."""
        nonlocal car, progress, tick, since_checkpoint
        stats["runs"] += 1
        stats["ended"] = reason
        stats["current"] = 0.0
        car, progress = Car.at(track.start_pose), Progress()
        tick, since_checkpoint = 0, 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                show_rays = not show_rays
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                car, progress, stats = reset()
                tick, since_checkpoint = 0, 0

        # Fixed-step accumulator (spec 2): physics always advances in exact cfg.dt steps, whatever the frame rate.
        accumulator += min(clock.tick(60) / 1000, 0.25)
        while accumulator >= cfg.dt:
            accumulator -= cfg.dt
            if policy and tick % cfg.frame_skip == 0:
                if since_checkpoint >= STALL_STEPS:
                    end_run("stalled")
                # Same as training: a fresh observation every FRAME_SKIP physics steps, action held in between.
                obs = np.array(observe(car, progress, track, cfg), dtype=np.float32)
                held = decode_action(int(policy.predict(obs, deterministic=True)[0]))
                since_checkpoint += 1
            throttle, steer = held if policy else read_input(pygame.key.get_pressed())
            result = race_step(car, progress, throttle, steer, cfg, track)
            tick += 1
            stats["current"] += cfg.dt
            stats["outs"] += result.went_out
            if result.checkpoints:
                since_checkpoint = 0
            if result.lap:
                stats["last"] = stats["current"]
                stats["best"] = min(stats["best"] or math.inf, stats["current"])
                stats["current"] = 0.0
            if policy and (result.went_out or result.lap):
                end_run("out of bounds" if result.went_out else f"lap {stats['last']:.2f}s")

        draw(screen, font, view, cfg, track, car, progress, stats, show_rays)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
