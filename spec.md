# RL Racer: Design Spec

Top-down racing game against a PPO agent. It's trained in Python (Stable-Baselines3 + Gymnasium) and runs in the browser (TypeScript). Physics must be identical in both.

## 1. Physics model: pure kinematic

**State:** `x, y, heading, speed`

```
speed   += throttle * ACCEL * dt
speed   *= (1 - DRAG * dt)
speed    = clamp(speed, -MAX_REVERSE, MAX_SPEED)
heading += steer * TURN_RATE * (speed / MAX_SPEED) * dt
x       += cos(heading) * speed * dt
y       += sin(heading) * speed * dt
```

- The turn is scaled by `speed / MAX_SPEED` so a stopped car can't spin in place. Without this, the agent tends to find spinning as an exploit.
- Constants (`ACCEL`, `DRAG`, `TURN_RATE`, `MAX_SPEED`, `MAX_REVERSE`) go in one shared `config.json` that both Python and TS load, so they can't drift apart.
- The car is a circle for collisions (radius `CAR_RADIUS`), which makes wall checks a simple distance-to-segment test.

## 2. Timestep: 60 Hz physics, agent acts every 4 steps

- `dt = 1/60`, fixed. In the browser, use a fixed-step accumulator loop, not raw `requestAnimationFrame` deltas.
- **Frame skip = 4:** the agent picks an action, it's held for 4 physics steps, and the rewards from those steps are summed. That's 15 decisions/sec, close to human reaction speed.
- The human player reads input every physics step.

## 3. Action space: Discrete(9)

| idx | throttle | steer |   | idx | throttle | steer |
|---|---|---|---|---|---|---|
| 0 | -1 | -1 | | 5 | 0 | +1 |
| 1 | -1 | 0 | | 6 | +1 | -1 |
| 2 | -1 | +1 | | 7 | +1 | 0 |
| 3 | 0 | -1 | | 8 | +1 | +1 |
| 4 | 0 | 0 | | | | |

`throttle = idx // 3 - 1`, `steer = idx % 3 - 1`. The keyboard maps onto the same pair: W/S for throttle, A/D for steering.

## 4. Observation vector: 11 floats

```
[ ray_-90, ray_-67.5, ray_-45, ray_-22.5, ray_0,
  ray_+22.5, ray_+45, ray_+67.5, ray_+90,     # dist / RAY_MAX, 0..1
  speed / MAX_SPEED,                           # -1..1
  angle_to_next_cp / π ]                       # -1..1, wrapped
```

- Ray angles are relative to the car's heading, and each ray is capped at `RAY_MAX`.
- `angle_to_next_cp` is measured from the car's heading to the midpoint of the next checkpoint and wrapped to [-π, π].

## 5. Reward

| Event | Reward |
|---|---|
| Pass the next checkpoint (in order) | **+1** |
| Complete a lap | **+10** |
| Each agent step | **-0.01** |
| Crash (training only) | **-5**, episode ends |

- A checkpoint only counts if it's the *next* one in order, so reversing and re-crossing earns nothing.

## 6. Collision

- **Training:** touching a wall gives -5 and ends the episode.
- **Game:** the car is pushed out of the wall and bounces back, with `speed *= -0.3`, and the race continues. The agent never trained on a bounce, so it will just recover and drive on.

## 7. Episode termination (training)

- **Success:** 2 laps completed.
- **Crash:** hit a wall.
- **Stall:** no new checkpoint reached within about 5 seconds (75 agent steps).
- Use `terminated` for a crash or finishing, and `truncated` for a stall. SB3 treats them differently when bootstrapping value estimates.

## 8. Track format: centerline + width, generated

You only author the centerline (in drive order) and a road width. `tools/gen_track.py` generates everything else:

```json
// tracks/<name>.centerline.json  (authored)
{ "width": 80, "checkpoint_spacing": 100, "segment_length": 10,
  "centerline": [[100,300],[250,120],[500,100],[700,250],[650,450],[400,500],[200,480]] }
```

```
python3 tools/gen_track.py tracks/<name>.centerline.json -o tracks/<name>.json --svg preview.svg
```

1. **Smooth:** closed centripetal Catmull-Rom spline through the points, resampled every `segment_length` units.
2. **Walls:** the centerline offset ±`width/2`, output as a list of segments `[[x1,y1],[x2,y2]]`.
3. **Checkpoints:** ordered segments across the road every ~`checkpoint_spacing` units. The last one is the finish line, which is also the start line.
4. **Start poses:** two slots side by side on the start line, facing along the centerline (`x, y, heading`).

The generated `tracks/<name>.json` holds `walls`, `checkpoints`, `start_poses`, `centerline`, `width`, `length`, and `min_corner_radius`. Python and TS both load this file, and only Python generates it. The script exits non-zero and warns if a corner is tighter than `width/2` or a wall folds back or crosses another wall.

Design tips: keep `width` above about 6× `CAR_RADIUS` and the tightest corner radius above the car's turning radius at speed. The track in use is `tracks/monza.json` (a simplified Monza: chicanes removed, tightest corner radius 48), set by `TRACK` in `config.json`.
