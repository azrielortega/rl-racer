# RL Racer: Design Spec

Top-down qualifying game: the player sets a lap time and tries to beat a PPO agent's lap. The agent is trained in Python (Stable-Baselines3 + Gymnasium). The browser game (TypeScript) runs the same physics for both cars, and the agent drives live in the browser through an ONNX export of its policy (spec 9). Physics and sensors must be identical in both.

## 1. Physics model: pure kinematic

**State:** `x, y, heading, speed`

```
speed   += throttle * ACCEL * dt
speed   *= (1 - DRAG * dt)
speed    = clamp(speed, -MAX_REVERSE, MAX_SPEED)
radius   = max(MIN_TURN_RADIUS, speed² / GRIP)
heading += steer * (speed / radius) * dt
x       += cos(heading) * speed * dt
y       += sin(heading) * speed * dt
```

- The turning circle is tight when slow (`MIN_TURN_RADIUS`, the steering lock) and grows with speed² when fast (`GRIP` is the maximum sideways acceleration), so tight corners need braking. With 40 / 450 the radius is 40 up to speed 134 and 200 at top speed 300.
- The turn rate is `speed / radius`, so a stopped car can't spin in place (the agent tends to find spinning as an exploit), and reversing steers the other way.
- Constants (`ACCEL`, `DRAG`, `MIN_TURN_RADIUS`, `GRIP`, `MAX_SPEED`, `MAX_REVERSE`) go in one shared `config.json` that both Python and TS load, so they can't drift apart.
- The car is a rectangle, `CAR_LENGTH` along its heading by `CAR_WIDTH` across, centred on `x, y`.

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

## 4. Observation vector: 12 floats

```
[ ray_-90, ray_-67.5, ray_-45, ray_-22.5, ray_0,
  ray_+22.5, ray_+45, ray_+67.5, ray_+90,     # dist / RAY_MAX, 0..1
  speed / MAX_SPEED,                           # -1..1
  angle_to_next_cp / π,                        # -1..1, wrapped
  out_of_bounds ]                              # 0 or 1
```

- Ray angles are relative to the car's heading, and each ray is capped at `RAY_MAX`.
- Rays start at the car's centre and measure the distance to the nearest road edge.
- `angle_to_next_cp` is measured from the car's heading to the midpoint of the next checkpoint and wrapped to [-π, π]. Positive means the checkpoint is to the right.
- `out_of_bounds` is needed because from outside the road the rays still just see an edge, so they can't tell inside from outside.

## 5. Reward

| Event | Reward |
|---|---|
| Pass the next checkpoint (in order) | **+1** |
| Complete a lap | **+10** |
| Each agent step | **-0.01** |
| Go out of bounds (voids the lap, ends the episode) | **-5** |

- A checkpoint only counts if it's the *next* one in order, so reversing and re-crossing earns nothing.

## 6. Out of bounds (no walls)

- The road edges don't block the car. The car is **out of bounds** as soon as any corner of its rectangle is farther than `width / 2` from the centerline, i.e. over a drawn edge.
- Each step reports `out_of_bounds` (off the road now) and `went_out` (left the road this step).
- In qualifying, going out voids the lap: training ends the episode with the -5 penalty on `went_out`, and in the game the player's lap time doesn't count.
- Checkpoints only span the road, so a car that cuts across off the road misses them and has to come back to earn progress.
- Same rules in training and in the game.

## 7. Episode termination (training)

- **Success:** 1 lap completed, from a standing start on the start line.
- **Out of bounds:** ends the episode with the -5 penalty, since leaving the road voids the lap.
- **Stall:** no new checkpoint reached within about 5 seconds (75 agent steps).
- Use `terminated` for finishing (or out of bounds, if it ends the episode), and `truncated` for a stall. SB3 treats them differently when bootstrapping value estimates.

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
3. **Checkpoints:** ordered segments across the road every ~`checkpoint_spacing` units. The last one is the finish line, which is also the start line. All checkpoints share one orientation, so a crossing counts only when `(q2 - q1) × (move) > 0` (forward).
4. **Start pose:** one slot in the middle of the start line, facing along the centerline (`x, y, heading`). The player and the agent both start there. The two cars pass through each other, so there's no second slot and no car-to-car collision.

The generated `tracks/<name>.json` holds `walls`, `checkpoints`, `start_pose`, `centerline`, `width`, `length`, and `min_corner_radius`. Python and TS both load this file, and only Python generates it. The script exits non-zero and warns if a corner is tighter than `width/2` or a wall folds back or crosses another wall.

Design tips: keep `width` above about 3× `CAR_LENGTH` and the tightest corner radius above the car's turning radius at speed. The track in use is `tracks/monza.json` (a simplified Monza: chicanes removed, tightest corner radius 48), set by `TRACK` in `config.json`.

## 9. Live agent in the browser (ONNX)

- `python -m tools.export_onnx <model.zip> -o web/agent` writes:
  - `web/agent.onnx`: the policy's actor network only. Input `obs` float32 `[1, 12]` (spec 4), output `logits` float32 `[1, 9]`. The action is the argmax of the logits, which is the same as SB3's `deterministic=True`. It's a single self-contained file (about 32KB, only `Gemm` and `Tanh` ops).
  - `web/agent.parity.json`: one deterministic run, recording for every agent step the car state, `next_checkpoint`, `out_of_bounds`, the observation, the logits and the action. The script checks that the ONNX action matches SB3 at every step before writing it.
- The browser runs the model with `onnxruntime-web`. Every `FRAME_SKIP` physics steps it builds the observation with the TS port of `sensors.py`, runs the model and holds the argmax action, exactly like training (spec 2). The agent's car uses the same TS physics as the player's.
- The TS sensors and physics must reproduce Python closely, or the agent sees inputs it never trained on and drives differently with no error. Check the port against `agent.parity.json`: from each recorded car state, the TS observation should match `obs` to about 1e-4, and the model should pick the same `action`.
