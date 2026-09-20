# Task-5 / Task-6 / Task-8 training GIFs (DCC, DCC+BC)

Date: 2026-09-16
Status: Task 5/6 launched as SLURM `17982404` (array 0–2). Task 8 launched
as SLURM `17988666` (array 3–4).

## Motivation

Need visual evidence of:

1. **DCC without Success-BC on Task 5** (`sawyer_handle_press_side`): the
   policy that first reaches the handle, and how later rollouts look after
   collapse.
2. **DCC with Success-BC on Task 5**: whether BC keeps a press throughout
   the 1M budget.
3. **DCC on Task 6** (`sawyer_push`): a task where contrastive propagation
   works, as a visual control.
4. **DCC without / with Success-BC on Task 8** (`sawyer_window_close`):
   the other short-contact latch in the 10-task sequence.

GIFs (320×240, ≤80 frames) instead of mp4 to keep disk small.

## What is recorded

| cell | source | GIFs |
|---|---|---|
| 0 `handle_nobc_gifs` | existing seed-6 mid-ckpts from job `17901349` | 10 even steps (100k…1M) **plus** a first-success hunt at ~150k |
| 1 `push_gifs` | same job, push dir | 10 even steps |
| 2 `handle_bc_train` | new 1-seed 1M train, `success_bc_weight=0.1`, `terminal_episode` | paced eval GIFs every 100k, extra hunt on first eval with success>0, then mid-ckpt export |
| 3 `window_nobc_gifs` | existing seed-6 mid-ckpts from job `17954891` | 10 even steps **plus** a first-success hunt at ~250k (first eval 60%) |
| 4 `window_bc_train` | new 1-seed 1M Task-8 Success-BC | same protocol as cell 2 |

Even steps map to existing snapshots
`100200, 200400, …, 951900`.

The first-success GIF is the **first successful eval episode** found at the
checkpoint / eval where success first appears. It is **not** the original
training-episode frame buffer (those frames were never rendered). Up to 20
eval resets are tried.

In-training BC GIFs are written under that run’s `eval_videos/` and only
copied into `gifs/<variant>/` after the 1M job finishes.

## Handle geometry (why the arm can clip through)

This is MetaWorld’s collision setup, not a broken joint in our wrapper.

**What the policy is told**

| quantity | xyz (m) | meaning |
|---|---|---|
| mechanism target `_goal` / `_target_pos` | `[-0.07, 0.68, 0.07]` | pressed **handleStart** position; success is \(\lvert z-0.07\rvert\le 0.02\) |
| corrected desired **hand** | `[-0.066, 0.704, 0.100]` | captured from a successful scripted transition |
| corrected desired **handle** | `[-0.071, 0.708, 0.075]` | handleStart at that same successful state |
| gripper | `0.297` | slightly open |

The desired hand is **almost** stacked on the handle in \(xy\) (\(\Delta x\approx 5\,\mathrm{mm}\), \(\Delta y\approx -4\,\mathrm{mm}\)) and only **2.5 cm above** it. It is **not** a 20 cm hover waypoint. MetaWorld’s scripted policy *does* first go 20 cm above `handleStart`, then commands `handleStart + (0,0,-0.5)`, i.e. straight **down through** the handle. Success never checks that the TCP stayed outside the mesh.

**Sites that exist in the XML** (`handle_press.xml` on the moving `handle_link`):

| site | local xyz on `handle_link` | used in our obs? |
|---|---|---|
| `handleStart` | `(0, -0.166, 0.022)` | **yes** — this is `_get_pos_objects()` / state `[4:7]` |
| `handleRight` | `(0.05, -0.166, 0.014)` | no — 5 cm to the grip’s side |
| `handleCenter` | `(0, -0.12, 0.0)` | no — lever center |
| `goalPress` | box frame `(0, -0.216, 0.075)` | native `_target_pos`; we overwrite with the fixed `[-0.07, 0.68, 0.07]` |

So we **do** have an edge-like coordinate (`handleRight`), but the wrapper never exposes it. The arm is not given a “press the rim” target.

**Why 穿模 / clipping is visible**

- Decorative handle meshes (`handle_press_grip`, lever, trim, base) have
  `contype=0 conaffinity=0`: **no contact**.
- Collision is a thinner cylinder (`size="0.022 0.07"`) plus a flat box on
  the lever. The red visual grip is larger than that cylinder.
- Sawyer’s TCP is mocap-welded; contacts with `hdlprs_col`
  (`conaffinity=1 contype=0`) are approximate.
- Official success is only handle **z**. A trajectory that sinks the
  gripper through the visual handle still counts if `handleStart.z` is
  within 2 cm of 0.07.

The legacy (pre-correction) hand goal was even more stacked:
`_goal + (0, 0, 0.03)` = 3 cm vertically above the pressed handle target.

## Code / config

- `contrastive/eval_video.py`: optional `observe_*`, `record_until_success`,
  `save_gif` / `downsample_frames`.
- `run_continual_contrastive.py`: write `log_dir/eval_videos/*.gif` on paced
  evals; `--eval_video_first_success` hunts a successful episode the first
  time `eval_success_rate > 0`.
- `scripts/record_checkpoint_rollout_gifs.py`: offline export from
  `task_0_step_*.pkl` (handle, push, window).
- `experiment_configs_jubail_task_videos.py`
- `DRAFT_jubail_task_videos.sh` (array 0–2), `DRAFT_jubail_task8_videos.sh`
  (array 3–4)
- `tests/test_eval_video.py`, `tests/test_jubail_task_videos.py`

Shared train flags match the action-advice diagnosis cell: decomposed DCC,
actor reset, full net, corrected wrapper, `dyn_aux_weight=1`, 1024×4, seed 6.

## Launch

```bash
sbatch DRAFT_jubail_task_videos.sh      # Task 5 / 6
sbatch DRAFT_jubail_task8_videos.sh     # Task 8
```

Outputs:

- GIFs: `logs/jubail_task_videos/gifs/{handle_nobc_gifs,push_gifs,handle_bc_train,window_nobc_gifs,window_bc_train}/`
- BC train videos also under that run’s `eval_videos/`
- stdout: `logs/jubail_task_videos/runs/%A_%a.out`
- W&B group `TASK58-JUBAIL-TASK-VIDEOS` (train cells)

## Validation

```bash
python tests/test_eval_video.py
python tests/test_jubail_task_videos.py
```

## Known limitations

- Headless EGL MuJoCo 640×480 then downsampled; camera is the default
  Sawyer view, not a custom press close-up.
- Deterministic policy mean; success hunt uses env reset stochasticity.
- BC cells are **new** 1M seed-6 runs on the diagnosis stack, not the
  10-task paper BC curriculum.
- If a first-success hunt misses, the GIF is tagged `first_success_miss`.
