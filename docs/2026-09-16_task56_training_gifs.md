# Task-5 / Task-6 training GIFs (DCC, DCC+BC, push)

Date: 2026-09-16
Status: launched on Jubail (`sbatch DRAFT_jubail_task_videos.sh`, array 0–2)

## Motivation

Need visual evidence of:

1. **DCC without Success-BC on Task 5** (`sawyer_handle_press_side`): the
   policy that first reaches the handle, and how later rollouts look after
   collapse.
2. **DCC with Success-BC on Task 5**: whether BC keeps a press throughout
   the 1M budget.
3. **DCC on Task 6** (`sawyer_push`): a task where contrastive propagation
   works, as a visual control.

GIFs (320×240, ≤80 frames) instead of mp4 to keep disk small.

## What is recorded

| cell | source | GIFs |
|---|---|---|
| 0 `handle_nobc_gifs` | existing seed-6 mid-ckpts from job `17901349` | 10 even steps (100k…1M) **plus** a first-success hunt at ~150k (first eval with success>0 on that run) |
| 1 `push_gifs` | same job, push dir | 10 even steps |
| 2 `handle_bc_train` | new 1-seed 1M train, diagnosis stack + `success_bc_weight=0.1`, `terminal_episode` | paced eval GIFs every 100k during training, extra hunt on first eval with success>0, then the same 10+first-success export from mid-ckpts |

Even steps map to existing snapshots
`100200, 200400, …, 951900`.

The first-success GIF is the **first successful eval episode** found at the
checkpoint / eval where success first appears. It is **not** the original
training-episode frame buffer (that was never rendered). Up to 20 eval
resets are tried.

## Code / config

- `contrastive/eval_video.py`: optional `observe_*`, `record_until_success`,
  `save_gif` / `downsample_frames`.
- `run_continual_contrastive.py`: write `log_dir/eval_videos/*.gif` on paced
  evals; `--eval_video_first_success` hunts a successful episode the first
  time `eval_success_rate > 0`. Video cadence uses `next_eval_video_at`
  instead of `env_steps % interval` (episode-aligned steps were skipping
  videos).
- `scripts/record_checkpoint_rollout_gifs.py`: offline export from
  `task_0_step_*.pkl`.
- `experiment_configs_jubail_task_videos.py` / `DRAFT_jubail_task_videos.sh`
- `tests/test_eval_video.py`, `tests/test_jubail_task_videos.py`

Shared train flags match the action-advice diagnosis cell: decomposed DCC,
actor reset, full net, corrected wrapper, `dyn_aux_weight=1`, 1024×4, seed 6.

## Launch

```bash
sbatch DRAFT_jubail_task_videos.sh
```

Outputs:

- GIFs: `logs/jubail_task_videos/gifs/{handle_nobc_gifs,push_gifs,handle_bc_train}/`
- BC train videos also under that run’s `eval_videos/`
- stdout: `logs/jubail_task_videos/runs/%A_%a.out`
- W&B group `TASK58-JUBAIL-TASK-VIDEOS` (cell 2 only)

## Validation

```bash
python tests/test_eval_video.py
python tests/test_jubail_task_videos.py
```

## Known limitations

- Headless EGL MuJoCo 640×480 then downsampled; camera is the default
  Sawyer view, not a custom press close-up.
- Deterministic policy mean; success hunt uses env reset stochasticity.
- Cell 2 is a **new** 1M seed-6 run. It is the diagnosis stack plus
  Success-BC, not the 10-task paper BC curriculum.
- If the 150k handle policy never succeeds in 20 evals, the GIF is tagged
  `first_success_miss` (policy at first-success *time*, not a success).
