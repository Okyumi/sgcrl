# W&B eval rollout videos for Task-5/Task-8 debugging

Date: 2026-08-31
Status: implemented

## Motivation

Handle-press corrected-wrapper runs show `actor/success_1000` peaking near ~0.18
around 250k env steps then collapsing, while `learner/binary_accuracy` stays
high. Visual rollouts are needed to see whether the policy loses contact,
presses incorrectly, or stops exploring the handle region.

## Implementation

- `contrastive/eval_video.py`: headless MuJoCo / `rgb_array` capture and one
  deterministic eval episode (`record_episode_frames`).
- `run_continual_contrastive.py`: flags `--eval_record_video`,
  `--eval_video_every` (default 50k), `--eval_video_fps` (default 20). When
  enabled at eval checkpoints, logs `wandb.Video` to
  `evaluator/rollout_video` plus return/success scalars.
- Task-58 fullnet launcher config enables video every 50k steps.

## Launch

Same as the 8M fullnet matrix (`sbatch DRAFT_task58_dcc_corrected.sh`). Videos
appear under the run Media tab as `evaluator/rollout_video`, keyed by
`evaluator/env_steps`.

## Validation

```bash
python tests/test_eval_video.py
```

## Known limitations

- One episode per checkpoint (deterministic policy mean), 640×480, ~150 steps.
- Rendering uses the eval env copy; failures are logged but do not abort training.
- W&B step collisions with actor/learner loggers may still occur unless
  `wandb.define_metric` is configured per family.
