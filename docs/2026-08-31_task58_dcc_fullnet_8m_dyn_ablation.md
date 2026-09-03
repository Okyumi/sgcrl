# Task-5/Task-8 full-network 8M-step DCC dyn-aux ablation

Date: 2026-08-31
Status: launcher ready

## Motivation

The 1M-step heads-only dyn ablation (`TASK58-DCC-CORRECTED-DYN-ABLATION-1M`)
showed unstable sparse success on handle press with high eval variance. This
matrix trains the **full** 1024-wide residual actor/critic (not
`adapt_heads_only`) for **8M** env steps per task to test whether longer
training and full capacity improve consolidation, and whether the dynamics
auxiliary (`dyn_aux_weight`) helps or hurts.

## Matrix (12 cells)

| Axis | Values |
|---|---|
| Task | `sawyer_handle_press_side` (5), `sawyer_window_close` (8) |
| Dynamics aux | `dyn_aux_weight` 1.0 vs 0.0 |
| Seeds | 5, 6, 7 |

All cells: single-task DCC, `critic_mode=decomposed`, `actor_mode=reset`,
corrected wrapper, full-state goal, no task ID, 1024×4 actor/critic,
`in_trajectory_negative_repeats=1`, simulator probes disabled.

## Code / config changes

- `experiment_configs_task58_dcc_corrected.py`: 8M steps,
  `adapt_heads_only=false`, `log_rl_metrics=true`, `log_pool_cosine=true`,
  W&B group `TASK58-DCC-CORRECTED-FULLNET-8M-DYN-ABLATION`.
- `DRAFT_task58_dcc_corrected.sh`: 48h wall, `#SBATCH --array=0-11`,
  `TASKS_PER_GPU=1`, logs under `logs/task58_dcc_fullnet_8m_v1/`.

## Launch

```bash
sbatch DRAFT_task58_dcc_corrected.sh
```

## Logged metrics

**Eval (every 50k steps, 10 deterministic episodes):**

- `evaluator/success_rate`, `evaluator/mean_return`
- `evaluator/task58/task_axis_success` (corrected success)
- `evaluator/task58/legacy_success`, `axis_rescued_success`
- `evaluator/task58/approach_success`, `mechanism_moved`,
  `max_task_axis_progress`, distances, `success_reward_mismatch_steps`

**Learner (when `log_rl_metrics=true`):** critic/actor losses, logits,
categorical accuracy, entropy.

**Runtime:** `runtime/*` fractions when `profile_runtime=true`.

**Eval video (every 50k steps):** `evaluator/rollout_video` (W&B Media panel),
`evaluator/video_episode_return`, `evaluator/video_episode_success`.

## Validation

```bash
python tests/test_task58_dcc_corrected.py
python tests/test_task58_reachable_success_goals.py
python tests/test_eval_video.py
```

## Known limitations

- Eval remains 10 episodes → 10% granularity on success rate.
- `dyn_aux_weight=0` disables dynamics auxiliary loss only; decomposed critic
  architecture is unchanged.
- Prior 1M heads-only checkpoints are not resumed (`heads_False` in ckpt path).
