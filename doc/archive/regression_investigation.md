# Regression Investigation — April 18, 2026

## Investigation Method

Compared every line of code between the last known-good commit (`6b56a78`, the seed=5 experiments) and the current HEAD, covering `continual_learning.py`, `networks.py`, `config.py`, `continual_config.py`, `utils.py`, and `run_continual_contrastive.py`.

## Findings

### No training-breaking bugs found in the core training loop

The following core components are unchanged and verified correct:
- `ContinualContrastiveLearner._update_step()`: critic loss (InfoNCE), actor loss (-diag(φᵀψ)), gradient computation
- `_scan_update()`: lax.scan inner SGD loop
- CKA composition: θ' = θ_base + pool_c + v_k
- Replay buffer creation, episode adder, dataset iterator
- Actor-learner loop: run_episode() → learner.step()

### Changes since last known-good that affect behavior

| Change | Commit | Impact on training performance |
|---|---|---|
| `use_task_id` default True → False | a4120fc | **HIGH**: Critic loses task conditioning. Reachability ambiguous across tasks. |
| `use_residual` default False → True | 5b3f7d9 | **MEDIUM**: Larger network (LayerNorm params in CKA composition). Works well single-task, untested in continual. |
| `k_max` 5 → 10 | 20248c6 | LOW: More pool capacity, unlikely to hurt. |
| `steps_per_task` 1M → 8M | 20248c6 | NONE: Just longer training. |
| Head detection fix (`Normal` not `normal_tanh_distribution`) | dcb5fd0 | **POSITIVE**: Pool now actually stores head vectors. |
| `critic_was_freshly_init` refactor | 20248c6 | NONE: Verified equivalent logic. |
| Env cleanup (close envs between tasks) | dcb5fd0 | **POSITIVE**: Prevents resource leaks. |
| Evaluator added to training loop | 5dafb89 | NONE on performance (separate env, no adder). |
| RL metrics sampling | 5594bdc | NEGLIGIBLE: One extra replay sample per 50K steps. |
| Learner metrics logging | 802bd12 | NEGLIGIBLE: Reads cached metrics, no extra computation. |

### Most likely cause of degraded performance: `use_task_id=False`

With `use_task_id=True`, the contrastive critic receives:
- sa_encoder input: [state(11), task_one_hot(10), action(4)] = 25 dims
- g_encoder input: [goal(11), task_one_hot(10)] = 21 dims

With `use_task_id=False`:
- sa_encoder input: [state(11), action(4)] = 15 dims
- g_encoder input: [goal(11)] = 11 dims

Without the task one-hot, the critic cannot distinguish goals from different tasks. When task 1 (push_wall) starts with a persistent critic from task 0 (hammer), the critic must map the same 11-dim goal space to reachability predictions that are task-specific — but it has no way to know WHICH task the goal belongs to.

This is a fundamental architectural limitation of `use_task_id=False` with `critic_mode=persistent`. The user requested this as the experimental default.

### Secondary contributor: `use_residual=True` in continual setting

The ResidualMLP adds LayerNorm parameters (scale, offset) to the CKA composition. With `adapt_heads_only=True`, the body's LayerNorm params are folded into θ_base after each task and the v_k for those params starts from zero. This means the LayerNorm scale starts at the correct value (from θ_base) and v_k adds a delta. Mathematically this is sound, but:

1. The optimization landscape may be harder — v_k controls both Dense weights and LayerNorm scale/offset, which have very different sensitivity
2. The number of parameters in the body is larger (LayerNorm adds 2 × width params per normalization layer), making the CKA decomposition operate in a higher-dimensional space

This was validated for single-task but not for the continual CKA setting.

## Recommendations

To diagnose which change caused the degradation, run these in order:

1. **Restore `use_task_id=True`** on one experiment and compare. If performance recovers, the task conditioning is critical.
   ```bash
   ACTOR_MODE=cka CRITIC_MODE=persistent USE_TASK_ID=true SEED=6 sbatch draft_3.sh
   ```

2. **Restore `use_residual=False`** (plain MLP) on one experiment and compare.
   ```bash
   ACTOR_MODE=cka CRITIC_MODE=persistent USE_RESIDUAL=false SEED=6 sbatch draft_3.sh
   ```

3. **Run the exact scaling study config** to verify the architecture works:
   ```bash
   SINGLE_TASK=sawyer_shelf_place USE_TASK_ID=true K_MAX=5 SEED=9 USE_RESIDUAL=true sbatch draft_3.sh
   ```
