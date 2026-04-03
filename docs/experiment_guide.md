# Experiment Guide

Reference for all configuration flags, experiment configurations, metrics, and launch commands.

---

## 1. Why 7 Runs Appeared Under SEED=5

### Expected: 5 jobs

The five submitted commands:

```bash
CRITIC_MODE=persistent SEED=5 sbatch draft_3.sh           # config 1
CRITIC_MODE=reset SEED=5 sbatch draft_3.sh                 # config 2
CRITIC_MODE=cka SEED=5 sbatch draft_3.sh                   # config 3
CRITIC_MODE=persistent USE_TASK_ID=false SEED=5 sbatch draft_3.sh  # config 4
ADAPT_HEADS_ONLY=false ENCODER_FROM_BASE=false SEED=5 sbatch draft_3.sh  # config 5
```

Each command maps to exactly one SLURM job. `draft_3.sh` has no loops, no sweep logic, no array jobs — it runs a single `python run_continual_contrastive.py` invocation per `sbatch`.

### Effective configs

| # | critic_mode | use_task_id | adapt_heads_only | checkpoint config_key |
|---|---|---|---|---|
| 1 | persistent | True | True | `critic_persistent_tid_True_heads_True` |
| 2 | reset | True | True | `critic_reset_tid_True_heads_True` |
| 3 | cka | True | True | `critic_cka_tid_True_heads_True` |
| 4 | persistent | False | True | `critic_persistent_tid_False_heads_True` |
| 5 | persistent | True | False | `critic_persistent_tid_True_heads_False` |

All defaults come from `draft_3.sh` lines 30-46. When an env var is not set, it falls back to the default (e.g., `CRITIC_MODE` defaults to `persistent`, `ADAPT_HEADS_ONLY` defaults to `true`).

### Diagnosis: 2 extra jobs are accidental double-submissions

There is **no code-level bug** causing duplication. The SLURM script and Python driver are 1:1 — each `sbatch` produces exactly one job. The 2 extra jobs (7 - 5 = 2) were most likely caused by accidentally running one or two of the commands twice (e.g., hitting Enter twice, or re-submitting after checking the queue).

**How to verify:** Run `sacct -u yd2247 --format=JobID,JobName,Submit,Start,State -S 2026-04-01` and check the submission timestamps. Duplicate jobs will have nearly identical submit times.

**Prevention:** Add `--job-name` with the config key to make duplicates visible in `squeue`:
```bash
CRITIC_MODE=persistent SEED=5 sbatch --job-name=crl_pers_tid_heads_s5 draft_3.sh
```

---

## 2. Configuration Flags

### `adapt_heads_only` (default: `True`)

**What it does:** Controls which part of the actor uses the CKA knowledge-vector decomposition (θ_base + Σ α_j v_j + v_k).

**Where it is used:**
- `run_continual_contrastive.py` lines 548-573: Post-task extraction splits v_k into head and body portions
- `contrastive/continual_learning.py` lines 143-159: Controls gradient masking (indirectly; `encoder_from_base` is the direct gradient control)

**Behavior:**
- `True` (CKA-RL default): Only the output head layers (`normal_tanh_distribution/*`) are decomposed into base + vectors. The body (MLP encoder) receives full gradients during training; after each task, the body portion of v_k is folded back into θ_base (so the encoder evolves across tasks), and only the head portion is stored in the knowledge pool. Pool vectors have zero-valued body entries.
- `False`: Full-policy adaptation. θ_base is frozen after task 0. The entire v_k (body + head) goes into the pool. The body only changes via the knowledge-vector composition, not by folding.

**Practical meaning:** `True` matches CKA-RL's original design where `fuse_shared=False, fuse_heads=True`. The encoder is free to evolve; only the output head has per-task specialization. `False` means the entire policy is frozen at base + pool + v_k.

**Current runs:** `True` for all standard configs; `False` for experiment 3 (actor decomposition ablation).

### `encoder_from_base` (default: `False`)

**What it does:** Controls whether the body (encoder) gradients are masked to zero for tasks k > 0.

**Where it is used:**
- `contrastive/continual_learning.py` lines 156-159: Sets `self._mask_body_grads`
- `contrastive/continual_learning.py` gradient masking in `_update_step` (applied after v_k gradient computation)

**Behavior:**
- `False` (default): Body receives full gradients. v_k accumulates updates for both body and head. After each task (with `adapt_heads_only=True`), body portion is folded into θ_base.
- `True`: Body gradients are zeroed out for tasks k > 0. The encoder is frozen at whatever θ_base was after task 0. Only head parameters in v_k get non-zero gradients.

**Practical meaning:** `False` matches CKA-RL's behavior (encoder is fine-tuned on every task). `True` is an ablation that freezes the encoder entirely.

**Current runs:** `False` for all configs.

### `time_delta_minutes` (default: `5`)

**What it does:** Sets the Acme checkpointing interval (in minutes) used by `CheckpointingRunner` in the LaunchPad distributed layout.

**Where it is used:**
- `contrastive/config.py` line 15: Config field
- `contrastive/distributed_layout.py` lines 189, 234: Passed to `savers.CheckpointingRunner`

**Behavior:** In the **distributed LaunchPad runner** (`lp_contrastive.py`), the learner and counter are wrapped in `CheckpointingRunner` which saves Acme-format checkpoints every `time_delta_minutes` minutes. This enables resumption after preemption.

**Important caveat for continual RL:** The **sequential runner** (`run_continual_contrastive.py`) does NOT use `CheckpointingRunner`. It passes `time_delta_minutes` into the config dict but never reads it. The sequential runner saves its own pickle-format checkpoints at the end of each task (not mid-task). This means `time_delta_minutes` has **no effect** on the continual RL pipeline.

**Current runs:** Default value (5), effectively unused.

### `use_cpc` (default: `False`)

**What it does:** Selects the contrastive loss variant for the critic.

**Where it is used:**
- `contrastive/continual_learning.py` lines 236-240: Inside `critic_loss_fn`
- `contrastive/learning.py` line 183: Same logic in the non-continual learner

**Behavior:**
- `True` (CPC / InfoNCE): Softmax cross-entropy loss with a logsumexp penalty:
  ```
  L = softmax_cross_entropy(logits, I) + 0.01 * logsumexp(logits, axis=1)²
  ```
  This is the CPC (Contrastive Predictive Coding) variant. The logsumexp penalty prevents logit magnitudes from exploding.
- `False` (NCE): Sigmoid binary cross-entropy against the identity matrix:
  ```
  L = sigmoid_binary_cross_entropy(logits, I)
  ```

**Practical meaning:** `use_cpc=True` is the recommended setting from SGCRL (Liu et al., ICLR 2025). It is set when `--alg=contrastive_cpc`.

**Current runs:** `True` (set automatically by `--alg=contrastive_cpc`).

### `use_random_actor` (default: `True`)

**What it does:** Controls whether the data-collection actor uses uniform random actions before the first learner update.

**Where it is used:**
- `run_continual_contrastive.py` lines 432-439: Chooses between `InitiallyRandomActor` and `GenericActor`
- `contrastive/utils.py` lines 196-210: `InitiallyRandomActor` implementation
- `contrastive/builder.py` lines 83-84: Same in distributed builder

**Behavior:**
- `True`: Uses `InitiallyRandomActor`, which outputs uniform random actions in [-1, 1] until the first policy update (detected by checking if bias params are still zero). After the first update, it switches to the learned policy. This improves initial replay buffer diversity.
- `False`: Uses `GenericActor`, which always samples from the learned policy (which is near-random at init due to random weights, but not uniformly random).

**Practical meaning:** `True` ensures the replay buffer is filled with diverse trajectories before training starts, which is beneficial for contrastive learning (more varied negative samples).

**Current runs:** `True` (set in `run_continual_contrastive.py` line 629: `'use_random_actor': True`).

### `reset_actor` (default: `False`) — NEW

**What it does:** Resets the actor from scratch at every task. No CKA decomposition, no knowledge pool, no θ_base carry-over.

**Where it is used:**
- `run_continual_contrastive.py`: Main loop passes `theta_base=None` and empty pool for tasks k > 0
- `contrastive/continual_learning.py` line 459: `if theta_base is None` triggers fresh policy init

**Behavior:**
- `True`: Each task initializes a fresh random policy and trains independently. The critic still follows `critic_mode` (so `--reset_actor --critic_mode=reset` gives a fully independent baseline; `--reset_actor --critic_mode=persistent` tests critic-only transfer).
- `False`: Standard CKA-RL decomposition with knowledge vectors.

**Current runs:** `False`. Use `--reset_actor --critic_mode=reset` for the fully independent baseline.

---

## 3. Experiment Configurations

### Core experiments (Section 3)

| Experiment | Command | What it tests |
|---|---|---|
| Exp 0: Single-task sanity | `NUM_TASKS=1 STEPS_PER_TASK=8000000 SEED=5 sbatch draft_3.sh` | Must match original SGCRL |
| Exp 1a: Persistent critic | `CRITIC_MODE=persistent SEED=5 sbatch draft_3.sh` | Contrastive critic carried forward (our hypothesis) |
| Exp 1b: Reset critic | `CRITIC_MODE=reset SEED=5 sbatch draft_3.sh` | Critic reinitialized each task (CKA-RL baseline) |
| Exp 1c: CKA critic | `CRITIC_MODE=cka SEED=5 sbatch draft_3.sh` | Knowledge vectors for critic too |
| Exp 2: No task ID | `CRITIC_MODE=persistent USE_TASK_ID=false SEED=5 sbatch draft_3.sh` | Task ID ablation |
| Exp 3: Full-policy | `ADAPT_HEADS_ONLY=false ENCODER_FROM_BASE=false SEED=5 sbatch draft_3.sh` | Full-policy adaptation (not head-only) |
| Exp 4: Reset both (baseline) | `RESET_ACTOR=true CRITIC_MODE=reset SEED=5 sbatch draft_3.sh` | Fully independent: no transfer at all |
| Exp 5: Reset actor, keep critic | `RESET_ACTOR=true CRITIC_MODE=persistent SEED=5 sbatch draft_3.sh` | Critic-only transfer (no actor CKA) |

### Checkpoint paths

Checkpoints are keyed by all ablation-relevant config to prevent cross-contamination:

```
{checkpoint_dir}/critic_{mode}_tid_{bool}_heads_{bool}_areset_{bool}/seed_{seed}/task_{id}.pkl
```

Example: `logs/continual_checkpoints/critic_persistent_tid_True_heads_True_areset_False/seed_5/task_3.pkl`

---

## 4. Metrics Pipeline

### Actor metrics vs. evaluator metrics

| Metric source | Policy | Purpose | Logger label |
|---|---|---|---|
| **Actor** | Stochastic (sampled from distribution) | Training data collection; noisy exploration | `actor` |
| **Evaluator** | Deterministic (distribution mode/mean) | True performance measurement; no exploration noise | `evaluator` |

Both log the same observer metrics (success rate, distance) but to separate loggers. The evaluator gives a cleaner signal because it uses the greedy policy.

**Implementation:**
- Actor: `contrastive_networks.apply_policy_and_sample(networks)` → stochastic sampling
- Evaluator: `contrastive_networks.apply_policy_and_sample(networks, eval_mode=True)` → `params.mode()` (deterministic mean)

Both are run periodically during training (every `eval_every` env steps). The evaluator was previously only in the LaunchPad distributed runner; it is now added to the sequential runner.

### Intra-task evaluation (during training)

Every `eval_every` env steps during each task's training:

1. **Evaluator run**: Deterministic policy evaluated on the CURRENT task for `eval_episodes` episodes. Logged to `evaluator` logger.
2. **Cross-task eval**: Deterministic policy evaluated on ALL tasks seen so far (tasks 0 through k). Logged to W&B under `intra_eval/`.

### Post-task evaluation (after training completes)

After each task k finishes training, evaluate the final composed policy on all tasks 0 through k:

```python
for eval_tid in range(task_id + 1):
    sr = evaluate_on_task(task_sequence[eval_tid], ...)
```

Results logged to W&B under `eval/`. This builds the success rate matrix R[k][j] = success rate on task j after training on task k.

### Continual RL metrics

The raw success rate matrix R[k][j] is logged after each task. From this matrix, the following metrics can be computed:

#### Forward Transfer (FT)

**Definition:** How much faster does the agent learn task k compared to learning it from scratch?

$$\text{FT}_k = R[k][k] - R_{\text{baseline}}[k]$$

where $R_{\text{baseline}}[k]$ is the success rate when training task k independently (from the `reset_actor + reset_critic` baseline).

**Computation:** Not computed automatically in the pipeline — requires comparing against the independent baseline. After running both the CKA experiment and the reset baseline:
```
FT_k = eval_success_CKA[task_k][task_k] - eval_success_reset[task_k][task_k]
```

**When measured:** After training on task k completes.

#### Forgetting

**Definition:** How much does performance on task j degrade after learning subsequent tasks?

$$\text{Forgetting}_j = R[j][j] - R[k][j], \quad k > j$$

The maximum forgetting for task j is:
$$\text{Forgetting}_j = R[j][j] - \min_{k > j} R[k][j]$$

And the average forgetting after all N tasks:
$$\text{Avg Forgetting} = \frac{1}{N-1} \sum_{j=0}^{N-2} \left( R[j][j] - R[N-1][j] \right)$$

**Computation:** Computed from the post-task cross-evaluation success rates logged to W&B under `eval/`. Each `eval/{env_name}` gives R[k][j] where k is the current task and j is the evaluated task.

**When measured:** After each task completes. The full forgetting picture requires all N tasks to be trained.

**Caveat:** The current implementation evaluates all past tasks with the COMPOSED policy from the current task (θ_base + pool_c + v_k). For CKA-RL, each past task j should ideally use its own v_j to reconstruct θ'_j. The current approach measures: "can the current policy still solve past tasks?" rather than "can the stored v_j still solve task j?". For reset_actor mode, this is equivalent since there's no pool.

#### Mean Success Rate

**Definition:** Average success rate across all tasks seen so far.

$$\text{Mean SR}_k = \frac{1}{k+1} \sum_{j=0}^{k} R[k][j]$$

**Computation:** Logged to W&B as `eval/mean_success` after each task.

---

## 5. File Reference

| File | Purpose |
|---|---|
| `run_continual_contrastive.py` | Main sequential training driver (~844 lines) |
| `contrastive/continual_learning.py` | ContinualContrastiveLearner: CKA composition, losses, gradient masking (~784 lines) |
| `contrastive/knowledge_pool.py` | Knowledge vector pool, merging, serialization (~143 lines) |
| `contrastive/networks.py` | Network definitions, policy sampling, K-sample-argmax (~190 lines) |
| `contrastive/config.py` | ContrastiveConfig dataclass |
| `contrastive/continual_config.py` | ContinualConfig, task sequences |
| `env_utils.py` | Sawyer wrappers, TaskIDGymWrapper (~1077 lines) |
| `draft_3.sh` | SLURM launcher for NYUAD HPC |
| `docs/section3_continual_rl.md` | Comprehensive project documentation |
| `docs/algorithm_pseudocode.md` | Algorithm pseudocode |
