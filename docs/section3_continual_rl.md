# Section 3: Continual Goal-Conditioned Contrastive RL

Branch: `section3_done`  
Entry point: `run_continual_contrastive.py`  
SLURM script: `draft_3.sh`

---

## Overview

This project combines **SGCRL** (single-goal contrastive RL) with **CKA-RL** (continual knowledge adaptation). The underlying RL algorithm is contrastive goal-conditioned RL from the SGCRL paper, and the continual learning structure follows CKA-RL:

- **Actor:** CKA-RL-style decomposition. θ' = θ_base + Σ α_j v_j + v_k, where θ_base is frozen after the base task, v_j are knowledge vectors from past tasks, and v_k is the current task's vector. α = softmax(β · α_scale). By default, only the output head layers are decomposed (matching CKA-RL's `fuse_heads=True, fuse_shared=False`).
- **Critic:** Contrastive dual-encoder critic φ(s,a)^T ψ(g) from SGCRL, trained with InfoNCE. Three evolution modes across tasks: persistent (never reset), reset (fresh each task, matching CKA-RL's SAC critic), or CKA (base + per-task vectors).
- **Replay:** Fresh per task (no cross-task data leakage).
- **Evaluation:** K-sample-argmax inference and cross-task forgetting measurement.

On a single task (`NUM_TASKS=1, USE_TASK_ID=false, ADAPT_HEADS_ONLY=false`), the behavior matches the original SGCRL implementation exactly.

---

## All Configuration Flags

### Environment Variables for `draft_3.sh`

| Variable | Default | Description |
|----------|---------|-------------|
| `SEED` | `42` | Random seed |
| `ALG` | `contrastive_cpc` | Contrastive loss variant: `contrastive_cpc` (MC InfoNCE, used by SGCRL paper), `c_learning` (TD InfoNCE), or `nce+c_learning` (combined) |
| `NUM_TASKS` | `10` | Number of tasks to run |
| `STEPS_PER_TASK` | `1000000` | Env steps per continual task (tasks 1+) |
| `BASE_STEPS` | `1000000` | Env steps for the base task (task 0) |
| `K_MAX` | `5` | Max knowledge pool size before cosine-similarity merging |
| `START_TASK` | `0` | Resume from this task (loads checkpoint from task START_TASK-1) |
| `EVAL_EVERY` | `50000` | Intra-task evaluation every N env steps (0 to disable) |
| `EVAL_EPISODES` | `10` | Episodes per task for cross-task evaluation (0 to disable all eval) |
| `K_SAMPLE_K` | `0` | K for K-sample-argmax evaluation (0 = deterministic mean) |
| `CRITIC_MODE` | `persistent` | Critic evolution: `persistent`, `reset`, or `cka` |
| `USE_TASK_ID` | `true` | Append one-hot task ID to both state and goal |
| `ADAPT_HEADS_ONLY` | `true` | Only adapt actor head layers via v_k (CKA-RL default) |
| `ENCODER_FROM_BASE` | `true` | Freeze actor encoder from base task (CKA-RL default) |
| `USE_20_TASKS` | `false` | Use 20-task sequence (2 passes of 10 tasks) |
| `USE_WANDB` | `true` | Enable Weights & Biases logging |
| `ADD_UID` | `true` | Add unique ID to log directories |
| `LOG_DIR` | `/scratch/yd2247/sgcrl/logs/continual` | Base log directory |
| `CHECKPOINT_DIR` | `/scratch/yd2247/sgcrl/logs/continual_checkpoints` | Cross-task checkpoint directory |

### Checkpoint path structure

Checkpoints are keyed by all ablation-relevant configuration to prevent cross-contamination:

```
{CHECKPOINT_DIR}/critic_{mode}_tid_{bool}_heads_{bool}/seed_{s}/task_{t}.pkl
```

Example: `.../critic_persistent_tid_True_heads_True/seed_42/task_3.pkl`

Resuming with `START_TASK=k` loads the checkpoint matching the exact same `CRITIC_MODE`, `USE_TASK_ID`, and `ADAPT_HEADS_ONLY` from the previous run. A mismatch raises a clear `FileNotFoundError`.

---

## Required Experiments

All experiments use 3 seeds (0, 1, 2) for statistical significance. Submit each seed in parallel.

### Experiment 0: Single-Task Sanity Check

Verify that single-task SGCRL matches the validated single-task results from earlier work. Run one task with no continual learning machinery.

```bash
# Sawyer Hammer (task 0) — should match original SGCRL results
for SEED in 0 1 2; do
  NUM_TASKS=1 USE_TASK_ID=false ADAPT_HEADS_ONLY=false \
  ENCODER_FROM_BASE=false CRITIC_MODE=persistent \
  SEED=$SEED sbatch draft_3.sh
done
```

### Experiment 1: Critic Evolution Ablation (3 × 3 seeds)

The central hypothesis: does the contrastive critic benefit from persistence across tasks, compared to CKA-RL's approach of resetting the critic?

All three use the CKA-RL-matching actor defaults: `ADAPT_HEADS_ONLY=true`, `ENCODER_FROM_BASE=true`, `USE_TASK_ID=true`.

```bash
# 1a. Persistent critic — never reset, carry forward (our proposal)
for SEED in 0 1 2; do
  CRITIC_MODE=persistent SEED=$SEED sbatch draft_3.sh
done

# 1b. Reset critic — reinitialize each task (matches CKA-RL's SAC critic treatment)
for SEED in 0 1 2; do
  CRITIC_MODE=reset SEED=$SEED sbatch draft_3.sh
done

# 1c. CKA critic — base + per-task knowledge vectors for critic
for SEED in 0 1 2; do
  CRITIC_MODE=cka SEED=$SEED sbatch draft_3.sh
done
```

### Experiment 2: Task ID Ablation (2 × 3 seeds)

Does providing task identity to the contrastive critic help? CKA-RL does not use task IDs (its SAC critic resets each task). With a persistent contrastive critic, task IDs may help the shared representations distinguish between tasks.

```bash
# 2a. With task ID (default)
for SEED in 0 1 2; do
  CRITIC_MODE=persistent USE_TASK_ID=true SEED=$SEED sbatch draft_3.sh
done

# 2b. Without task ID
for SEED in 0 1 2; do
  CRITIC_MODE=persistent USE_TASK_ID=false SEED=$SEED sbatch draft_3.sh
done
```

### Experiment 3: Actor Decomposition Ablation (2 × 3 seeds)

CKA-RL only decomposes the actor's output head. Does decomposing the full policy (including the shared encoder) help or hurt?

```bash
# 3a. Heads only (CKA-RL matching, default)
for SEED in 0 1 2; do
  ADAPT_HEADS_ONLY=true ENCODER_FROM_BASE=true SEED=$SEED sbatch draft_3.sh
done

# 3b. Full policy adaptation
for SEED in 0 1 2; do
  ADAPT_HEADS_ONLY=false ENCODER_FROM_BASE=false SEED=$SEED sbatch draft_3.sh
done
```

### Experiment 4: K-Sample-Argmax Evaluation Ablation

Does using the critic to select among K candidate actions improve evaluation? Run one configuration with different K values. This only affects evaluation, not training — so you can re-evaluate from saved checkpoints.

```bash
# 4a. Deterministic mean (K=0, default)
for SEED in 0 1 2; do
  K_SAMPLE_K=0 SEED=$SEED sbatch draft_3.sh
done

# 4b. K=10
for SEED in 0 1 2; do
  K_SAMPLE_K=10 SEED=$SEED sbatch draft_3.sh
done

# 4c. K=50
for SEED in 0 1 2; do
  K_SAMPLE_K=50 SEED=$SEED sbatch draft_3.sh
done
```

### Experiment 5: 20-Task Stress Test

CKA-RL evaluates on 20 tasks (2 passes through the 10-task sequence). Run the best configuration from earlier experiments on the extended sequence.

```bash
for SEED in 0 1 2; do
  USE_20_TASKS=true NUM_TASKS=20 SEED=$SEED sbatch draft_3.sh
done
```

### Quick Debug Runs

Before committing to full-scale experiments, validate each configuration at small scale:

```bash
# 2 tasks, 20k steps each, fast turnaround
NUM_TASKS=2 STEPS_PER_TASK=20000 BASE_STEPS=20000 \
EVAL_EPISODES=5 USE_WANDB=false SEED=42 sbatch draft_3.sh
```

### Batch Submission Script

To submit the full critic evolution ablation (Experiment 1) across 3 seeds:

```bash
for MODE in persistent reset cka; do
  for SEED in 0 1 2; do
    CRITIC_MODE=$MODE SEED=$SEED sbatch draft_3.sh
  done
done
```

---

## 10-Task Sequence

| Task ID | Env Name | Meta-World Task |
|---------|----------|-----------------|
| 0 | `sawyer_hammer` | hammer-v2 |
| 1 | `sawyer_push_wall` | push-wall-v2 |
| 2 | `sawyer_faucet_close` | faucet-close-v2 |
| 3 | `sawyer_push_back` | push-back-v2 |
| 4 | `sawyer_stick_pull` | stick-pull-v2 |
| 5 | `sawyer_handle_press_side` | handle-press-side-v2 |
| 6 | `sawyer_push` | push-v2 |
| 7 | `sawyer_shelf_place` | shelf-place-v2 |
| 8 | `sawyer_window_close` | window-close-v2 |
| 9 | `sawyer_peg_unplug_side` | peg-unplug-side-v2 |

The 20-task sequence (`--use_20_tasks`) repeats this sequence twice (tasks 0-9, then 10-19 = same envs).

---

## Architecture

### Actor (CKA-RL style)

The SGCRL actor is `hk.Sequential([MLP(256, 256), NormalTanhDistribution(act_dim)])`.

- **Body (shared encoder):** `mlp/~/linear_0`, `mlp/~/linear_1` — two hidden layers
- **Head (output):** `normal_tanh_distribution/~/linear` (mean), `normal_tanh_distribution/~/linear_1` (log_std)

**Task 0 (base phase):** Always trains the full policy (body + head) with no gradient masking, regardless of `adapt_heads_only` or `encoder_from_base`. This matches CKA-RL (which trains a full base model) and SGCRL (which has no decomposition). After task 0, θ_base is a fully-trained policy.

**Tasks k > 0:** When `adapt_heads_only=True` (default, matching CKA-RL):
- v_k only modifies head parameters. Body gradients are zeroed via `tree_map_with_path`.
- The body always comes from θ_base (frozen from task 0).

When `adapt_heads_only=False`:
- v_k modifies the entire policy pytree.
- `encoder_from_base` controls whether the body can change.

### Critic (contrastive dual-encoder)

The SGCRL critic has two sub-networks:
- φ(s,a): `sa_encoder` MLP, input = concat(state, action)
- ψ(g): `g_encoder` MLP, input = goal

Score: φ(s,a)^T ψ(g) (inner product, matching SGCRL paper).

Three evolution modes across tasks:
- **persistent:** carry forward (φ, ψ) across all tasks, fine-tune with InfoNCE on each task.
- **reset:** reinitialize (φ, ψ) from scratch at each new task (matching how CKA-RL treats its SAC critic).
- **cka:** q_base frozen from task 0, per-task critic knowledge vectors w_k with cosine-similarity merging. Inner loop trains the composed critic normally; w_k extracted post-task.

### Task conditioning

When `use_task_id=True`, a one-hot vector is appended to both state and goal at the gym level (`TaskIDGymWrapper` in `env_utils.py`). Observation layout:

```
[state_spatial (11), task_one_hot (num_tasks), goal_spatial (11), task_one_hot (num_tasks)]
```

State and goal have identical dimensionality. The contrastive critic sees the task ID in both φ(s,a) and ψ(g).

### Evaluation

- **Cross-task evaluation:** After each task, evaluates the current policy on ALL tasks seen so far. Measures forgetting.
- **Intra-task evaluation:** Every `eval_every` env steps during training, evaluates on all tasks 0..current_task_id. Provides learning curves.
- **K-sample-argmax:** At eval time, sample K actions from the policy, score each with φ(s,a)^T ψ(g), pick the best. The critic acts as a reranker.
- **Monitoring:** `SuccessObserver` (0/1 task success) and `DistanceObserver` (L2 distance to goal: init_dist, final_dist, delta_dist, min_dist).

---

## Output Structure

```
/scratch/yd2247/sgcrl/logs/
├── continual/                                          # SLURM stdout/stderr
│   ├── 12345.out
│   └── 12345.err
├── continual_contrastive_cpc/
│   └── critic_persistent_tid_True_heads_True/          # Config-specific logs
│       ├── task0_sawyer_hammer_s42/
│       │   ├── learner/                                # Learner metrics (CSV)
│       │   └── actor/                                  # Actor metrics (CSV)
│       ├── task1_sawyer_push_wall_s42/
│       └── ...
└── continual_checkpoints/
    └── critic_persistent_tid_True_heads_True/          # Config-specific checkpoints
        └── seed_42/
            ├── task_0.pkl
            ├── task_1.pkl
            └── ...
```

---

## Running Experiments

### Quick start (default settings = CKA-RL-matching)

```bash
cd /scratch/yd2247/sgcrl
git checkout section3_done
sbatch draft_3.sh
```

### Resume from a specific task

```bash
START_TASK=5 sbatch draft_3.sh
```

Loads the checkpoint from task 4 with matching configuration.

### Run multiple seeds in parallel

```bash
SEED=0 sbatch draft_3.sh
SEED=1 sbatch draft_3.sh
SEED=2 sbatch draft_3.sh
```

### Disable evaluation (faster training)

```bash
EVAL_EPISODES=0 sbatch draft_3.sh
```

### Run directly without SLURM (interactive / debug)

```bash
cd /scratch/yd2247/sgcrl
python run_continual_contrastive.py \
    --seed=42 --num_tasks=2 --steps_per_task=10000 --base_steps=10000 \
    --k_max=2 --log_dir=logs/ --checkpoint_dir=logs/ckpts \
    --nouse_wandb --nouse_task_id --eval_episodes=5
```

---

## Key Design Decisions

### Why replace SAC with contrastive RL?

CKA-RL uses SAC, where the critic Q(s,a) → R estimates expected return and requires a hand-crafted reward function. SGCRL's contrastive critic φ(s,a)^T ψ(g) learns representations of state-action reachability in a self-supervised way (InfoNCE). The hypothesis is that these learned representations generalize better across tasks than SAC's scalar Q-function, making a persistent contrastive critic more viable than a persistent SAC critic.

### Why persist the critic?

In CKA-RL, the SAC critic is reset each task because a scalar Q-function doesn't transfer well across tasks with different reward functions. But the contrastive critic learns environment dynamics (which states are reachable from which state-action pairs), which is more likely to transfer across manipulation tasks sharing the same robot.

### Actor loss: inner product (matching SGCRL)

The actor maximizes `diag(φ(s,a)^T ψ(g))` — the diagonal of the inner-product logit matrix. This is the same score used in the InfoNCE critic loss. The SGCRL paper (Section 4.2) shows the inner-product architecture is essential for effective exploration.

### Head-only vs. full-policy adaptation

CKA-RL only decomposes the actor's output head (`fc_mean`, `fc_logstd`), keeping the shared encoder body frozen from the base task. Our default matches this. The `adapt_heads_only=False` option is available to test whether adapting the full policy helps in the contrastive RL setting.

### Single-task = original SGCRL

With `NUM_TASKS=1, USE_TASK_ID=false, ADAPT_HEADS_ONLY=false`, the behavior matches the original SGCRL implementation exactly:
- Task 0 trains the full policy with no gradient masking or decomposition
- θ' = θ_base + 0 + v_0 = θ_base + v_0 (pool is empty, no β_k)
- Actor loss: `-diag(φ(s,a)^T ψ(g))` (inner product)
- Critic loss: InfoNCE with CPC (softmax CE + logsumexp penalty)
- Optimizer: Adam(lr=3e-4, eps=1e-7) for both actor and critic
- entropy_coefficient=0.0, random_goals=0.5, use_random_actor=True

Even with `ADAPT_HEADS_ONLY=true` (default), task 0 is unaffected because the base task never masks gradients. The defaults match CKA-RL's architecture for subsequent tasks while preserving exact SGCRL behavior for the base task.
