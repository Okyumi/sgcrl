# Experiment Setup Guide

Complete reference for all experiment configurations, their meanings, and how to run them.

---

## Configuration Axes

The experiment design decouples three effects along two independent axes:

### Actor Mode (`--actor_mode`)

| Value | Behavior | Knowledge retention | Decomposition |
|---|---|---|---|
| `reset` | Reinitialize actor from scratch each task | None | None |
| `persistent` | Continuously trained single network, no decomposition | Yes (implicit) | None |
| `cka` | CKA-RL decomposition: θ' = θ_base + Σ α_j v_j + v_k | Yes (via pool) | Yes |

### Critic Mode (`--critic_mode`)

| Value | Behavior | Knowledge retention | Decomposition |
|---|---|---|---|
| `reset` | Reinitialize critic from scratch each task | None | None |
| `persistent` | Carry forward φ(s,a)ᵀψ(g) across tasks | Yes (implicit) | None |
| `cka` | CKA decomposition: (φ,ψ) = (φ,ψ)_base + Σ α_j w_j + w_k | Yes (via pool) | Yes |

---

## The 9 Configurations

### Group A: Reset Actor

| ID | Actor | Critic | What it tests |
|---|---|---|---|
| A1 | reset | reset | **Baseline 1**: fully independent, no transfer |
| A2 | reset | persistent | **Baseline 2**: critic-only transfer |
| A3 | reset | cka | Critic CKA decomposition without actor transfer |

### Group B: CKA Actor

| ID | Actor | Critic | What it tests |
|---|---|---|---|
| B1 | cka | persistent | CKA actor + persistent critic (main hypothesis) |
| B2 | cka | cka | Full CKA decomposition for both |
| B3 | cka | reset | **Baseline 4**: CKA-RL style but with GCRL instead of SAC |

### Group C: Persistent Actor

| ID | Actor | Critic | What it tests |
|---|---|---|---|
| C1 | persistent | persistent | Both continuously trained, no decomposition |
| C2 | persistent | reset | **Baseline 3**: actor-only persistence |
| C3 | persistent | cka | Persistent actor + CKA critic |

---

## Current Default Settings

These are the shared defaults for all experiments unless explicitly overridden:

| Setting | Value | Rationale |
|---|---|---|
| `use_task_id` | `false` | Agent must distinguish tasks from observations alone |
| `adapt_heads_only` | `true` | Match CKA-RL: only output head uses knowledge vectors |
| `encoder_from_base` | `false` | Match CKA-RL: encoder fine-tunes on each task |
| `use_residual` | `true` | ResidualMLP (LayerNorm + Swish + skip), based on scaling ablation |
| `steps_per_task` | 8,000,000 | Match SGCRL single-task training length |
| `base_steps` | 8,000,000 | Same as above for the base task |
| `k_max` | 10 | Knowledge pool capacity |
| `alg` | `contrastive_cpc` | CPC variant of InfoNCE (SGCRL default) |
| `critic_depth` | 4 | 1 residual block per encoder |
| `actor_depth` | 4 | 1 residual block for actor body |
| `network_width` | 256 | Hidden dimension |
| `energy_fn` | `inner_product` | φ(s,a)ᵀψ(g) |
| `logsumexp_penalty` | 0.01 | CPC regularizer |
| `eval_every` | 50,000 | Evaluate every 50K env steps |
| `eval_episodes` | 10 | 10 episodes per evaluation |

---

## Flag Reference

### Actor/Critic Configuration

| Flag | Type | Default | Description |
|---|---|---|---|
| `--actor_mode` | string | `cka` | Actor evolution: `cka`, `reset`, or `persistent` |
| `--critic_mode` | string | `persistent` | Critic evolution: `persistent`, `reset`, or `cka` |
| `--adapt_heads_only` | bool | `True` | Only head layers use knowledge vectors (CKA actor) |
| `--encoder_from_base` | bool | `False` | Freeze encoder from base task (CKA actor) |
| `--use_task_id` | bool | `False` | Append one-hot task ID to observations |

### Training

| Flag | Type | Default | Description |
|---|---|---|---|
| `--steps_per_task` | int | 8,000,000 | Env steps per continual task |
| `--base_steps` | int | 8,000,000 | Env steps for the base task |
| `--num_tasks` | int | 10 | Number of tasks in sequence |
| `--k_max` | int | 10 | Max knowledge pool size |
| `--seed` | int | 42 | Random seed |
| `--alg` | string | `contrastive_cpc` | Contrastive loss variant |

### Network Architecture

| Flag | Type | Default | Description |
|---|---|---|---|
| `--use_residual` | bool | `True` | Use ResidualMLP (LayerNorm + Swish + skip) |
| `--network_width` | int | 256 | Hidden dimension for ResidualMLP |
| `--critic_depth` | int | 4 | Dense layers in critic residual blocks (multiple of 4) |
| `--actor_depth` | int | 4 | Dense layers in actor residual blocks (multiple of 4) |
| `--energy_fn` | string | `inner_product` | Energy function: `inner_product` or `l2` |
| `--logsumexp_penalty` | float | 0.01 | CPC regularizer coefficient |

### Evaluation

| Flag | Type | Default | Description |
|---|---|---|---|
| `--eval_every` | int | 50,000 | Evaluate every N env steps (0 to disable) |
| `--eval_episodes` | int | 10 | Episodes per evaluation |
| `--intra_eval_previous_tasks` | bool | `False` | Evaluate on all previous tasks during current-task training |
| `--log_rl_metrics` | bool | `True` | Log representation metrics (weight norms, feature rank, NRC, etc.) |
| `--k_sample_k` | int | 0 | K-sample-argmax K (0 = deterministic mean) |

### Infrastructure

| Flag | Type | Default | Description |
|---|---|---|---|
| `--start_task` | int | 0 | Resume from this task (0 = auto-resume) |
| `--single_task` | string | `` | Override task sequence with one environment |
| `--use_wandb` | bool | `True` | Enable W&B logging |
| `--log_dir` | string | see draft_3.sh | Base log directory |
| `--checkpoint_dir` | string | see draft_3.sh | Checkpoint directory |

---

## Checkpoint System

### Path Format

```
{ckpt_dir}/actor_{mode}_critic_{mode}_tid_{bool}_heads_{bool}/seed_{seed}/task_{id}.pkl
```

Every configuration produces a unique directory. No cross-contamination possible.

### Auto-Resume

When `--start_task=0` (default), the runner scans for existing checkpoints matching the same config + seed and resumes from the latest completed task. Just resubmit the same `sbatch` command after preemption.

### What Is Restored

| State | Description |
|---|---|
| `theta_base` | Base policy (None for reset actor) |
| `pool_vectors` | Actor knowledge pool |
| `q_params` | Critic parameters |
| `target_q_params` | Target critic |
| `q_optimizer_state` | Critic optimizer Adam state |
| `q_base` | Frozen critic base (CKA critic only) |
| `critic_pool_vectors` | Critic knowledge pool (CKA critic only) |

### What Is Fresh Each Task

- v_k (initialized to zero)
- Blending weights β_k
- Replay buffer
- Actor optimizer state

---

## Experiment Commands

All commands use `use_task_id=false` and `adapt_heads_only=true` (the shared defaults).

### Group A: Reset Actor

```bash
# A1: reset actor + reset critic (baseline1: fully independent)
ACTOR_MODE=reset CRITIC_MODE=reset USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# A2: reset actor + persistent critic (baseline2: critic-only transfer)
ACTOR_MODE=reset CRITIC_MODE=persistent USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# A3: reset actor + CKA critic
ACTOR_MODE=reset CRITIC_MODE=cka USE_TASK_ID=false SEED=6 sbatch draft_3.sh
```

### Group B: CKA Actor

```bash
# B1: CKA actor + persistent critic
ACTOR_MODE=cka CRITIC_MODE=persistent USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# B2: CKA actor + CKA critic
ACTOR_MODE=cka CRITIC_MODE=cka USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# B3: CKA actor + reset critic (baseline4: CKA-RL with GCRL)
ACTOR_MODE=cka CRITIC_MODE=reset USE_TASK_ID=false SEED=6 sbatch draft_3.sh
```

### Group C: Persistent Actor

```bash
# C1: persistent actor + persistent critic
ACTOR_MODE=persistent CRITIC_MODE=persistent USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# C2: persistent actor + reset critic (baseline3)
ACTOR_MODE=persistent CRITIC_MODE=reset USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# C3: persistent actor + CKA critic
ACTOR_MODE=persistent CRITIC_MODE=cka USE_TASK_ID=false SEED=6 sbatch draft_3.sh
```

### Single-Task (for debugging or ablation)

```bash
# Single task on a specific environment
SINGLE_TASK=sawyer_shelf_place SEED=6 sbatch draft_3.sh
```

---

## Metrics

### Per-Task Training Performance

Logged during training by the `evaluator` logger (deterministic policy, `params.mode()`). This gives the learning curve for the current task. Always enabled.

### Intra-Task Cross-Evaluation (Optional)

When `--intra_eval_previous_tasks` is enabled, periodically evaluates the current policy on ALL tasks seen so far during training. Logged to W&B under `intra_eval/`. Disabled by default because it is expensive (creates environments for every past task at each eval interval). Enable with `INTRA_EVAL_PREVIOUS=true`.

### RL Representation Metrics (enabled by default)

When `--log_rl_metrics` is enabled (default), representation-quality metrics are logged at three frequency levels:

| Level | Frequency | Metrics logged |
|---|---|---|
| Frequent | Every `eval_every` steps | weight_norm, final_layer_norm, feature_entropy, gini_sparsity |
| Occasional | Every `5 × eval_every` steps | + feature_rank, NRC1, NRC2, dormant_ratio |
| Rare | Every `20 × eval_every` steps | + intrinsic_dimension (TWO-NN) |

All metrics are logged to W&B under `rl_metrics/`. Disable with `LOG_RL_METRICS=false`.

See `contrastive/rl_metrics.py` for implementation details.

### Post-Task Cross-Evaluation (Forgetting)

After each task k completes training, evaluate the final composed policy on all tasks 0..k. Logged to W&B under `eval/`. This always runs and measures how much past task performance degrades.

### Forward Transfer

Compare learning speed on task k (with knowledge from tasks 0..k-1) against the baseline where task k is trained independently (A1 configuration).

### Mean Success Rate

Average success rate across all tasks seen so far: `(1/(k+1)) Σ R[k][j]` for j in 0..k.

---

## Network Architecture Summary

Default architecture (`use_residual=True`, `critic_depth=4`, `actor_depth=4`):

```
Critic (per encoder):
  Input → Dense(256) → LayerNorm → Swish
        → 4× (Dense(256) → LayerNorm → Swish) + skip
        → Dense(64)

Actor:
  Input → Dense(256) → LayerNorm → Swish
        → 4× (Dense(256) → LayerNorm → Swish) + skip
        → Dense(256)
        → NormalTanhDistribution → mean, log_std
```

To revert to plain MLP: `USE_RESIDUAL=false`.

---

## Task Sequence

10 MetaWorld Sawyer manipulation tasks (in order):

| # | Task | Key challenge |
|---|---|---|
| 0 | hammer | Tool use |
| 1 | push_wall | Contact + push |
| 2 | faucet_close | Precision rotation |
| 3 | push_back | Push toward goal |
| 4 | stick_pull | Tool use (pull) |
| 5 | handle_press_side | Press from side |
| 6 | push | Simple push |
| 7 | shelf_place | Multi-step: grasp + lift + place |
| 8 | window_close | Sliding contact |
| 9 | peg_unplug_side | Extraction |

Unified observation: state ∈ ℝ¹¹, goal ∈ ℝ¹¹. When `use_task_id=True`, a one-hot vector is appended.

---

## Comparison with CKA-RL

| Aspect | CKA-RL | This project |
|---|---|---|
| RL algorithm | SAC | Contrastive GCRL (SGCRL) |
| Actor | CKA decomposition | CKA decomposition (same) |
| Critic | SAC Q(s,a)→ℝ, reset each task | Contrastive φ(s,a)ᵀψ(g), configurable |
| Reward | Hand-crafted per task | Self-supervised (hindsight relabeling) |
| Batch size | 128 | 256 |
| Learning rate | 1e-3 | 3e-4 |
| UTD ratio | 1 | 64 |
| Network | Plain MLP (256,256) | ResidualMLP (depth=4, width=256) |
| Pool size | 9 (code) / 5 (paper) | 10 |

The closest comparison to CKA-RL is configuration B3 (`actor_mode=cka, critic_mode=reset`).
