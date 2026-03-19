# Section 3: Continual Goal-Conditioned Contrastive RL

Branch: `section3_done`  
Entry point: `run_continual_contrastive.py`  
SLURM script: `draft_3.sh`

---

## Running Experiments with `draft_3.sh`

### Quick Reference

All commands below are run from the repo root on the HPC login node.

```bash
# Make sure you're on the right branch
cd /scratch/yd2247/sgcrl
git checkout section3_done
```

### 1. Full 10-task experiment (default settings)

```bash
sbatch draft_3.sh
```

This runs all 10 tasks sequentially with the default parameters:
- Seed: 42, Algorithm: contrastive_cpc, K_max: 5
- 1M env steps per task (10M total), W&B logging enabled
- Checkpoints saved to `/scratch/yd2247/sgcrl/logs/continual_checkpoints/`
- SLURM: 48h walltime, 1 GPU, 32GB RAM

### 2. Quick debug run (2 tasks, 10k steps each)

```bash
NUM_TASKS=2 STEPS_PER_TASK=10000 BASE_STEPS=10000 sbatch draft_3.sh
```

### 3. Resume from a specific task

If a job times out or fails at task 5, resume from where it left off:

```bash
START_TASK=5 sbatch draft_3.sh
```

This loads the checkpoint from task 4 and continues training from task 5 onward.

### 4. Run with a different seed

```bash
SEED=0 sbatch draft_3.sh
```

### 5. Run multiple seeds in parallel

```bash
SEED=0 sbatch draft_3.sh
SEED=1 sbatch draft_3.sh
SEED=2 sbatch draft_3.sh
```

Each job writes to its own checkpoint subdirectory (`seed_0/`, `seed_1/`, etc.), so they don't conflict.

### 6. Change the algorithm variant

```bash
# C-learning (TD-based)
ALG=c_learning sbatch draft_3.sh

# NCE + C-learning (combined)
ALG=nce+c_learning sbatch draft_3.sh
```

### 7. Change the knowledge pool size

```bash
K_MAX=3 sbatch draft_3.sh
K_MAX=8 sbatch draft_3.sh
```

### 8. Disable W&B logging

```bash
USE_WANDB=false sbatch draft_3.sh
```

### 9. Custom log and checkpoint directories

```bash
LOG_DIR=/scratch/yd2247/sgcrl/logs/exp_v2 \
CHECKPOINT_DIR=/scratch/yd2247/sgcrl/logs/exp_v2_ckpts \
sbatch draft_3.sh
```

### 10. Run a single task at a time (for shorter SLURM allocations)

If your cluster has strict walltime limits, you can run one task per job:

```bash
# Task 0 (base phase)
NUM_TASKS=1 START_TASK=0 sbatch draft_3.sh

# After task 0 finishes, submit task 1
NUM_TASKS=2 START_TASK=1 sbatch draft_3.sh

# After task 1 finishes, submit task 2
NUM_TASKS=3 START_TASK=2 sbatch draft_3.sh

# ... and so on up to task 9
NUM_TASKS=10 START_TASK=9 sbatch draft_3.sh
```

Note: `NUM_TASKS` sets the upper bound (exclusive), so `NUM_TASKS=3 START_TASK=2` runs only task 2.

### 11. Combine multiple overrides

```bash
SEED=7 ALG=contrastive_cpc NUM_TASKS=5 K_MAX=3 STEPS_PER_TASK=500000 sbatch draft_3.sh
```

### 12. Run directly without SLURM (interactive / debug)

If you're on an interactive GPU node:

```bash
# Source the environment setup from the script, then run directly
cd /scratch/yd2247/sgcrl
python run_continual_contrastive.py \
    --seed=42 \
    --alg=contrastive_cpc \
    --num_tasks=2 \
    --steps_per_task=10000 \
    --base_steps=10000 \
    --k_max=2 \
    --log_dir=logs/ \
    --checkpoint_dir=logs/continual_checkpoints
```

---

## All Environment Variables for `draft_3.sh`

| Variable | Default | Description |
|----------|---------|-------------|
| `SEED` | `42` | Random seed |
| `ALG` | `contrastive_cpc` | Algorithm: `contrastive_cpc`, `c_learning`, or `nce+c_learning` |
| `NUM_TASKS` | `10` | Number of tasks to run (up to 10) |
| `STEPS_PER_TASK` | `1000000` | Env steps per continual task (tasks 1-9) |
| `BASE_STEPS` | `1000000` | Env steps for the base task (task 0) |
| `K_MAX` | `5` | Max knowledge pool size before merging |
| `START_TASK` | `0` | Task ID to start from (0-9). Loads checkpoint from task START_TASK-1 |
| `EVAL_EVERY` | `50000` | Evaluate every N env steps within each task |
| `USE_WANDB` | `true` | Enable Weights & Biases logging (`true` or `false`) |
| `ADD_UID` | `true` | Add unique ID to log directories (`true` or `false`) |
| `LOG_DIR` | `/scratch/yd2247/sgcrl/logs/continual` | Base log directory |
| `CHECKPOINT_DIR` | `/scratch/yd2247/sgcrl/logs/continual_checkpoints` | Cross-task checkpoint directory |

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

---

## Output Structure

```
/scratch/yd2247/sgcrl/logs/
├── continual/                              # SLURM stdout/stderr
│   ├── 12345.out
│   └── 12345.err
├── continual_contrastive_cpc/              # Per-task training logs
│   ├── task0_sawyer_hammer_s42/
│   │   └── learner/                        # Learner metrics (CSV)
│   ├── task1_sawyer_push_wall_s42/
│   └── ...
└── continual_checkpoints/                  # Cross-task checkpoints
    └── seed_42/
        ├── task_0.pkl                      # θ_base, pool, critic after task 0
        ├── task_1.pkl
        └── ...
```

---

## New and Modified Files (Section 3)

### New files

| File | Description |
|------|-------------|
| `contrastive/continual_config.py` | `ContinualConfig` dataclass and 10-task sequence constant |
| `contrastive/knowledge_pool.py` | Knowledge vector pool with cosine-similarity merging, pytree utilities |
| `contrastive/continual_learning.py` | `ContinualContrastiveLearner`: CKA actor (`θ' = θ_base + Σα_j·v_j + v_k`), persistent critic, L2 actor loss |
| `contrastive/continual_builder.py` | `ContinualContrastiveBuilder` for LaunchPad integration |
| `run_continual_contrastive.py` | Sequential single-process training driver (recommended) |
| `lp_continual_contrastive.py` | LaunchPad-based entrypoint (alternative, has state-extraction limitations) |
| `draft_3.sh` | SLURM submission script for HPC |

### Modified files

| File | Change |
|------|--------|
| `contrastive/__init__.py` | Added exports for `ContinualConfig`, `ContinualContrastiveLearner`, `ContinualContrastiveBuilder`, `KnowledgePool` |

---

## Algorithm Design Notes

### Actor loss

Changed from the original inner product `diag(φ^T ψ)` to L2 distance `-‖φ(s,a) - ψ(g)‖₂`, matching the pseudocode in `CCRL.tex` and `strictly_c.tex`.

### Knowledge vectors

Full-policy pytree vectors (all MLP layers), not just the output head. Each `v_k` has the same structure as the complete policy parameters.

### Pool initialisation

After the base phase, the pool is initialised with a single zero vector (per pseudocode `V = {0}`). The base training delta `v_0` is folded into `θ_base` directly.

### JIT boundaries

- The main update step (critic loss, actor loss, gradient application for `v_k`) is JIT-compiled.
- Pool contribution `Σ α_j v_j` is computed outside JIT because the pool has variable length.
- `β_k` and `α_scale` gradients are computed outside JIT for the same reason.

### Replay

Per-task replay: each task gets a fresh Reverb server and buffer. No data is shared across tasks.
