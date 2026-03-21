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
| `contrastive/utils.py` | Added `TaskIDWrapper` environment wrapper for task conditioning |

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

### Training loop

The training loop within each task uses `env_loop.run_episode()` to run one full episode per iteration. We use `run_episode()` instead of `run(num_episodes=1)` because the installed Acme version's `EnvironmentLoop.run()` returns `None` (no return statement), while `run_episode()` returns a metrics dict containing `episode_length`. After each episode, the result is written to the actor logger with `env_loop._logger.write(result)` to replicate the logging that `run()` would normally do.

Each loop iteration: one episode (~150 env steps) followed by one `learner.step()` (64 SGD updates via `jax.lax.scan` internally).

The first `learner.step()` triggers JAX JIT compilation of the full update function (critic loss + actor loss + gradient steps). This typically takes ~1-2 minutes on GPU. The script prints explicit markers so you know when compilation starts and finishes.

### Logging

Three logging mechanisms run during training:

- **Actor logger** (label `[Actor]`): writes after each episode via the explicit `env_loop._logger.write(result)` call following `run_episode()`. Throttled by `TimeFilter(time_delta=10.0)` to one entry every 10 seconds. Logs episode length, return, success rate, distances.
- **Learner logger** (label `[Learner]`): writes after each `learner.step()`. Also throttled to once per 10 seconds. Logs critic loss, actor loss, accuracy, entropy.
- **W&B logging**: when `--use_wandb` is set, `wandb.init()` is called per task in `main()` (with `reinit=True`), and `wandb.finish()` is called after each task completes. The `WandbLogger` instances in `default.py` assume `wandb.init()` has already been called — without it, all `wandb.log()` calls silently fail because `WandbLogger.write()` wraps errors in a bare `try/except pass`.

Both actor and learner loggers also write to CSV files under `log_dir/`. The `PYTHONUNBUFFERED=1` environment variable in `draft_3.sh` ensures stdout is flushed immediately to the SLURM `.out` file.

Stdout progress lines (e.g. `Task 0 [sawyer_hammer]: 50000/990000 env steps (334 episodes)`) are printed every 10,000 env steps independently of the TimeFilter loggers.

### Task conditioning

A one-hot task identifier is inserted between the state and goal portions of each observation via `TaskIDWrapper` (defined in `contrastive/utils.py`). The raw environment observation `[state_spatial (11), goal_spatial (11)]` becomes `[state_spatial (11), task_one_hot (num_tasks), goal_spatial (11)]`, and `obs_dim` is updated to `11 + num_tasks`.

During training, goals are relabeled as future states from the same trajectory. Since a goal is a future state, it has the **same dimensionality as the state** — including the task identifier. The training observation for the critic is therefore `[state (obs_dim), goal (obs_dim)]` where both halves include the task one-hot.

The `DistanceObserver` uses `end_index = raw_obs_dim` (11) to measure spatial distance only, excluding the task identifier from distance computation.

### Replay

Per-task replay: each task gets a fresh Reverb server and buffer. No data is shared across tasks. The server is created at the start of `train_single_task` and stopped at the end.

---

## Bug Fixes History

### 1. `env_loop.run()` returns `None` (commit `4059eb8`)

The installed Acme version's `EnvironmentLoop.run()` has no `return` statement (the GitHub master source has `return step_count`, but the pip-installed version does not). The original code `episode_steps = env_loop.run(num_episodes=1)` crashed with `TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'`.

Fix: use `env_loop.run_episode()` which returns a metrics dict, then read `result['episode_length']`. Also manually call `env_loop._logger.write(result)` to replicate the logging that `run()` does internally.

### 2. Missing `wandb.init()` (commit `1354c24`)

`run_continual_contrastive.py` created `WandbLogger` instances via `make_default_logger(use_wandb=True)` but never called `wandb.init()`. Since `WandbLogger.write()` catches all exceptions with a bare `try/except pass`, all W&B logging silently failed.

Fix: call `wandb.init(project='continual_gcrl', ..., reinit=True)` per task in `main()` before calling `train_single_task()`, and `wandb.finish()` after each task. Mirrors the pattern in `lp_continual_contrastive.py`.

### 3. `DuplicateFlagError` for `--log_dir` (commit `4059eb8`)

`run_continual_contrastive.py` defined `flags.DEFINE_string('log_dir', ...)` but Abseil already registers `--log_dir` internally. Removed the duplicate definition; `FLAGS.log_dir` now comes from Abseil's built-in registration.

### 4. `lax.scan` refactor (commit `7a39200`)

Refactored `ContinualContrastiveLearner.step()` to use `jax.lax.scan` for the inner SGD loop instead of Python for-loops, matching the original SGCRL learner pattern. The `_scan_update` wrapper reshapes transitions `[B*N, ...] → [N, B, ...]` and scans `update_step` over mini-batches while broadcasting the pool contribution (which is param-shaped, not batch-indexed).

### 5. Rate limiter deadlock during prefill (commit `f798a50`)

The `SampleToInsertRatio` rate limiter blocked inserts once the replay table reached ~`min_size_to_sample` episodes, causing prefill to hang at a deterministic point (~72 episodes). In the sequential driver, prefill only inserts (no sampling), so this limiter is the wrong fit.

Fix: switched to `rate_limiters.MinSize(min_replay_traj)` which only gates sampling (not inserts). Also removed `jax_utils.prefetch` which caused backpressure deadlocks during prefill, and set `num_parallel_calls=1` for the dataset interleave to avoid `drop_remainder` batching issues with small replay buffers.

### 6. Codebase cleanup and task conditioning (commit `df62961`)

Removed all debug timing/print instrumentation from `ContinualContrastiveLearner.step()` and the dead `_compose_theta` function. Simplified the verbose prefill loop. Added `TaskIDWrapper` for task conditioning and documented replay isolation.
