# Batch Experiment Launcher — April 20, 2026

## Overview

`draft_4.sh` + `experiment_configs.py` enable running the full 9-configuration ablation grid across multiple seeds using SLURM job arrays, with 2 experiments per GPU.

## Architecture

```
experiment_configs.py     Enumerates all (actor_mode, critic_mode, seed) triples
        │                 as integer-indexed configurations.
        │
        ▼
draft_4.sh                SLURM job array script. Each array task:
  ┌─────┴─────┐           1. Reads 2 config indices for its array task ID
  │           │           2. Launches 2 Python processes in parallel
  slot 0    slot 1        3. Each gets its own .out and .err log file
  │           │           4. Waits for both to finish
  ▼           ▼
  python run_continual_contrastive.py (×2 per GPU)
```

## The 9-Configuration Grid

| # | Actor Mode | Critic Mode | Group |
|---|------------|-------------|-------|
| 0 | reset      | reset       | A — Reset actor |
| 1 | reset      | persistent  | A |
| 2 | reset      | cka         | A |
| 3 | persistent | reset       | C — Persistent actor |
| 4 | persistent | persistent  | C |
| 5 | persistent | cka         | C |
| 6 | cka        | reset       | B — CKA actor |
| 7 | cka        | persistent  | B |
| 8 | cka        | cka         | B |

Each configuration × 5 seeds = 45 total experiments.

## JAX GPU Memory

**Problem**: JAX preallocates 75% of GPU memory by default. With 2 processes per GPU, the second process fails to allocate.

**Solution**: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45` gives each process 45% of GPU memory (total 90%, 10% headroom for CUDA context).

This is set in `draft_4.sh` and applies to all launched Python processes via environment variable inheritance.

**Alternatives considered**:
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — on-demand allocation. Avoids the fixed reservation but is prone to fragmentation during long (8M-step) training runs. Not recommended.
- `XLA_PYTHON_CLIENT_ALLOCATOR=platform` — exact allocation with deallocation. Very slow, not suitable for training.

## Log File Structure

Each experiment gets its own `.out` and `.err` file with the FULL output, identical in format to what `draft_3.sh` produces:

```
/scratch/yd2247/sgcrl/logs/continual/
├── {ARRAY_JOB_ID}_{ARRAY_TASK_ID}.out    ← SLURM-level log (launcher info)
├── {ARRAY_JOB_ID}_{ARRAY_TASK_ID}.err    ← SLURM-level errors
├── {ARRAY_JOB_ID}_{TASK_ID}_{CONFIG_IDX}.out  ← per-experiment stdout
├── {ARRAY_JOB_ID}_{TASK_ID}_{CONFIG_IDX}.err  ← per-experiment stderr
└── ...
```

The per-experiment `.out` files contain the same run-info header as `draft_3.sh` (seed, actor_mode, critic_mode, all flags, etc.), followed by training output. The per-experiment `.err` files capture stderr (JAX warnings, Python tracebacks, etc.).

The SLURM-level `.out` file (from `#SBATCH --output`) contains the launcher summary: which configs were dispatched, PIDs, and the final "All experiments complete" message.

## Usage

### Run the full grid (45 configs, 23 GPUs)

```bash
sbatch draft_4.sh
# Array range 0-22 is already set in the script header
```

### Run a subset

```bash
# Only the first 10 configs (5 GPUs)
sbatch --array=0-4 draft_4.sh

# Only configs 30-44 (CKA actor, all critic modes)
sbatch --array=15-22 draft_4.sh
```

### Override shared defaults

```bash
# Quick debug: 2 tasks, 100K steps
NUM_TASKS=2 STEPS_PER_TASK=100000 sbatch draft_4.sh

# Disable W&B
USE_WANDB=false sbatch draft_4.sh
```

### Check what's in a specific config index

```bash
python experiment_configs.py --setting 35
# ACTOR_MODE=cka
# CRITIC_MODE=persistent
# SEED=1
```

### List all configurations

```bash
python experiment_configs.py --list
```

## Customising the Grid

Edit `experiment_configs.py`:

```python
# Change the grid:
ACTOR_MODES  = ['reset', 'persistent', 'cka']
CRITIC_MODES = ['reset', 'persistent', 'cka']
SEEDS        = [1, 2, 3, 4, 5]

# Add overrides applied to ALL configs:
EXTRA_OVERRIDES = {'steps_per_task': 4000000}
```

After editing, update the `--array` range in `draft_4.sh`:

```bash
total=$(python experiment_configs.py --total)
max_id=$(( (total + 1) / 2 - 1 ))  # ceil(total / TASKS_PER_GPU) - 1
echo "Set --array=0-${max_id}"
```

## Relationship to draft_3.sh

`draft_4.sh` is a superset of `draft_3.sh`:
- The **environment setup** (lines 93-119) is copied verbatim from `draft_3.sh` — module loads, conda, LD_LIBRARY_PATH, all exports.
- The **flag building** logic is identical (extracted into `build_flags()` function).
- The **per-experiment log output** matches the format of `draft_3.sh`.
- `draft_3.sh` is still used for single ad-hoc runs with custom overrides.
- `draft_4.sh` is for systematic batch sweeps.
