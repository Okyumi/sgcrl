# `results/` — continual-GCRL experimental results

Reproducible pipeline for the experimental results that back the
RLC 2026 Continual RL Workshop submission (paper repo
`Okyumi/CGCRL---RLC-workshop-2026`).

## Layout

```
results/
├── README.md                       # this file
├── scripts/
│   ├── fetch_wandb_runs.py         # pull histories from W&B
│   ├── compute_metrics.py          # cell metrics + CRL metrics
│   └── render_tables.py            # markdown tables for review
├── data/
│   ├── raw/                        # one folder per W&B group
│   │   ├── for_real/
│   │   ├── real1/
│   │   └── real2/
│   └── processed/                  # CSVs the rest of the codebase reads
│       ├── per_seed_per_task.csv
│       ├── per_task_aggregate.csv
│       ├── cell_summary.csv
│       ├── crl_metrics.csv
│       └── documentation.md        # auto-generated metric reference
├── tables/                         # human-readable markdown tables
│   ├── per_task_table.md
│   ├── cell_summary_table.md
│   └── crl_metrics_table.md
└── docs/
    └── 2026-05-14_results_overview.md   # one-stop summary
```

## Quick start

```bash
# 1. Fetch from W&B (~25 min for 764 runs across the three groups).
WANDB_API_KEY=$YOUR_KEY python results/scripts/fetch_wandb_runs.py

# 2. Compute metrics from the cached histories.
python results/scripts/compute_metrics.py

# 3. Render the markdown tables.
python results/scripts/render_tables.py
```

Steps 2 and 3 take seconds once step 1 has run.

## What gets reported, in one paragraph

Each run on W&B trains a single (task k, seed) pair and logs the
per-step ``evaluator/success_rate`` trajectory. From that trajectory
we compute, per (cell, seed, task k): ``best_success`` (the max),
``end_success`` (the last value), ``mean_success_during_training``,
``auc_success``, and ``stability = end / best``. We aggregate across
seeds within a cell to get per-task mean ± std, and across the ten
tasks to get a single ``avg_best`` per cell. Forward transfer is the
``best_success`` of a cell minus that of the from-scratch reference
cell ``actor=reset-critic=reset`` on the same (task, seed) for
k ≥ 1. Classical BWT / forgetting requires cross-task evaluation
during training, which none of the existing runs logged; we report
the within-task ``stability`` as the in-data proxy and document the
shortcut.

## Group routing (the rule the markdown tables apply)

Group | Cells it provides | Why
---|---|---
``for_real`` | GCRL ``{reset, persistent}`` × ``{reset, persistent}`` (4 cells) plus ``actor != cka, critic = cka`` (2 cells) | the GCRL grid + the never-re-run mixed cells
``real1 + real2`` | ``actor = cka`` cells (3 cells), pooled per seed by keeping the more-complete run | CKA re-runs

The routing is encoded in
``results/scripts/render_tables.py`` (constants ``GCRL_CELLS``,
``CKA_ACTOR_CELLS``).

## What is queued but not here yet

* Sparse goal-conditioned SAC baseline (different algorithm family).
* Cross-task BWT re-run with ``--intra_eval_previous_tasks=true``.
* Decomposed-critic numbers (paper §4) — currently in flight on the
  cluster under the ``C2`` family.
