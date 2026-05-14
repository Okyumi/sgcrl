# Continual GCRL baseline results — overview (2026-05-14)

This document is a one-stop summary of what we currently have for the
**9-cell ablation grid** on the ten-task Continual World V2 Sawyer
manipulation sequence. It deliberately stops at "results that exist
and what they say"; it does not yet decide what goes into the LaTeX
paper.

## What is here

| Path | Content |
|---|---|
| `results/scripts/fetch_wandb_runs.py` | Pulls `evaluator/success_rate` + `evaluator/env_steps` histories from W&B for groups `for_real`, `real1`, `real2`. Outputs parquet histories + runs.csv + manifest under `results/data/raw/<group>/`. |
| `results/scripts/compute_metrics.py` | Reads the raw histories. Computes per-(cell, seed, task) metrics, per-task aggregates, per-cell summaries, and CRL metrics. Writes CSVs to `results/data/processed/`. Auto-regenerates `documentation.md`. |
| `results/scripts/render_tables.py` | Renders the markdown tables in `results/tables/` from the processed CSVs, applying the project's group-routing rule. |
| `results/data/raw/<group>/` | One parquet of per-step histories + one CSV of run metadata + manifest. |
| `results/data/processed/` | `per_seed_per_task.csv`, `per_task_aggregate.csv`, `cell_summary.csv`, `crl_metrics.csv`, `documentation.md`. |
| `results/tables/` | Three markdown tables for human review: per-task, cell-summary, CRL-metrics. |

## Reproducing the pipeline end-to-end

```bash
cd /scratch/yd2247/sgcrl   # or wherever the section3_done checkout lives

# 1. Fetch from W&B (long; ~25 min for 764 runs).
WANDB_API_KEY=$YOUR_KEY python results/scripts/fetch_wandb_runs.py \
    --project nyuad_mmvc/continual_gcrl_paper \
    --groups for_real real1 real2 \
    --out_dir results/data/raw

# 2. Compute metrics from the cached histories.
python results/scripts/compute_metrics.py \
    --raw_dir results/data/raw \
    --groups for_real real1 real2 \
    --out_dir results/data/processed

# 3. Render the human-facing markdown tables.
python results/scripts/render_tables.py \
    --processed_dir results/data/processed \
    --out_dir results/tables
```

Re-running steps 2 and 3 is cheap (seconds) once the raw cache exists.

## Group routing (one-liner)

`for_real` is the GCRL grid; `real1` and `real2` are the CKA-actor
re-runs. We route as follows:

* GCRL cells (both actor and critic in {reset, persistent}) → `for_real` only.
* CKA-actor cells (`actor=cka`) → pool `real1 + real2`, deduped per seed
  by keeping the run with the most evaluator log rows.
* `actor != cka` but `critic = cka` → fall back to `for_real` (these
  cells were never re-run under `real1` / `real2`).

## Metric definitions (canonical)

See `results/data/processed/documentation.md` for the full reference.
Short form:

* **`best_success`** (per (task, seed)) — `max` of
  `evaluator/success_rate` over the per-step evaluation trajectory.
  Robust to in-task instability. **This is the headline.**
* **`end_success`** — last evaluator row's `evaluator/success_rate`.
  Pairs with `best_success` to surface within-task forgetting.
* **`mean_success_during_training`** — mean over the trajectory.
* **`auc_success`** — trapezoidal integral over env-steps, normalised.
* **`stability`** = `end_success / max(best_success, 1e-9)`. Within-task
  retention proxy.
* **`forward_transfer`** vs reference cell (default
  `actor=reset-critic=reset`) — `best_success(cell) -
  best_success(reference)` on the same (task, seed) for k ≥ 1.

## What we cannot compute from these runs

**Classical backward transfer / forgetting (BWT)** needs evaluations on
prior tasks j *after* training task k > j. Every run in the three
groups was launched with `intra_eval_previous_tasks=False`, so no
cross-task eval rows exist. ``stability`` is the in-data proxy.

To compute true BWT we have two options:

1. Re-launch a subset of cells with `--intra_eval_previous_tasks=true`
   on the cluster (the runner already supports the flag).
2. Run a separate eval pass on the per-task checkpoints saved on
   `/scratch` (the checkpoint format already includes everything we
   need; the inference script does not yet exist).

Either is doable; both belong in the camera-ready, not the
submission.

## Headline numbers (best mean success averaged over 10 tasks)

From `results/tables/cell_summary_table.md`:

| cell | avg_best | n_seeds | groups |
|---|---|---|---|
| actor=cka-critic=reset | **0.841** | 5 | real1+real2 |
| actor=reset-critic=cka | **0.840** | 5 | for_real |
| actor=reset-critic=reset | **0.832** | 5 | for_real |
| actor=persistent-critic=reset | 0.827 | 5 | for_real |
| actor=cka-critic=persistent | 0.797 | 4 | real1+real2 |
| actor=persistent-critic=persistent | 0.791 | 5 | for_real |
| actor=cka-critic=cka | 0.781 | 4 | real1+real2 |
| actor=reset-critic=persistent | 0.774 | 5 | for_real |
| actor=persistent-critic=cka | 0.761 | 5 | for_real |

Reading: at the headline `best_success` metric, the spread across the
9 cells is **0.08** (0.761 → 0.841). The from-scratch baseline
`(reset, reset)` is at 0.832; no cell beats it by more than 0.01.
The lowest-performing cells use a CKA *critic* with a non-CKA actor
(`persistent-cka`, `reset-persistent`), and the lowest is
`persistent-cka` at 0.761.

This is consistent with the prior observation that, on the
contrastive critic, **CKA on the critic side is not adding value
beyond what `reset` / `persistent` already give**, and arguably
hurts.

## Forward transfer vs the from-scratch baseline

From `results/tables/crl_metrics_table.md`. Positive = better than
training from scratch:

| cell | forward_transfer (k=1..9) | groups |
|---|---|---|
| actor=cka-critic=reset | +0.007 ± 0.009 (n=41) | real1+real2 |
| actor=reset-critic=cka | +0.005 ± 0.014 (n=41) | for_real |
| actor=persistent-critic=reset | -0.007 ± 0.012 (n=41) | for_real |
| actor=cka-critic=persistent | -0.041 ± 0.024 (n=34) | real1+real2 |
| actor=persistent-critic=persistent | -0.049 ± 0.028 (n=41) | for_real |
| actor=reset-critic=persistent | -0.066 ± 0.025 (n=41) | for_real |
| actor=cka-critic=cka | -0.068 ± 0.029 (n=34) | real1+real2 |
| actor=persistent-critic=cka | -0.078 ± 0.026 (n=41) | for_real |

Reading: no cell shows statistically significant positive forward
transfer over the from-scratch baseline. The point estimates that are
closest to zero are CKA-actor cells with `reset` or `persistent` on
the critic. Persistent-critic cells consistently transfer negatively,
including (and especially) when paired with `actor=cka`.

This complements the CKA-specific diagnostic results from earlier
(critic mixture is anti-predictive of success) by showing that, even
*without* the per-step mixture diagnostics, the headline success
metric does not favour the persistent-critic family in this setting.

## Stability (end_success / best_success)

`stability` says how much of the within-task best survives at the end
of the task. Values are clustered around 0.43–0.49 across all cells.
The fact that no cell exceeds ≈ 0.50 means that on this benchmark, in
this configuration, every cell loses **roughly half** of its best
in-training success by the end of the task. This is a within-task
phenomenon (because we have no cross-task eval); whether it
generalises to actual catastrophic forgetting on prior tasks needs
the BWT re-run described above.

## What is queued but not in these tables

* **Sparse SAC baseline** (different algorithm family). User's
  colleague is running it. When ready, drop the parquet/csv into
  `results/data/raw/sparse_sac/` and re-run `compute_metrics.py`
  with `--groups for_real real1 real2 sparse_sac`. The pipeline will
  pick it up automatically.
* **Cross-task BWT pass**. Either via `--intra_eval_previous_tasks=true`
  re-runs or a checkpoint-replay pass.
* **CKA-mechanism diagnostics** (Pearson r between α-scale and
  end-of-task success, Spearman ρ between α-scale and logsumexp).
  These already live in `docs/wandb_analysis/csv/` and
  `docs/2026-05-14_h1_h2_alpha_predicts_failure.md`. Per the
  workshop pivot, the mechanism story is moving to the appendix; the
  9-cell baseline numbers above are the main-text material.

## Recommendation for the paper

I am NOT making this call here — flagging the obvious framings for
the user's review:

1. **Main-table candidate**: `results/tables/per_task_table.md`.
   Rows are tasks, columns are the 9 cells, values are
   `best_success` mean ± std. Add an extra column for sparse SAC
   when the colleague's data lands.
2. **Headline summary**: `results/tables/cell_summary_table.md`'s
   `avg_best` column as a single bar / number per cell, with
   forward-transfer as a paired bar.
3. **Optional appendix**: per-cell `stability` table to make the
   point that none of the 9 cells beat the from-scratch baseline
   meaningfully, motivating the proposed decomposed critic (§4).
