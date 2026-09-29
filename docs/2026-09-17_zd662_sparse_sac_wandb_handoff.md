# Collaborator sparse SAC+HER W&B handoff

Date: 2026-09-17  
Status: local copy complete; cloud upload in progress to `nyuad_mmvc/zd662_sparse_sac_her`.

## Motivation

The paper's sparse-reward non-contrastive baseline is goal-conditioned
SAC+HER. Those runs were logged by the collaborator under
`d_konoki/continual_sac`. This account **cannot read that project** (W&B
returns "Could not find project continual_sac"), so the paper agent that
authenticates as `nyuad_mmvc` cannot pull success-rate or AUC curves from
it. The collaborator granted filesystem access to `/scratch/zd662/sgcrl`
and pointed at `HANDOFF_extract_wandb.md` inside the local `wandb/`
folder.

## What was copied

Full offline W&B dump, 7.7 GB / 375 run directories:

```
/scratch/zd662/sgcrl/wandb/  ->  /scratch/yd2247/sgcrl/logs/zd662_wandb/
```

`logs/` is gitignored, so this does **not** land in the live
repo-root `wandb/` directory used by our own jobs. The collaborator's
decoder and note are at:

- `logs/zd662_wandb/HANDOFF_extract_wandb.md`
- `logs/zd662_wandb/run-20260525_003915-n5q38mn1/` (example task-6
  R/R width-256 run, already decoded to CSV/JSONL)
- `scripts/read_offline_wandb.py` (copy of their decoder)

Per-run inventory: `results/data/raw/zd662_continual_sac/run_catalog.csv`.

## What the dump contains

374 / 375 runs are `run_continual_sac.py` / `alg_name=sac_her`. Headline
paper cells (`actor_mode == critic_mode ∈ {reset, persistent}`,
`step_penalty_reward=True`) cover a complete 10-task grid for seeds
1–6 at **both** widths:

| Width | Cell | Seeds | Tasks | Unique (cell, seed, task) |
|---|---|---|---|---|
| 256 | R/R and P/P | 1–6 | 0–9 | 120 |
| 1024 | R/R and P/P | 1–6 | 0–9 | 120 |

The remaining runs are CKA-actor cells, non-penalty reward ablations,
and retries. Run names match the paper convention
`task{k}_{env}_s{seed}`. Original W&B group is hardcoded `sac_test`.

This is **more** seed coverage than the 2026-05-20 cloud pull
(`results/docs/2026-05-20_sparse_sac_baseline_results.md`, 3 seeds from
`d_konoki/continual_sac`).

## Cloud upload (so the paper agent can see it)

Do **not** dump these into `continual_gcrl_paper`: that project already
holds the contrastive 9-cell grid, and mixing 375 SAC runs there would
break group-based fetches. They go to a dedicated project:

- entity: `nyuad_mmvc`
- project: `zd662_sparse_sac_her`
- group: `sac_test` (preserved from the original `wandb.init`)

Smoke test (handoff run `n5q38mn1`) succeeded:
https://wandb.ai/nyuad_mmvc/zd662_sparse_sac_her/runs/n5q38mn1
(`task6_sawyer_push_s6`, 159 evaluator rows, config intact). A non-fatal
artifact error (`cannot create manifest for artifact in state DELETED`)
appears on sync; the metric history still lands.

Batch upload (CPU SLURM job, so a dropped chat cannot kill it):

```bash
sbatch DRAFT_sync_zd662_wandb.sh
```

The script skips run IDs already present in the target project. Progress:
`results/data/raw/zd662_continual_sac/sync_log.jsonl` and
`logs/zd662_wandb_sync_slurm_<jobid>.out`.

## How the paper agent should fetch AUC / success

Same fetcher as the contrastive cells, retargeted:

```bash
WANDB_API_KEY=... python results/scripts/fetch_wandb_runs.py \
    --project nyuad_mmvc/zd662_sparse_sac_her \
    --groups sac_test \
    --out_dir results/data/raw
```

Then `results/scripts/compute_metrics.py` on that dump for
`best_success`, `end_success`, and trapezoidal `auc_success`. Filter
config to the paper cells:

- `alg_name == sac_her`
- `actor_mode == critic_mode` in `{reset, persistent}`
- `step_penalty_reward == True`
- choose `network_width` 256 (legacy collaborator sweep) or 1024
  (matches the current paper SAC rerun)

Local fallback if the cloud fetch is still catching up:

```bash
python scripts/extract_zd662_offline_histories.py
```

That writes `results/data/raw/zd662_continual_sac/histories.csv` and
`runs.csv` (evaluator success-rate trajectories plus per-run
best/end success). Single-file decode: `scripts/read_offline_wandb.py`
with wandb 0.15.12 (`contrastive_rl` env), as in the collaborator
handoff.

## Validation

- Copy: `du -sh logs/zd662_wandb` = 7.7G; HANDOFF note and example run
  present.
- API identity: default entity `nyuad_mmvc`; `d_konoki/continual_sac`
  is not readable.
- Full sync (SLURM `18243711`, 2026-09-22): **375 / 375** local run
  dirs are in `nyuad_mmvc/zd662_sparse_sac_her` (0 missing). Group
  `sac_test` has 374 SAC+HER runs plus one contrastive leftover.
- Paper cells are complete: width `{256,1024}` × `{R/R, P/P}` × seeds
  1–6 × tasks 0–9 = 240 unique keys. Task 9 width-1024 spot check
  (`task9_sawyer_peg_unplug_side_s2`) has 159 evaluator rows.

## Known limitations

- Original `wandb.init` used `entity='entity'` / `group="sac_test"`;
  entity is overwritten on sync, group is not.
- Metadata `state` is often still `running` even when the binary log
  finished; the synced cloud state for the smoke test was `finished`.
- Duplicate run dirs exist for some `(width, cell, seed, task)` keys
  (retries). Prefer the latest start time / largest `.wandb` file when
  aggregating.
- Mid-task replay is not in these logs; only W&B metrics.
- The 7.7G binary dump stays gitignored under `logs/`.
