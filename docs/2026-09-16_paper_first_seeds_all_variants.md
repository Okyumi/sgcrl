# Breadth-first paper seeds: every 9-cell variant, then the rest of 10 seeds

Date: 2026-09-16  
Status: launched on Torch HPC. Supersedes the R/P-only 40-run rush in
`docs/2026-09-16_paper_rp_priority_finish.md`.

## Motivation

The abstract needs **some finished seeds of every transfer cell**, not
ten seeds of only R/R–P/P. The previous 40-run RP array would have used
the whole `QOSMaxGRESPerUser` cap (~6 L40S) on four cells while CKA
stayed unfinished.

No cell has completed the 10-task curriculum. The furthest seeds are:

- R/R, R/P, P/R, P/P: mostly task 7
- CKA/R: task 5
- R/CKA, P/CKA: task 4
- CKA/P, CKA/CKA: task 0 (those jobs were on task 1 when cancelled;
  `task_0.pkl` remains). Status `task_0` was latest-completed-task, not
  a resume hang.

This wave trains **3 seeds of each non-CKA cell** and **2 seeds of each
CKA cell** (22 runs, 6 array tasks). A later sweep can fill seeds 5–14.

Sparse SAC is not in this wave: 4-way packing still dies in `tf.data`
(`FunctionDef` registry). It needs a 1-process-per-GPU retry after the
contrastive first seeds are on the GPU.

## Mathematical objective

Unchanged 3×3 actor/critic grid `{reset, persistent, CKA}` with
`native_info` success, width 1024, `k_max=5`.

## Code and configuration changes

- `experiment_configs_paper_first_seeds.py`: 22 runs, same checkpoint
  identity as `experiment_configs_paper_9baseline_10seed.py`. Packs are
  grouped by remaining work so task-7 seeds are not stuck behind task-0
  CKA roommates.
- `DRAFT_paper_first_seeds.sh`: array `0-5`, 4 learners per L40S, 48h
  `afterany` chain (max 6 hops). Array 5 has only the two CKA/CKA seeds.
- `scripts/paper_first_seeds_status.py` and unit tests.

The pending `paper_rp` array `17897850` is cancelled so these six packs
can take the GPU cap.

## Launch command

```bash
sbatch DRAFT_paper_first_seeds.sh
python scripts/paper_first_seeds_status.py --summary
```

After this wave, remaining 10-seed work is queued as
`paper_rest` (`docs/2026-09-16_paper_remaining_10seed_fill.md`): the
other 68 contrastive runs plus 1-per-GPU sparse SAC, with `--nice` and
`--dependency=after:17897878` so they cannot jump these six packs.

## Logged metrics

Unchanged. W&B groups stay `PAPER-9BASELINE-10SEED-{actor}-{critic}`.

## Validation

```bash
python tests/test_paper_first_seeds.py
```

## Known limitations

- CKA/P and CKA/CKA still need ~9 remaining 8M-step tasks, so even 2
  finished seeds of those cells take multiple 48h hops. The R/P cells
  should finish this hop.
- Mid-task replay is not serialised.
- Sparse SAC is still not producing paper curves.
