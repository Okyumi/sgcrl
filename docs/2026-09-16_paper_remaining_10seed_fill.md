# Queue the leftover 10-seed paper runs behind the first-seed wave

Date: 2026-09-16  
Status: first leftover submit hit `QOSMaxGRESPerUser` (pending+running
GPUs count toward 16). Remaining work is filled by a CPU dispatcher
after `paper_fs` `17898476` is running.

## Motivation

The breadth-first wave (`docs/2026-09-16_paper_first_seeds_all_variants.md`)
only covers 22 of 90 contrastive runs. The abstract still needs **10 seeds
of every 9-cell variant**. Those leftover 68 jobs must be in the queue
now, but they must not take GPUs until the first-seed packs are running.

## Mathematical objective

Unchanged 3×3 `{reset, persistent, CKA}` grid, `native_info`, width 1024,
`k_max=5`, seeds 5–14. This launch is the complement of the 22 first-seed
runs: no duplicated actor/critic/seed keys, same checkpoint paths.

## Code and configuration changes

- `experiment_configs_paper_remaining_seeds.py`: 68 runs in 17 packs of
  4. Arrays 0–6 are leftover R/R, R/P, P/R, P/P (task 7 then 6 then 5).
  Arrays 7–16 are leftover CKA-side seeds.
- `DRAFT_paper_remaining_seeds.sh`: `#SBATCH --nice=100`, 48h chain.
  Initial submit uses `--dependency=after:17897878` so it cannot jump
  the first-seed array. Continuations use `afterany` of their own hop.
- Sparse SAC is resubmitted at **1 learner per GPU** (`TASKS_PER_GPU=1`,
  `--nice=200`) because 4-way packing hits a TensorFlow `FunctionDef`
  registry crash. That is 20 array tasks behind both contrastive waves.

## Launch command

```bash
sbatch DRAFT_paper_cap_dispatcher.sh
python scripts/paper_cap_dispatcher.py   # one-shot dry check
```

The dispatcher runs on the `cs` CPU partition, waits until `paper_fs` is
RUNNING, then `sbatch --array=<id>` leftover packs into free slots
(`gpu48` cap 16, one spare). Remaining GPU jobs use `MAX_CHAIN=0` so
they do not stack extra pending GPUs; the dispatcher resubmits on
timeout. SAC is 1 learner per GPU after all 17 contrastive leftover
packs are queued.

## Logged metrics

Same W&B project/groups as the 9-baseline rerun. SAC groups remain
`PAPER-SPARSE-SAC-10SEED-{reset-reset|persistent-persistent}`.

## Validation

```bash
python tests/test_paper_remaining_seeds.py
```

## Known limitations

- `after:17897878` only waits until the first-seed array **starts**, not
  until those 22 curricula finish. Nice=100 then keeps leftover packs
  behind running first-seed jobs under `QOSMaxGRESPerUser`.
- CKA leftover seeds that never started still need a full 10-task run.
- SAC at 1/GPU uses the cap slowly; it is intentionally last.
