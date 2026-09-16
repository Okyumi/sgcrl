# Queue the leftover 10-seed paper runs behind the first-seed wave

Date: 2026-09-16  
Status: leftover work is filled by a CPU dispatcher. First-seed packing
is two learners × two CPU actors per L40S
(`docs/2026-09-16_torch_parallel_actors.md`).

## Motivation

The breadth-first wave (`docs/2026-09-16_paper_first_seeds_all_variants.md`)
only covers 22 of 90 contrastive runs. The abstract still needs **10 seeds
of every 9-cell variant**. Those leftover 68 jobs must be in the queue
now, but they must not take GPUs until the incomplete first-seed arrays
are queued.

## Mathematical objective

Unchanged 3×3 `{reset, persistent, CKA}` grid, `native_info`, width 1024,
`k_max=5`, seeds 5–14. This launch is the complement of the 22 first-seed
runs: no duplicated actor/critic/seed keys, same checkpoint paths.

## Code and configuration changes

- `experiment_configs_paper_remaining_seeds.py`: 68 runs with
  `num_actors=2`. Config order is still 17 groups of 4 (leftover R/P
  then CKA); the launcher packs two runs per GPU, so the array is
  `0-33`.
- `DRAFT_paper_remaining_seeds.sh`: `#SBATCH --nice=100`,
  `TASKS_PER_GPU=2`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`.
- Sparse SAC is resubmitted at **1 learner per GPU** (`TASKS_PER_GPU=1`,
  `--nice=200`) because 4-way packing hits a TensorFlow `FunctionDef`
  registry crash. That is 20 array tasks behind both contrastive waves.

## Launch command

```bash
sbatch DRAFT_paper_first_seeds.sh
sbatch DRAFT_paper_cap_dispatcher.sh
python scripts/paper_cap_dispatcher.py   # one-shot dry check
```

The dispatcher runs on the `cs` CPU partition, submits incomplete
`paper_fs` arrays first, then leftover packs into free slots (`gpu48`
cap 16, one spare). All GPU jobs use `MAX_CHAIN=0` so they do not stack
extra pending GPUs; the dispatcher resubmits on timeout. SAC is 1 learner
per GPU after leftover contrastive arrays are queued.

## Logged metrics

Same W&B project/groups as the 9-baseline rerun. SAC groups remain
`PAPER-SPARSE-SAC-10SEED-{reset-reset|persistent-persistent}`.

## Validation

```bash
python tests/test_paper_remaining_seeds.py
python tests/test_paper_cap_dispatcher.py
```

## Known limitations

- Nice=100 keeps leftover packs behind running first-seed jobs under
  `QOSMaxGRESPerUser`.
- CKA leftover seeds that never started still need a full 10-task run.
- SAC at 1/GPU uses the cap slowly; it is intentionally last.
