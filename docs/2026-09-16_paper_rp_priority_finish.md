# Priority finish of R/R, R/P, P/R, P/P for the abstract

Date: 2026-09-16  
Status: **superseded** the same day by the breadth-first first-seed wave
in `docs/2026-09-16_paper_first_seeds_all_variants.md`. The 40-run
`paper_rp` array was cancelled before it left the queue so every 9-cell
variant can get a few finished seeds first. The RP packing files remain
for the later 10-seed fill of R/R, R/P, P/R, P/P.

## Motivation

The 9-baseline × 10-seed paper array was not going to finish before the
abstract deadline. After seven days, **0/90** curricula were complete.
The four non-CKA cells (R/R, R/P, P/R, P/P) were the closest: 32/40
seeds had finished task 7, 4 had finished task 6, 4 had finished task 5.
They were sitting in the queue (`Priority` / `QOSMaxGRESPerUser`) while
six late CKA array tasks occupied every allowed L40S.

Runs reported as `task_0` were **not a resume bug**. Status prints the
latest *completed* task-boundary pickle. Those CKA jobs started on
Sep 15 after the Sep 10 cancel of `17270289_[16-22]`, wrote `task_0.pkl`,
and were already on task 1. Sparse SAC still has no usable checkpoints
(`InvalidArgumentError` in `tf.data` when four SAC processes share a GPU).

## Mathematical objective

Unchanged from the 9-baseline rerun. Each cell is single-task contrastive
GCRL with a task-boundary rule on actor `θ` and critic `φ`:

- Reset (R): reinitialise network, optimiser, and replay.
- Persistent (P): keep parameters, targets, and optimiser; drop replay.

This launch does **not** train CKA or SAC.

## Code and configuration changes

- `experiment_configs_paper_rp_priority_10seed.py`: 40 runs, same
  `native_info` / width-1024 / `k_max=5` flags as the 9-baseline config.
  Packs are ordered by remaining work (8× task-7, 1× task-6, 1× task-5)
  so almost-done seeds share a GPU and are not mixed with CKA.
- Checkpoint paths are identical to the 9-baseline rerun, so
  `run_continual_contrastive.py` auto-resumes from the existing
  `task_{k}.pkl`.
- `DRAFT_paper_rp_priority_10seed.sh`: 10 array tasks, 4 learners per
  L40S, 48h `afterany` chain (max 4 hops).
- `scripts/paper_rp_priority_status.py` and a small unit test.

Competing `paper9base` / `paper_sac` jobs were cancelled so the account
`QOSMaxGRESPerUser` cap (~6 L40S) is used only by these 40 runs.

## Launch command

```bash
sbatch DRAFT_paper_rp_priority_10seed.sh
python scripts/paper_rp_priority_status.py --summary
```

## Logged metrics

Unchanged: success/return and representation metrics as in the 9-baseline
rerun, W&B groups `PAPER-9BASELINE-10SEED-{reset|persistent}-{...}`.

## Validation

```bash
python tests/test_paper_rp_priority_10seed.py
```

## Known limitations

- Mid-task replay is still not serialised. Cancelling the CKA jobs
  discarded their in-progress task-1 env steps; CKA `task_0.pkl` files
  remain for a later resume.
- Sparse SAC is still broken under 4-way packing and was not relaunched.
- A full 10-task cell still needs ~16–40h after resume. With a 6-GPU
  cap, 10 packs cannot all run at once; arrays 0–7 should finish first.
