# Uniform Success-BC: recency vs λ warmup on 7 then 5/8

Date: 2026-09-21
Status: recency sampling implemented. Task 7 warmup already running as
`18064660` (with Task 4). This grid tests recency on Task 7, then both
ideas on Tasks 5 and 8.

## Why recency

A uniform algorithm cannot use warmup on 7 and full λ on 5/8. Recency
keeps the large buffer (diversity) and **full λ from the first terminal
success** (what 5/8 need), but BC samples more from newer successes
(half-life 4 episodes). On Task 7, if a better place arrives later, it
is cloned more than the first local optimum. On 5/8 the latest latch
is what you want to keep.

Warmup is the other idea: same last-bit insert, uniform samples, λ
starts small and rises with `|D_succ|`.

## Launch

Task 7 warmup is `18064660`. The rest:

```bash
sbatch DRAFT_jubail_success_bc_uniform.sh
```

W&B group `PAPER-DCC-SUCCESS-BC-UNIFORM`. Array 0–49, seeds 5–14, 8M.

## Validation

```bash
python tests/test_success_bc_uniform.py
python tests/test_success_bc_warmup.py
```
