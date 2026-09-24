# 10-seed gated episode-BC on Tasks 4, 7, 8

Date: 2026-09-23  
Status: submitted on Jubail (`DRAFT_jubail_episode_local_gate_478_10seed.sh`).

## Motivation

Paper tables need mean±std on the gated method, not seed 6 alone.
Tasks 4/7 are scored against plain DCC; Task 8 against DCC+BC.
The 1M Task-8 probe (50%) was too short to decide latch retention.

## Method

Same algorithm as `docs/2026-09-23_episode_local_gate.md`:
`episode_sparse_reward` + local \(\sigma_s\) gate, λ=0.1, paper stack.
Seeds 5–14. Each cell is one 8M task resumed from the paper Success-BC
predecessor (`task_3` / `task_6` / `task_7`).

## Launch

```bash
mkdir -p logs/episode_local_gate_478_10seed/runs \
         logs/episode_local_gate_478_10seed/checkpoints
sbatch DRAFT_jubail_episode_local_gate_478_10seed.sh
```

W&B `nyuad_mmvc/continual_gcrl_paper` /
`PAPER-DCC-EPISODE-LOCALGATE-478-10SEED`. Array 0–29.

Seed-6 T4/T7 in 18131472 is the same recipe; this array reruns them in
the paper group. T5 is not in this grid.

## Validation

```bash
python tests/test_episode_local_gate_478_10seed.py
```

## Limitations

Resume is paper Success-BC, not plain DCC. T8 is sequential task 8, not
from scratch. One seed of T8 may hit H200 EGL as before.
