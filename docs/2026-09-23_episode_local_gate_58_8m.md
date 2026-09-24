# Tasks 5/8 gated episode-BC at 8M

Date: 2026-09-23  
Status: submitted on Jubail (`DRAFT_jubail_episode_local_gate_58_8m.sh`).

## Motivation

The 1M gated probes (array 18131472 cells 2–3) finished at 80% (handle)
and 50% (window). Ungated DCC+BC latches to ~100% over the full 8M
budget. This run keeps the same algorithm (whole-episode
`episode_sparse_reward` + local \(\sigma_s\) gate) and gives Tasks 5/8
that 8M budget.

Tasks 4/7 from 18131472 stay running. Task 7 is still in the seed-6
DCC no-success window (~2.2M); do not relaunch it.

## Method

Unchanged from `docs/2026-09-23_episode_local_gate.md`. From scratch
(`single_task`), seed 6, λ=0.1, eval every 100k. Separate log and
checkpoint directories so the finished 1M `task_0.pkl` is not resumed.

## Launch

```bash
mkdir -p logs/episode_local_gate_58_8m/runs logs/episode_local_gate_58_8m/checkpoints
sbatch DRAFT_jubail_episode_local_gate_58_8m.sh
```

W&B `nyuad_mmvc/continual_gcrl_paper` /
`PAPER-DCC-EPISODE-LOCALGATE-58-8M`. Array 0–1: T5 8M, T8 8M.

## Validation

```bash
python tests/test_episode_local_gate_58_8m.py
```

## Limitations

Seed 6 only, from scratch (not sequential resume). Compare 5/8 to
ungated DCC+BC; keep waiting on Task 7 vs plain DCC seed 6.
