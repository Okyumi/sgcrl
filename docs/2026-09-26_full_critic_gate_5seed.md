# From-scratch 5-seed gated DCC (tasks 0–9)

Date: 2026-09-26  
Status: submitted on Jubail (`DRAFT_jubail_full_critic_gate_5seed.sh`).

## Motivation

Paper tables for the proposed method need a full curriculum, not
resume-from-Success-BC probes on Tasks 4/7/8. This grid trains the
unified algorithm from task 0.

## Method

Decomposed critic, reset actor, whole-episode success matching, local
critic gate. Paper stack: `native_info`, dyn=0, 1024×4, λ=0.1, 4096
ring, 8 probes, ε=0.1. Seeds 5–9. 8M steps × 10 tasks. From scratch
(`start_task=0`, no resume file). Chain hops auto-resume from the
latest `task_k.pkl`.

## Launch

```bash
mkdir -p logs/full_critic_gate_5seed/runs \
         logs/full_critic_gate_5seed/checkpoints
sbatch DRAFT_jubail_full_critic_gate_5seed.sh
```

W&B `nyuad_mmvc/continual_gcrl_paper` /
`PAPER-DCC-FULL-CRITICGATE-5SEED`. Array 0–4, one seed per A100.
Each cell chains up to 8 × 48h hops.

## Validation

```bash
python tests/test_full_critic_gate_5seed.py
```

## Limitations

Originally five seeds (5–9). Paper 10-seed fill (10–14) is
`DRAFT_jubail_full_critic_gate_10seed_fill.sh`; see
`docs/2026-09-29_full_critic_gate_10seed.md`. Mid-task death restarts
the current task; only completed `task_k.pkl` files resume. H200 is
avoided by requesting A100.
