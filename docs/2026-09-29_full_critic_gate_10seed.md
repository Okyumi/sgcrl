# Paper 10-seed fill for gated DCC (seeds 5–14)

Date: 2026-09-29  
Status: seeds 5–9 already running/complete; seeds 10–14 launched from scratch
with `DRAFT_jubail_full_critic_gate_10seed_fill.sh`.

## Motivation

Paper tables use seeds 5–14. The first from-scratch gated-DCC grid only
covered 5–9. This fill adds 10–14 with the same stack, checkpoint root, and
W&B group so the method has a full 10-seed curriculum.

## Method

Same unified algorithm as the 5-seed grid: decomposed critic, reset actor,
whole-episode success matching, local σ_s critic gate. Paper stack:
`native_info`, dyn=0, 1024×4, λ=0.1, 4096 ring, 8 probes, ε=0.1. 8M × 10
tasks, `start_task=0`. Configs 5–9 of
`experiment_configs_full_critic_gate_5seed.py` (now seeds 5–14).

## Launch

```bash
sbatch DRAFT_jubail_full_critic_gate_10seed_fill.sh
```

Offset 5 / limit 5, array 0–4 → seeds 10–14. Does not touch the seed-5
chain (`CONFIG_LIMIT=5`, offset 0). Same W&B group
`PAPER-DCC-FULL-CRITICGATE-5SEED`.

## Seed diagnostics (5–9)

T5 failures are **found-then-lost**, not “never found”. End-of-task W&B
summaries show the critic gate stays high (~0.81–0.97) on collapse seeds
too. Discriminating stats:

- Collapse (seeds 5, 8; seed 6 never latched): `source_success_fraction`
  0.05–0.20, σ_s ~27–42.
- Retain (seeds 7, 9): `source_success_fraction` 0.69–0.76, σ_s ~84–115.

T4 seed 7 is the opposite failure: **never found** (peak=last=30%).
`source_success_fraction` 0.098, lowest T4 σ_s (4.16) and gate (0.736).

## Validation

```bash
python tests/test_full_critic_gate_5seed.py
```

## Limitations

Seed 5 still finishing T7–T9. Gate/σ time series are end-of-task summaries
only (`scan_history` hangs). Seeds 10–14 will take ~2–3 chained 48h hops.
