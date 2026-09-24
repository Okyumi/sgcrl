# Success-BC λ warmup (terminal last-bit, large D_succ)

Date: 2026-09-21
Status: implemented + submitted on Jubail (`DRAFT_jubail_success_bc_warmup.sh`).

## Motivation

Task 7 lock-in is cloning the first terminal successes at full
`λ=0.1` while `D_succ` is still one (or a few) 150-step episodes.
Tasks 5/8 are the opposite: they need **full λ on the first rare
press/close**, or the latch is lost. This experiment therefore applies
warmup only to Tasks 4 and 7. Tasks 5/8 stay at constant `λ=0.1`.

## Method

Paper-legal: insert a full-horizon episode iff the last bit is 1
(`terminal_episode`, Appendix C \(Y(\tau)\)). Capacity stays 4096
(~27 episodes). BC still samples uniformly from whatever is in the
ring.

The only change is the scale on λ:

```
λ_eff = λ * min(1, |D_succ| / (N0 * 150))
```

with `λ=0.1`, `N0=8`. First episode: `λ_eff = 0.0125`. Eight stored
episodes (1200 transitions): full `λ`. More data does not shrink the
buffer; it only raises λ toward the paper value.

If Tasks 4/7 never fill: they still clone the episodes they have, at
a smaller λ. They do not wait.

Tasks 5/8 are **not** in this grid.

## Paper

Eq. 7 currently uses a constant `λ_succ` (and zero while `D_succ` is
empty). If Task 7 holds, the paper can keep constant λ on latch tasks
and use `λ_succ(n)` only where first-success lock-in is the failure.
Do not invert the schedule on 5/8 (that would decay λ after the press
and can drop the latch).

## Launch

```bash
sbatch DRAFT_jubail_success_bc_warmup.sh
```

W&B: `nyuad_mmvc/continual_gcrl_paper`, group
`PAPER-DCC-SUCCESS-BC-WARMUP`. Array 0–19, seeds 5–14, 8M, resume
paper Success-BC `task_3.pkl` / `task_6.pkl`.

## Validation

```bash
python tests/test_success_bc_warmup.py
```

Pass bar: Task 4 stays in the terminal-BC / plain-DCC band; Task 7
should not collapse after the first 10% hits. Tasks 5/8 warmup/recency
are in `docs/2026-09-21_success_bc_uniform.md`.
