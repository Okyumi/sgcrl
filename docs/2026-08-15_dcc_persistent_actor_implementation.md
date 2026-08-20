# DCC persistent-actor ablation

Date: 2026-08-15

## Purpose

Test whether the reset actor used in the reported Sawyer DCC experiments
discarded useful policy-side transfer. The active batch contains two matched
ten-task cells:

| Cell | Actor | Critic | Dynamics weight | Seeds |
|---|---|---|---:|---|
| DCC + dynamics + persistent actor | persistent | decomposed | 1.0 | 5, 6, 7 |
| DCC without dynamics + persistent actor | persistent | decomposed | 0.0 | 5, 6, 7 |

These seeds match the DCC seeds reported in the RLC paper. The nine-cell
contrastive baseline grid used seeds 97--101, and sparse SAC R/R and P/P used
seeds 1--3, according to the W&B exports under `results/data/`.

## Actor lifecycle

For `actor_mode=persistent, critic_mode=decomposed`, the runner now carries:

- policy parameters;
- policy Adam state;
- adaptive entropy parameter;
- entropy-optimizer state.

The replay buffer and task-specific critic encoder `phi_task` still reset at
every task boundary. The shared DCC components `b_shared`, `h_phi`, `h_dyn`,
and `psi`, together with their optimizer states, continue to persist exactly
as before. `actor_mode=reset` retains the original DCC behavior.

All actor state is checkpointed. A persistent-actor resume at task `k > 0`
fails explicitly if the checkpoint lacks that state, preventing a silent
fallback to a reset actor.

## Active configuration batch

`experiment_configs.py` resolves to exactly six runs. Previously submitted
DCC-SAC, AC-DCC, and task-5/task-8 diagnostic cells remain in
`ARCHIVED_CELLS` for provenance but are not enumerated.

## Validation

- Four dependency-light persistent-actor checks passed.
- All Python files parsed and compiled.
- All four contrastive launchers passed `bash -n`.
- Configuration enumeration produced exactly six full-curriculum runs.
- `git diff --check` passed.

The current runtime does not provide the project JAX/Acme/Reverb stack, so a
short two-task cluster smoke is still required before the full six-run batch.
