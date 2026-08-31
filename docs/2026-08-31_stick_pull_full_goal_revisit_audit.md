# Stick-Pull full state–goal revisit audit

Date: 2026-08-31
Status: implemented; rerun `DRAFT_stick_pull_wrapper_smoke.sh` to populate v4 metrics.

## Motivation

The corrected Stick-Pull wrapper exposes an audited 11-dimensional
reachable-success goal to DCC. The v3 smoke test verified that the fixed goal
is emitted correctly and that reward/info match the insertion predicate, but it
did not report how closely successful scripted-policy states revisit that full
goal. Task-5/Task-8 already log these diagnostics
(`full_goal_linf_error_at_success`, `full_goal_visits_within_*`).

## Change

- Extended `scripts/smoke_test_stick_pull_corrected_wrapper.py`:
  - `full_goal_errors()` compares visited `state[:11]` against `goal[:11]`.
  - Per-success aggregates for hand, gripper, stick COM, handle, and insertion
    margin distances.
  - Revisit counters at tolerances `1e-6`, `1e-3`, and `1e-2`.
  - `minimum_any_state_full_goal_linf_error` over the training horizon.
  - `closest_successful_state` among successful steps.
- Protocol bumped to `stick_pull_corrected_wrapper_smoke_v4`.
- Pass/fail gates unchanged; revisit metrics are diagnostic only.

## Launch

```bash
sbatch DRAFT_stick_pull_wrapper_smoke.sh
```

Dependency-light checks (no MuJoCo):

```bash
conda activate contrastive_rl
python tests/test_stick_pull_wrapper_smoke.py
python tests/test_stick_pull_reachable_success_goal.py
```

## How to read the new metrics

- `full_goal_linf_error_at_success_*`: distance between visited state and fixed
  goal on steps where the corrected success predicate is true.
- `full_goal_visits_within_1e-2`: count of successful steps with L∞ error ≤
  0.01 across all 11 coordinates.
- `minimum_any_state_full_goal_linf_error`: closest approach to the full goal
  during the rollout, even on non-success steps.
- Low revisit counts do **not** fail the audit. Success remains handle distance
  plus insertion; hand/gripper coordinates in the goal are auxiliary.

## Known limitations

- Metrics are logged only within the 150-step training horizon.
- One fixed canonical goal remains hardcoded in `env_utils.py`; revisit stats
  measure consistency with that representative state, not uniqueness of success.
