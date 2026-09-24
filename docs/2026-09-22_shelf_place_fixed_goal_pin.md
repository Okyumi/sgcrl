# Task 7: pin the shelf mesh to the commanded goal

Date: 2026-09-22  
Status: wrapper fix in `env_utils.SawyerShelfPlace.reset`.

## Motivation

`fix_goals=True` sets the Task-7 success center and observation goal to
\((0.02, 0.89, 0.30)\). MetaWorld `shelf-place-v2` still randomizes the
shelf **body** on every reset (`random_init`). The wrapper overwrote
`_goal` / `_target_pos` without moving that body, so the camera showed a
wandering shelf while success was scored at a fixed xyz that often sat
off the physical deck. Object can then "succeed" in mid-air and fall.

This is a wrapper bug, not a Success-BC labeling change.

## Rule

On reset with `fixed_start_end` set, keep MetaWorld's randomized object
start, then apply the same body placement as `reset_model`:

\[
\text{shelf body} = g - (0,0,0.3),\qquad
g = (0.02, 0.89, 0.30).
\]

The XML goal site is local \((0,0,0.3)\) on that body, so the green site,
`_target_pos`, and the observation goal coincide. `synchronize_simulator_after_reset`
forwards MuJoCo so site xpos matches the body write.

## Code

- `env_utils.py` `SawyerShelfPlace.reset`
- `tests/test_shelf_place_fixed_goal_pin.py`

No actor, buffer, or label-mode change.

## Validation

```bash
python tests/test_shelf_place_fixed_goal_pin.py
```

Source check: reset pins `shelf` and forwards. Simulator check (when
MetaWorld imports): five resets keep the shelf and site at \(g\), while
object xy still varies.

## Limitations

Already-collected Task-7 rings and GIFs were recorded under the old
mismatch. Retrain to see whether Success-BC vs DCC on Task 7 was an
artifact of cloning "success" off the shelf. The 10-seed retrain is
`docs/2026-09-22_terminal_bc_shelf_pin.md`
(`sbatch DRAFT_jubail_terminal_bc_shelf_pin.sh`).
