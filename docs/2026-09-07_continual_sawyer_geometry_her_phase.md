# Continual Sawyer geometry audit + HER-phase lucky-spike measurement

Date: 2026-09-07
Status: launched (`TASK58-HER-PHASE-GEOMETRY-1M`)

## Correction to the earlier claim

It is **not** true that only Task-5 (handle-press) and Task-8
(window-close) have short-contact / mechanism-jump structure. In the
10-task continual sequence, continuous object-progress tasks are the
exception; several others are contact-heavy.

| k | Task | Geometry family | Short-contact / jump risk? |
|---|---|---|---|
| 0 | `sawyer_hammer` | multi_phase_contact | **yes** (grasp + strike; nail joint) |
| 1 | `sawyer_push_wall` | continuous_object_progress | no |
| 2 | `sawyer_faucet_close` | short_contact_mechanism | **yes** (handle rotation after contact) |
| 3 | `sawyer_push_back` | continuous_object_progress | no |
| 4 | `sawyer_stick_pull` | multi_phase_contact | **yes** (grasp, insert, pull) |
| 5 | `sawyer_handle_press_side` | short_contact_mechanism | **yes** |
| 6 | `sawyer_push` | continuous_object_progress | no |
| 7 | `sawyer_shelf_place` | multi_phase_contact | **yes** (pick + place) |
| 8 | `sawyer_window_close` | short_contact_mechanism | **yes** |
| 9 | `sawyer_peg_unplug_side` | short_contact_mechanism | **yes** |

Empirical hardness in prior metrics already flagged **k ∈ {5, 8, 9}**
(and representation rank crashes there). Push-family tasks (k=1,3,6)
are the clean contrast class for “HER futures ≈ partial progress toward
the same coordinates.”

So the earlier story should be: **contact / mechanism tasks are
over-represented among failures**, not “only handle and window.”

## Lucky-spike story — previously untested

The narrative was:

1. A few press trajectories insert success-like HER futures.
2. Actor briefly gets press-supporting gradients.
3. Buffer refills with approach / hover futures.
4. Most actor updates again say “match nearby unpressed futures.”
5. Policy at \(g_{\text{task}}\) collapses to hover even if a probe under
   \(g_{\text{task}}\) still likes press states when they appear.

That was a **plausible explanation**, not a measurement. This change
logs the composition of **HER goals actually consumed by the learner**.

## Measurement: `her_phase/*`

Module: `contrastive/her_future_phase.py`.

After each `learner.step()`, classify the goal half of the latest
replay batch (HER future states) into:

- `success`
- `hover_unsolved` (near object/mechanism, not solved)
- `object_progress` (push family only: object closer to target)
- `far_reach`
- `other`

Logged as EMA fractions (`learner/her_phase/frac_*`) plus stdout
`[her phase @ steps]`.

**Falsification:**

- If after a success spike on handle-press, `frac_success` stays high
  while eval collapses → lucky-spike buffer story is **wrong** (failure
  is elsewhere).
- If `frac_hover_unsolved` dominates after the spike while
  `frac_success` falls → story is **supported**.
- If push keeps high `frac_success_or_progress` while contact tasks do
  not → geometry-family hypothesis is **supported**.

## Launch

```bash
sbatch DRAFT_task58_her_phase_geometry.sh
```

12 cells (push / handle / faucet / peg × seeds 5/6/7), 1M steps,
probe + HER-phase logging. W&B group: `TASK58-HER-PHASE-GEOMETRY-1M`.

## Validation

```bash
python tests/test_her_future_phase.py
python tests/test_task58_her_phase_geometry.py
```

## Limitations

- Phase labels are geometric proxies, not simulator contact sensors.
- Hammer / stick / shelf use simplified object-distance success proxies
  in this logger; the comparative sweep focuses on push vs
  handle/faucet/peg.
- Logging uses the first `batch_size` rows of each learner batch (not
  every SGD substep) for CPU cost.
