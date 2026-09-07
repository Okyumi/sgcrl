# Task-5 actor retention v3: train–eval goals + critic-guided success

Date: 2026-09-07
Status: launched

## Motivation

Retention diagnostics v2 showed:

- The critic usually still prefers press over hover under \(g_{\text{task}}\)
  at collapse (fake-goal largely falsified).
- Mild HER reweights, freeze-critic, and entropy-off do not retain success.
- Terminal Success-BC *does* solve Task 5 (~0.8–0.9), by forcing the actor
  to clone successful actions from a success buffer.

Open question: why can the actor not retain press from the critic signal
alone? Leading hypotheses:

1. **Train–eval goal mismatch.** Actor updates use HER futures; eval uses
   the fixed desired goal.
2. **Rare-mode drowning.** Press transitions are too rare in the actor batch
   even when the critic would score them highly under \(g_{\text{task}}\).
3. **ψ aliasing (to measure).** \(\psi(g_{\text{task}}) \approx \psi(g_{\text{hover}})\)
   would be a *different* claim from phase ranking under one fixed goal.

HER `success_oversample` (v2) is **not** the same as Success-BC: it only
reweights future goals in the critic/actor HER batch, and it failed. This
sweep tests actor-side interventions instead.

## Code changes

- `actor_goal_mode ∈ {her, task, mix}`: actor loss conditions on HER futures,
  env desired goal, or a 50/50 mix (`actor_task_observation` extras).
- `actor_success_score_weight`: maximise \(\phi(s,\pi)^\top\psi(g)\) on the
  terminal success buffer (policy actions; **no** action cloning).
- Critic phase probe extended with `embed_fn`: logs
  `probe/psi_l2_task_vs_hover`, `probe/psi_cosine_task_vs_hover`, and
  cross-goal score swaps under a mean hover-state goal.
- Checkpoint identity includes actor-goal / success-score / Success-BC
  settings (`_ret_*` fingerprint).
- `DRAFT.sh` wires `ACTOR_GOAL_MODE`, `ACTOR_SUCCESS_SCORE_WEIGHT`,
  `SUCCESS_BC_LABEL_MODE`.

## Matrix (15 cells, handle-press, 1M, seeds 5/6/7)

| Index | Variant | Tests |
|---|---|---|
| 0–2 | `dcc_control` | Baseline + extended ψ probe |
| 3–5 | `actor_goal_task` | Train–eval goal match |
| 6–8 | `actor_goal_mix` | Partial goal match |
| 9–11 | `actor_success_score` | Critic signal on success states without BC |
| 12–14 | `success_bc_terminal` | Positive control (action cloning) |

Shared: corrected wrapper, full network, probe every eval, mid-ckpts 50k,
videos 100k. Roots: `logs/task58_actor_retention_v3/`.

## Launch

```bash
sbatch DRAFT_task58_actor_retention_v3.sh
```

W&B group: `TASK58-ACTOR-RETENTION-1M-V3`

## Interpretation

| Outcome | Implication |
|---|---|
| `actor_goal_task` retains | Train–eval goal mismatch was primary |
| `actor_success_score` ≈ Success-BC | Critic signal on success states is enough; cloning not required |
| Only Success-BC retains | Actor needs explicit action targets / basin escape |
| High `psi_cosine_task_vs_hover` + similar cross-goal scores | Goal embedding aliasing worth addressing |
| Low cosine + hover-goal flips press preference | Geometric anti-press under HER goals |

## Validation

```bash
python tests/test_task58_actor_retention_v3.py
python tests/test_critic_phase_probe.py
bash -n DRAFT_task58_actor_retention_v3.sh
```

## Known limitations

- Success-BC positive control burns three seeds; treat as replication, not
  discovery.
- Hover goal for ψ distance is the mean hover *state* used as a HER-like
  future, not a hand-crafted unpressed mechanism target.
- Probe still depends on the current policy producing success/hover buckets;
  empty buckets yield NaNs.
