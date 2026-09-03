# Task-5 success-buffer BC retry with eval videos

Date: 2026-08-31
Status: launcher ready

## Motivation

Dyn-ablation logs show handle-press eval peaks with high `mechanism_moved`,
then later drops with `approach` sometimes still positive but `moved=0`.
An earlier Outcome-Calibrated BC attempt (stage-3 `raw-H25-retention`,
`success_bc_weight=0.1`, `action_effect_actor_mode=effect_only`) did not
help. This retry uses the corrected Task-5 wrapper, full-network DCC,
**combined** actor mode (DCC Q + outcome head, not effect-only), and W&B
eval videos every 100k steps.

## Matrix (6 cells)

| Index | Variant | Critic | BC | Seeds |
|---|---|---|---|---|
| 0–2 | `dcc_baseline` | `decomposed` | off | 5, 6, 7 |
| 3–5 | `success_bc_combined` | `advantage_decomposed` | 0.1 | 5, 6, 7 |

Shared: 4M steps, corrected wrapper, full-state goal, full network,
`dyn_aux_weight=1.0`, eval every 50k (10 episodes + task58 stages).

## Launch

```bash
sbatch DRAFT_task58_success_bc_video.sh
```

W&B group: `TASK58-SUCCESS-BC-VIDEO-4M`

Videos: `evaluator/rollout_video` on the Media tab, keyed by
`evaluator/env_steps` (every 100k).

## Validation

```bash
python tests/test_task58_success_bc_video.py
python tests/test_task58_reachable_success_goals.py
```

## Known limitations

- BC cells require `advantage_decomposed`; baseline uses plain `decomposed`
  (not a perfect control for the extra head alone).
- Prior stage-3 falsification used `effect_only`; this retry deliberately
  uses `combined` so contrastive Q still drives the actor.
- Outcome success labels use 7-D L2 threshold 0.05, not the 0.02 axis gate.
