# Success-propagation diagnostics: Task5 vs push (D1–D4)

Date: 2026-09-08
Status: implemented + submitted (SLURM `17188038`, W&B `TASK58-SUCCESS-PROP-1M`)

## Motivation

Critic probes show Task5 often separates success vs hover under \(g_{\text{task}}\),
yet HER success mass stays ~3–5% after discovery while push rises toward
~10–35%. The open question is **why success does not propagate into the
HER/actor training distribution** after discovery.

Grounding: SGCRL + Demystifying emergent exploration — after success, a
similarity “trace” should form so the actor keeps revisiting the successful
path and refill replay. We measure where that loop breaks.

## Changes

### Code
- `contrastive/success_propagation_probe.py`: D2 success-trace, D3 actor-follow,
  D4 successful-episode Reverb injection (no Success-BC).
- `run_continual_contrastive.py`: flags + post-eval wiring; D1 print already
  logs the full A/B/C/D score matrix.
- `contrastive/continual_config.py`, `DRAFT.sh`: new flags.
- Configs/launcher: `experiment_configs_task58_success_propagation.py`,
  `DRAFT_task58_success_propagation.sh`.
- Tests: `tests/test_success_propagation_probe.py`,
  `tests/test_task58_success_propagation.py`.

### Experiment matrix (12 × 1M)
| idx | variant | task | D4 inject |
|---:|---|---|---|
| 0–2 | handle_measure | handle_press_side | off |
| 3–5 | push_measure | push | off |
| 6–8 | handle_inject | handle_press_side | on (N=256 @ ≥20% success) |
| 9–11 | push_inject | push | on |

All cells: DCC decomposed, corrected wrapper, HER-phase log, D2+D3 on;
Task5 also runs D1 critic-phase probe.

## Launch

```bash
sbatch DRAFT_task58_success_propagation.sh
```

W&B group: `TASK58-SUCCESS-PROP-1M`

## Metrics to read (stdout + W&B)

After first eval success ≥ 0.2:

1. **HER mass**: `her_phase/frac_success`, `frac_success_or_progress`
2. **D2 trace**: `trace/score_success_ep_mean` vs `trace/score_fail_*`, gap
3. **D3 follow**: `follow/pi_is_argmax_frac`, `follow/score_gap_argmax_minus_pi_mean`
4. **D4**: `inject/n_transitions` then whether HER/eval rise vs measure twin
5. **D1 (handle)**: A/B/C/D cells in `[critic phase probe]`

## Predictions

| Outcome | Interpretation |
|---|---|
| Push: trace gap rises + HER fills; Task5: gap flat + HER flat | Missing success trace on Task5 |
| Task5: critic argmax ≠ π at hover (`pi_is_argmax` low, gap large) | Actor cannot use critic |
| Task5 inject → lasting HER fill + retention | Scarcity was the bottleneck |
| Task5 inject → no propagation | Representation/actor still fails with data present |

## Limitations

- D1 geometric families are Task5/8-specific; push relies on D2/D3 + HER-phase.
- Injection writes full successful episodes so HER futures stay success-like;
  isolated `(s,a,g_task)` rows would be HER-relabeled away.
- Probe/trace/follow add eval-time rollouts (extra compute, not train steps).
- W&B may still drop some probe logs if step ordering conflicts; trust stdout.
