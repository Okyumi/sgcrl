# Task-5 retention diagnostics v2 (checkpoint fix + entropy-off)

Date: 2026-09-06
Status: launcher ready

## Motivation

v1 (`TASK58-RETENTION-DIAGNOSTICS-1M`) only fully trained four cells
(`her_discounted`×3 and `her_uniform` seed 5). Cells 4–14 auto-resumed a
shared `task_0.pkl` because HER/freeze settings were not part of
`_ckpt_path`. Probe evidence from those four runs still largely falsifies
the fake-goal conjecture; HER/freeze/entropy conclusions require this v2
re-run.

## Checkpoint identity fix

`run_continual_contrastive._ckpt_path` now appends
`_ret_{fingerprint}` whenever any of:

- `her_future_sampling_mode != discounted`
- `her_future_discount >= 0` (explicit uniform discount)
- `freeze_critic_after_success_rate >= 0`
- `use_action_entropy == false`

Legacy discounted DCC (entropy on, no freeze) keeps the old path.

## New flag

`--use_action_entropy` / `--nouse_action_entropy` (default true). When
false, the actor loss drops the entropy bonus while keeping
`entropy_coefficient=None` (decomposed learner requirement).

## Matrix (15 cells, handle-press, 1M)

| Index | Variant | Notes |
|---|---|---|
| 0–2 | `her_uniform` | equal future weights |
| 3–5 | `her_final_state` | last-state goals |
| 6–8 | `her_success_oversample` | boost near reachable success |
| 9–11 | `freeze_critic_0p3` | freeze critic after eval ≥ 0.3 |
| 12–14 | `discounted_entropy_off` | legacy HER, actor entropy off |

Shared: corrected wrapper, full network, probe every eval, mid-ckpts 50k,
videos 100k. Separate roots under `logs/task58_retention_diagnostics_v2/`.

## Launch

```bash
sbatch DRAFT_task58_retention_diagnostics.sh
```

W&B group: `TASK58-RETENTION-DIAGNOSTICS-1M-V2`

Compare against v1 discounted baselines (`i1lzx3tb`, `uiqyz589`, `fy8d2k22`).

## Validation

```bash
python tests/test_task58_retention_diagnostics.py
python tests/test_critic_phase_probe.py
```

## Known limitations

- v1 discounted cells are not re-run here; use W&B/logs from v1 as the
  entropy-on discounted control.
- Probe metrics should be read from stdout/`[critic phase probe]` until
  W&B step collision for probe logs is cleaned up.
- Entropy-off removes exploration pressure in the actor objective; early
  discovery may worsen even if late exploitation improves.
