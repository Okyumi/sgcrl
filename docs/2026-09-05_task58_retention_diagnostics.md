# Task-5 retention diagnostics: HER / freeze / critic phase probe

Date: 2026-09-05
Status: launcher ready

## Motivation

Handle-press (and window-close) DCC runs discover success near ~100k, then
often lose `mechanism_moved` / success by ~250k. Two competing accounts:

1. **Fake-goal hover** (user conjecture): critic scores a nearby non-pressed
   state almost as highly as the fixed task goal; actor rushes there and stays.
2. **Phase drowning + nonstationary critic**: HER futures are dominated by
   reach/hover; critic keeps fitting that majority; press becomes rare and is
   abandoned. Distances at the drop get *worse*, which favors (2) over (1).

This sweep implements the first three falsifiers from the 2026-09-05
discussion, with an online critic phase probe on every cell.

## Mathematical / algorithmic changes

### Online critic phase probe
After each eval, roll the deterministic policy, classify transitions into
`success` / `hover` / `mid_reach`, and log
`φ(s,a)ᵀψ(g_task)` aggregates (`probe/score_*`, `probe/gap_success_minus_*`).

Interpretation at peak vs drop:
- hover ≈ success ≫ mid-reach → supports fake-goal conjecture
- success collapses vs hover/reach while hand–handle distance worsens →
  phase drowning / critic nonstationarity

### HER future sampling (InfoNCE discount unchanged)
New flag `her_future_sampling_mode` with separate `her_future_discount`
(defaults preserve legacy γ^Δt using `ContrastiveConfig.discount` for HER
only when `her_future_discount < 0`):

| Mode | Behaviour |
|---|---|
| `discounted` | legacy `γ^Δt` |
| `uniform` | equal weight on futures (`her_future_discount=1`) |
| `final_state` | always last episode state as goal |
| `success_oversample` | boost futures within L2 0.05 of reachable success goal |

### Freeze critic after success
`freeze_critic_after_success_rate=0.3` (min 50k steps): once eval success
hits the threshold, critic/dyn writes are reverted each step while the actor
keeps updating against the frozen landscape.

## Matrix (15 cells, handle-press only, 1M steps)

| Index | Variant | Seeds |
|---|---|---|
| 0–2 | `her_discounted` | 5/6/7 |
| 3–5 | `her_uniform` | 5/6/7 |
| 6–8 | `her_final_state` | 5/6/7 |
| 9–11 | `her_success_oversample` | 5/6/7 |
| 12–14 | `freeze_critic_0p3` | 5/6/7 |

Shared: corrected wrapper, full network, `dyn_aux_weight=1.0`, eval+probe
every 50k, mid-task checkpoints every 50k, videos every 100k.

## Launch

```bash
sbatch DRAFT_task58_retention_diagnostics.sh
```

W&B group: `TASK58-RETENTION-DIAGNOSTICS-1M`

Offline re-probe of a mid-task snapshot:

```bash
python scripts/probe_critic_phases.py \
  --checkpoint logs/task58_retention_diagnostics_v1/checkpoints/.../task_0_step_100000.pkl \
  --env-name sawyer_handle_press_side \
  --episodes 10 \
  --output /tmp/probe_100k.json
```

## Validation

```bash
python tests/test_task58_retention_diagnostics.py
python tests/test_critic_phase_probe.py
```

## Known limitations

- Probe families are geometry heuristics; early training may yield empty
  `success` buckets (metrics NaN) until the policy presses.
- Offline probe assumes Task58 DCC residual widths (1024 / phi 256x4).
- Freeze keeps the actor entropy/success-buffer path; it does not snapshot
  an explicit frozen ψ copy beyond freezing parameter writes.
- Handle-press only in this sweep; window-close can reuse the same flags.

## Note on "bad task design"

The sparse press is harder for HER+discount CRL than continuous cube
transport because informative futures occupy a short contact phase, not
because the Meta-World reward is misspecified. Cube push fills the
trajectory with object-progress futures that match the goal geometry;
handle press mostly does not. The ablations above test that claim without
changing the env reward.
