# Jubail Task-5 action-advice vs mass-inject diagnostic

Date: 2026-09-10 (results 2026-09-11)
Status: finished on Jubail SLURM `17901349` (all 3 cells COMPLETED)

## Motivation

Torch is at GPU QOS, so the queued same-state action-advice probe and the
fair 1-seed 20% mass inject could not start there. Those are the two open
Task-5 tests from `why_task58_failed.md` / `docs/2026-09-10_action_advice_vs_state_preference.md`:

1. The critic ranks finished **success states** above **hover states** under
   \(g_{\text{task}}\). That does **not** prove it prefers a press **action**
   in a fixed hover state, nor that \(\nabla_a s\) at \(a_\pi\) points at press.
2. The earlier D4 inject was ~0.15% of the buffer and did not test scarcity.
   A mass-matched clone to ~20% of replay, still with `success_bc_weight=0`,
   is the remaining training counterfactual.

Jubail has no Torch checkpoints, so this job **trains** the 1M-step cells
here, writes mid-task snapshots every 50k, then runs
`scripts/measure_hover_action_advice.py` on those snapshots in the same
GPU allocation.

## Mathematical objective

Critic score \(s(s,a,g)=\phi(s,a)^\top\psi(g)\). After training we measure,
at hover states:

- \(s(s_{\text{hov}}, a_{\text{press}}, g) - s(s_{\text{hov}}, a_\pi, g)\)
  under \(g_{\text{task}}\) and under hover-goal \(g=s\)
- press rank among \(\{\pi, \text{press}, 32\text{ random}\}\)
- \(\mathrm{std}_a s / \mathrm{std}_s s\) (action- vs state-sensitivity)
- \(\cos(\nabla_a s|_{a_\pi},\, d_{\text{press}})\) and a local step toward press

Training inject (cell 2) clones successful episodes at the first eval
success \(\ge 20\%\) until replay success mass is about 20% of the current
buffer. No Success-BC.

## Code / config

- `set_up/jubail_hpc_env.sh`: Jubail CUDA/conda library path (from `draft_3.sh`).
- `DRAFT_jubail.sh`: modern flag builder copied from `DRAFT.sh`, Jubail env.
- `experiment_configs_jubail_task5_action_advice.py`: 3 cells, seed 6, 1M steps.
- `DRAFT_jubail_task5_action_advice.sh`: `#SBATCH --partition=nvidia`, array 0–2.
- `scripts/run_action_advice_from_checkpoints.py`: pick peak/late mid-ckpts.
- `tests/test_jubail_task5_action_advice.py`

| idx | variant | what |
|---:|---|---|
| 0 | `handle_measure` | Task 5, no inject; probe ~100k + latest |
| 1 | `push_measure` | Push comparison; probe ~250k + latest |
| 2 | `handle_inject_20pct` | Task 5, clone to ~20% buffer; probe latest |

Shared: DCC decomposed, full network (`adapt_heads_only=false`), 1024×4,
corrected Sawyer wrapper, HER-phase / dwell / press−π / critic-phase /
success-trace / actor-follow on. Videos every 100k.

## Launch

```bash
sbatch DRAFT_jubail_task5_action_advice.sh
```

Logs: `logs/jubail_task5_action_advice/runs/`
Checkpoints: `logs/jubail_task5_action_advice/checkpoints/`
W&B: project `continual_gcrl_paper`, group `TASK58-JUBAIL-ACTION-ADVICE-1SEED`

## Metrics to read

Stdout / W&B during training:

- `[stage dwell]` steps/150 on success episodes
- `[press vs pi]` same-state gap at hover under \(g_{\text{task}}\)
- HER `frac_success` / `frac_success_or_progress`
- D1 A/B/C/D critic-phase matrix; D2 success-trace; D3 actor-follow
- `[success inject]` n_transitions and target frac (cell 2)

JSON after each cell:

- `logs/jubail_task5_action_advice/runs/action_advice_*.json`

Interpretation:

- Handle `action_over_state_std_ratio_task << 1` and tiny press−π gap,
  while push is large → critic is scene-like on Task 5.
- Gap \(>0\) under \(g_{\text{task}}\) but \(\le 0\) under hover \(g\) → HER
  goal mismatch.
- 20% inject retains eval/HER → scarcity was the bottleneck.
- 20% inject still collapses → actor optimization / landscape remains.

## Validation

```bash
python tests/test_jubail_task5_action_advice.py
python experiment_configs_jubail_task5_action_advice.py --list
```

## Results (2026-09-11)

Job `17901349`: handle_measure on H100 (~1.1h), push_measure and
handle_inject_20pct on V100 (~2.3h). Final eval: handle 0%, push 90%,
inject 0%.

### Same-state action advice at hover

| | handle 100k | handle late (measure) | push 250k |
|---|---:|---:|---:|
| press−π gap under \(g_{\text{task}}\) | **−0.13** | +0.38 | **+2.04** |
| press beats π | 42% | 72% | **100%** |
| press rank (1 = best of 34) | 16.4 | 3.0 | **1.04** |
| \(\cos(\nabla_a s, d_{\text{press}})\) | **−0.21** | +0.29 | **+0.86** |
| local step toward press | −0.19 | +0.21 | **+1.44** |
| \(\mathrm{std}_a/\mathrm{std}_s\) | **0.075** | **0.047** | **1.95** |
| press−π gap under hover \(g\) | **−0.47** | **−0.20** | −0.22 |

Handle late measure numbers recovered from `17901349_0.out` (inject later
overwrote `action_advice_handle_press_side_s6_step951900.json`).

Handle critic is scene-like: scores vary across states, almost not across
actions. Push at discovery prefers the task-progress action and the local
gradient points that way. Under hover-goal \(g\), press loses on handle.

When both buckets exist, online D1 still usually has A>C (e.g. handle 100k
gap_s-h@task = +7.8). That **state** ranking coexists with useless **action**
advice at hover.

### 20% mass inject (fair scarcity test)

`[success inject @ 250500] n=50100 target=50100 eps=8 clones=326 attempts=80
buf≈250500 frac_target=0.20 (success=20.0%; no BC)`

HER succ rose 0.03 → 0.15 then decayed to 0.07. Eval 20% → 10% → 0% and
did not recover. `success_1000` never rose (ended 0.044). So matching
push-scale success mass, without Success-BC, does **not** retain Task 5.

### HER / dwell (progress-band claim stands; “short success eps” overstated)

Handle HER: succ stuck 2–4%, **prog = 0 always**, far 0.51 → 0.79.
Push HER: succ 0.04 → 0.48, prog ~0.28, succ|prog 0.13 → 0.77.

Push success episodes at 250k: ~11 far / 7 near / 25 hover / **52 progress**
/ 55 success steps per 150. Handle success episodes, when they exist:
**prog = 0**; geometric success steps often 40–120/150. Eval can be 20%
axis success with dwell `n_succ=0` (HER still requires hand-near).

Actor follow: `pi_argmax ≈ 0` on handle throughout. π is not the critic’s
discrete argmax at hover, and the press-biased action is not that argmax
either.

## Limitations

- One seed (6).
- Injected mass is 8 unique eval episodes cloned 326 times (diversity is
  low even though count is 20% of the buffer).
- Handle-measure late JSON was overwritten by the inject probe; use the
  wrapper stdout for the measure-cell late audit.
- Peak times (100k / 250k) are historical; this seed’s handle eval peak
  was 20% at 150k and 300k, not 100k.
