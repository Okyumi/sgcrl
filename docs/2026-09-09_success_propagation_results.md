# Results: Task5 vs push success-propagation (D1–D4)

Date: 2026-09-09
Status: analyzed (SLURM `17188038`, W&B `TASK58-SUCCESS-PROP-1M`)

## Coverage note

- 11/12 cells finished 1M training. `handle_measure` seed5 (`17188038_0`)
  auto-resumed an incomplete prior ckpt and exited with “all tasks completed”
  (no new metrics). Conclusions use handle seeds 6/7 (+ inject 5/6/7).

## Headline numbers (late / post-discovery)

| Variant | peak eval | late eval | HER succ late | HER succ\|prog late | Δ HER after first ≥20% |
|---|---:|---:|---:|---:|---|
| handle_measure (s6/7) | 30% / 20% | ~13% / ~2% | 0.02 / 0.05 | same | **~0** |
| push_measure (s5–7) | 80–100% | 48–100% | 0.13–0.63 | **0.59–0.83** | **+0.4 to +0.8** |
| handle_inject (s5/6) | 20–30% | ~5–7% | 0.01–0.03 | same | **≤0** (no fill) |
| push_inject (s5–7) | 100% | ~70% | 0.40–0.48 | **0.75–0.79** | **+0.64 to +0.72** |

Handle HER `object_progress` stays **0.00** for the entire run.

## Conjecture scorecard

| Conjecture | Verdict | Evidence |
|---|---|---|
| Critic confuses hover≈success under \(g_{\text{task}}\) | **Falsified** | When both buckets exist, `gap_s-h@task` mean ~+28 to +54; A>C in 18/19 handle measure probes |
| Under \(g_{\text{hover}}\), hover can outrank success | **Often true** | `gap_s-h@hovg` often negative (e.g. peak s6: −34) |
| Critic fails to mark success episodes (no “trace”) | **Falsified as primary cause** | When probe has success, trace gap mean ~+76 to +84 on handle (success ≫ fail) |
| Actor fails to keep generating success → HER stays empty | **Confirmed** | After discovery, HER succ flat ~2–5%; later probes often `n_s=0` |
| Lucky-spike then hover *refills* HER | **Falsified** | HER success never spikes on handle |
| Scarcity alone: inject success into replay → propagation | **Falsified / insufficient** | Inject 150–300 transitions; HER stays ~0.01–0.07; next eval often collapses (e.g. 30%→0%) |
| Push closes SGCRL loop via progress ladder | **Confirmed** | Push HER `succ\|prog` climbs 0.02→0.8; handle `prog` always 0 |

## D3 actor-follow

π is rarely critic-argmax at hover on **both** tasks (`pi_is_argmax` ~0–8%).
So “π ≠ argmax” alone does **not** explain Task5 vs push. The
discriminating fact is whether the actor **keeps visiting** success/progress
states (push yes, Task5 no).

## Causal answer (Task5)

HER only receives what the **actor rolls out**. The critic’s preference for
success under \(g_{\text{task}}\) does not write into replay.

On Task5 after discovery:

1. Critic usually ranks success states/episodes above hover/fail under \(g_{\text{task}}\).
2. Actor briefly discovers press (eval 20–30%).
3. Actor does **not** keep producing successful trajectories (`n_s` often returns to 0).
4. HER success mass therefore never rises (~2–5%).
5. Actor updates stay dominated by hover/far HER futures → non-press policy locks in.
6. One-shot success injection does not restart this generation loop.

Push differs because continuous object progress creates a dense intermediate
HER ladder (`object_progress`), so after first success the actor keeps
emitting progress/success and HER `succ|prog` self-amplifies.

This is why Success-BC retains Task5 (forces successful **actions** into the
actor update) while critic-score maximisation / inject-without-BC do not.
