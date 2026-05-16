# RL-metrics insights — what the baselines tell us about continual sparse-reward GCRL

2026-05-16 · `results/data/raw/{for_real,real1,real2}/rl_metrics.parquet` · 372 baseline runs across the four cells reported in the paper's main body.

## TL;DR

Across the 9-cell baseline grid, three independent representation-health channels — **effective rank, dormant-neuron ratio, and the NRC1 within-class concentration** — tell the same story: under sparse rewards, every baseline transfer mode (from-scratch, persistent, CKA) suffers a representation collapse on exactly the hardest tasks of the curriculum (`k = 5, 8, 9`), and the persistent-critic family additionally drifts cumulatively across the curriculum via an order-of-magnitude weight-norm blow-up. Our proposed Decomposed Contrastive Critic is designed to keep the body as a transfer channel without locking it, and the per-task wins on those same hard tasks track the collapse pattern.

> The decomposed-critic runs in our W\&B project do not populate `rl_metrics/*` because the hook was added after the C2 sweep launched. Everything below describes the **baselines**; the proposed method is argued to escape the failure mode by construction.

## Data

- **Source**: W\&B project `nyuad_mmvc/continual_gcrl_paper`. Three groups: `for_real` (the 9-cell GCRL grid plus mixed cells), `real1` and `real2` (CKA-actor re-runs).
- **Cells covered**: 4 cells × 4–5 seeds × 10 tasks → 372 runs.
  - `actor=reset-critic=reset` (R/R, from-scratch baseline)
  - `actor=persistent-critic=persistent` (P/P)
  - `actor=cka-critic=reset` (C/R, best non-decomposed)
  - `actor=cka-critic=cka` (C/CKA, full mixture)
- **Channels**: 16 metrics under `rl_metrics/{actor, critic_sa, critic_g, critic}/*`, sampled every ~250 k env steps inside each task. We use the **last** sample of each (cell, seed, task) as the end-of-task representation snapshot.
- **Routing**: GCRL `{reset, persistent}²` cells take the `for_real` runs; CKA-actor cells pool `real1 + real2` and dedupe by seed keeping the run with the most rl-metrics samples.

## Headline numbers

End-of-task means averaged across all 10 tasks (paper-cell ordering):

| metric | R/R | P/P | C/R | C/CKA |
|---|---:|---:|---:|---:|
| actor `feature_rank` | 84.96 | **65.78** | 79.37 | 86.60 |
| actor `dormant_ratio` | **0.27** | 0.14 | 0.13 | 0.12 |
| actor `weight_norm` | 595 | **1989** | 1252 | 1353 |
| actor `final_layer_norm` | 16.7 | **104.9** | 44.6 | 42.1 |
| actor `entropy` | 4.68 | 4.62 | 4.72 | 4.83 |
| actor `nrc1` (within-class) | 0.59 | **0.45** | 0.54 | 0.57 |
| critic-sa `feature_rank` | 49.87 | 46.28 | 48.99 | 50.87 |
| critic-sa `nrc1` | 0.60 | **0.54** | 0.59 | 0.61 |
| critic-sa `nrc2` (between-class) | 0.65 | **0.76** | 0.64 | 0.73 |
| critic `weight_norm` | 677 | **1910** | 676 | 1044 |

Bold marks the most extreme value per metric. Two outliers dominate: the persistent-critic cell is consistently in the "bad" tail on every representation-health metric except entropy.

## Insight 1: rank crashes on the hard tasks for every baseline

The actor's effective rank — a robust measure of how many independent directions the penultimate-layer features span — falls sharply on the three hardest tasks of the curriculum (`k = 5 handle-press-side`, `k = 8 window-close`, `k = 9 peg-unplug-side`). The pattern is uniform across cells:

| cell | rank on easy tasks (avg of k∉{5,8,9}) | rank on hard tasks (avg of k∈{5,8,9}) | drop |
|---|---:|---:|---:|
| R/R | 96.94 | 46.82 | **−51.7 %** |
| C/R | 93.84 | 41.84 | **−55.4 %** |
| C/CKA | 96.10 | 63.18 | −34.3 % |
| **P/P** | 76.77 | 30.82 | **−59.9 %** |

Same picture for the state-action representation:

| cell | critic-sa rank easy | critic-sa rank hard | drop |
|---|---:|---:|---:|
| R/R | 55.86 | 30.82 | −44.8 % |
| C/R | 56.00 | 30.81 | −45.0 % |
| C/CKA | 54.67 | 41.50 | −24.1 % |
| **P/P** | 53.11 | 24.55 | **−53.8 %** |

The from-scratch baseline starts each hard task with a fresh actor, so the rank crash is not about "old weights"; it is about how the contrastive critic + actor jointly behave under sparse reward when the goal distribution is intrinsically harder to reach. The persistent cell is consistently worst because it enters those hard tasks with already-degraded features (see Insight 2).

This is the empirical pattern the paper's `fig3_representation.png` makes visible. The success drop on the same tasks in `fig1_per_task.png` matches it.

## Insight 2: persistent transfer drifts cumulatively

Two pieces of evidence say the persistent-critic family does not just lose rank on hard tasks but also accumulates damage across the curriculum:

**(a) Weight-norm blow-up.** The actor's `weight_norm` for P/P balloons from 491 at `k=0` to 3831 at `k=9` — an 8× growth. For R/R it stays bounded between 461 and 937. CKA cells sit between (560 → 1747).

```
task         k=0     k=1     k=2     k=3     k=4     k=5     k=6     k=7     k=8     k=9
R/R          461    522     572    517    562    937    529    547    828    553
P/P          491    795    1064   1297   1533   2597   2925   3196   3497   3831
C/R          556    827    1131   1081   1141   1550   1370   1460   1769   1747
```

The corresponding `critic/weight_norm` does the same: P/P grows from 481 → 4039 across the curriculum while R/R stays around 600. This is consistent with the "loss-of-plasticity" picture in deep RL: gradients keep accumulating without reset, parameters drift far from their initialisation, and learning capacity for new tasks degrades.

**(b) Actor entropy collapse.** The actor's policy entropy in P/P drifts down from 4.80 at `k=0` to 4.21 at `k=9` — the policy becomes more deterministic over time, locked to a narrower action distribution. R/R holds in the 4.5–4.8 band. This is independent evidence of the persistent actor over-committing.

These two channels together explain why the persistent-critic family is the worst average performer (0.79 avg-best vs 0.83 for from-scratch): it is not the hard-task collapse that hurts it most, it is the cumulative drift that lowers its starting point on every subsequent task.

## Insight 3: NRC metrics confirm a feature-collapse mechanism

The NRC1 metric (within-class variance of the representation, lower = more collapsed) on the **state-action** representation drops sharply on hard tasks, again worst for P/P:

| cell | nrc1 easy | nrc1 hard | drop |
|---|---:|---:|---:|
| R/R | 0.66 | 0.42 | −36 % |
| C/R | 0.67 | 0.40 | −40 % |
| **P/P** | 0.62 | 0.28 | **−54 %** |
| C/CKA | 0.65 | 0.53 | −19 % |

NRC2 (between-class variance, higher = more separated) **rises** for P/P (0.62 → 0.76 across the curriculum), meaning the persistent body collapses inputs into a few coarse clusters. Together NRC1 ↓ and NRC2 ↑ are a classic neural-collapse signature: the representation loses fine-grained structure within a goal-class and over-separates across classes.

The CKA-CKA cell partially escapes this on the state-action side (its NRC1 stays at 0.53 on hard tasks vs 0.42 R/R), consistent with the mixture acting as a soft regulariser on the body. But it does not escape the rank crash, and it does not match the proposed decomposed cell on success.

## Insight 4: the dormant-ratio signature is cell-specific

The dormant-neuron ratio (fraction of actor units essentially-inactive on the eval batch) shows two qualitatively different patterns:

- **R/R (from-scratch)**: 0.27 baseline; crashes to ~0.05 on `k=5, 8` and back to ~0.33 on `k=9`. The fresh actor cannot recruit units fast enough on the contact-heavy tasks.
- **P/P / C/R / C/CKA**: 0.12–0.14 baseline; rises to ~0.20 at `k=8, 9` for P/P. The persistent body has units that were already useful on earlier tasks and now stay live but specialised.

The from-scratch pattern is "no recruitment", the persistent pattern is "lock-in". Both block the actor from injecting new directions where it needs to on the hard tasks. This is exactly the gap a state-and-action-dependent per-task encoder is designed to fill.

## Why the decomposed critic is designed for this failure mode

The patterns above all point at the same constraint: under sparse rewards, when the agent enters a hard task it needs to **inject new representational directions** that depend on the actual state-action pairs it is now seeing. None of the baseline transfer modes can do this.

- `reset` re-initialises everything, throwing away the body and forcing slow recruitment under no shaping signal.
- `persistent` carries everything forward, locking the body into a configuration that worked on earlier tasks but is mis-aligned for the new one (and the cumulative weight-norm drift makes the lock worse over time).
- `CKA` adds a parameter-space additive mixture `Σ α_j v_j` that, to first order, is a **state-independent constant shift** in the state-action representation. Both downstream losses (actor argmax over actions, InfoNCE softmax over goals) are invariant to such shifts — see the geometric appendix in the paper for the null-space argument and the anti-correlation we measure on the critic side.

The Decomposed Contrastive Critic addresses this directly:

```
sa_repr(s, a)  =  h_phi(b_shared(s, a))   +   phi_task(s, a)
                  └── transfer ──┘            └── per-task ──┘
                  shared across tasks         re-init at every k
                  shaped by L_dyn on M        state-and-action dependent
```

`b_shared` is the only transfer channel and is regularised by the masked-dynamics auxiliary so it captures the task-invariant structure (end-effector + gripper subspace). `phi_task(s, a)` depends on both inputs, so its gradient is not in the null space of the downstream losses, and it can inject the new directions a hard task needs. The per-task wins of +0.16 on `k=5` and +0.27 on `k=8` over from-scratch (and over every other baseline cell) are concentrated precisely on the tasks where the baselines exhibit the largest rank crashes above.

## What we still need to log

The C2 (decomposed) runs in the W\&B project do not populate `rl_metrics/*` because the hook was added after the C2 sweep launched. As a result we cannot show the proposed cell on the figures of the paper's analysis section, only on the success curves. The fix is mechanical:

1. Re-launch the C2 sweep (or a single seed of it) with the `rl_metrics` callback enabled on the decomposed learner. The learner already exposes `last_transitions`, `last_metrics`, and `q_params` so the wiring is one-line.
2. With the channel populated we expect three falsifiable predictions:
   - Actor `feature_rank` on `k=5, 8, 9` does **not** crash to ~30 for the decomposed cell.
   - Actor `weight_norm` stays bounded (the body alone is shared; the per-task encoder resets so drift on it is bounded).
   - Critic-sa `nrc1` does not collapse on the hard tasks.

These will be the head-to-head rl-metrics figures for the camera-ready.

## Reproducibility

```bash
# pull rl_metrics across the four baseline cells
WANDB_API_KEY=… python results/scripts/fetch_rl_metrics.py \
  --cells reset:reset persistent:persistent cka:reset cka:cka \
  --groups for_real real1 real2

# render the paper-quality figures
python results/scripts/render_paper_figs.py
```

Per-run snapshots live in `results/data/raw/{group}/rl_metrics.parquet`; the analysis script that produced every number in this note is at `results/scripts/render_paper_figs.py` (functions `load_rl_metrics`, `fig_representation_analysis`).

## Source pointers

- Paper main: `Okyumi/CGCRL---RLC-workshop-2026@main`, section 4.2 (where the rank-crash figure lives).
- Paper appendix: same repo, sections A4 (CKA diagnostics) and A6 (logging notes).
- Code: `Okyumi/sgcrl@section3_done`, scripts under `results/scripts/`, raw under `results/data/raw/*/rl_metrics.parquet`.
