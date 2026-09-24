# Local-σ_s critic-gated Success-BC

Date: 2026-09-23  
Status: resubmitted as SLURM `18122639` after a Python scoping crash
(`obs_dim` rebound inside `actor_loss_fn`). First array `18122324`
died in ~90s.

## Motivation

Batch \(\sigma_s\) (std of scores across the BC minibatch) is mostly
“how close is this \(s\) to \(g\)”, so the gate stays ~0.7–0.9 even
when the critic ranks actions. The intended rule is local: at one
observed success state, compare action sensitivity to a small jitter
of **that same** \(s\).

## Method

\(\mathcal{D}_{\mathrm{succ}}\): current sparse bit \(r_t=1\).
\(\lambda=0.1\), 4096 ring, NLL target \(a_{\mathrm{succ}}\) only.

At the sampled \((s,g)\), query the existing DCC score
\(\varphi(s,a)^\top\psi(g)\):

- \(\sigma_a\): 8 random actions, fixed \((s,g)\).
- \(\sigma_s^{\mathrm{loc}}\): 8 jitters \(s + 0.1\,\hat\sigma_{\mathrm{batch}}\,z\),
  same \(g\), same \(a_\pi\). \(z\sim\mathcal{N}(0,I)\). \(\hat\sigma_{\mathrm{batch}}\)
  is per-coordinate std in the current BC batch (z-score units).
- \(w=\sigma_s^{\mathrm{loc}}/(\sigma_s^{\mathrm{loc}}+\sigma_a)\), stop-gradient.

No hover radius, task list, or stored action as a ranking label.

Hypothesis: latch Tasks 5/8 stay action-flat (\(w\) high, BC on);
place/pull 4/7 are action-informative locally (\(w\) lower, DCC leads).

## Launch

```bash
sbatch DRAFT_jubail_local_critic_gate.sh
```

W&B `nyuad_mmvc/continual_gcrl_paper` /
`PAPER-DCC-LOCAL-CRITICGATE-4758`. Seed 6. Array 0–3: T4 8M, T7 8M,
T5 1M, T8 1M. Paper stack. The batch-\(\sigma_s\) jobs are left running
as an ablation.

## Validation

```bash
python tests/test_local_critic_gate.py
```

## Paper

This is the candidate unified regularizer only if T5/T8 retain and
T4/T7 stay near plain DCC. Until those four cells return, it is a
hypothesis, not a result.

## Limitations

Seed 6 only. T4/T7 resume paper Success-BC, not plain DCC. Linger
still enters \(\mathcal{D}_{\mathrm{succ}}\); the gate only weights NLL.
T8 may fail on H200 `libEGL`.
