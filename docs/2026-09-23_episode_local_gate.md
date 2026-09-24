# Whole-episode Success-BC + local-σ_s gate

Date: 2026-09-23  
Status: resubmitted as array **18131472** after 18131426 failed
immediately (log dir did not exist when Slurm opened stdout).

## Motivation

Ungated whole-episode Success-BC retains latch Tasks 5/8 and can stall
4/7. Gating current-sparse (\(r_t=1\) only) did not retain 5/8, because
that ring is hold/linger, not the success trajectory. The unified
recipe is the buffer that already works on 5/8, with the existing
local critic gate left as-is (not tightened).

## Method

If a replay episode has any sparse bit \(r=1\), every transition of
that episode enters \(\mathcal{D}_{\mathrm{succ}}\)
(`episode_sparse_reward`). λ=0.1 NLL under \(g_{\mathrm{task}}\).
The local gate is unchanged: at the sampled \((s,g)\),

\[
w=\frac{\sigma_s^{\mathrm{loc}}}{\sigma_s^{\mathrm{loc}}+\sigma_a}.
\]

Question for 4/7: does this still stay near plain DCC, or does
whole-episode cloning lock the first lucky path even with local \(w\)?

## Launch

```bash
sbatch DRAFT_jubail_episode_local_gate.sh
```

W&B `nyuad_mmvc/continual_gcrl_paper` /
`PAPER-DCC-EPISODE-LOCALGATE-4758`. Seed 6. Array 0–3: T4 8M, T7 8M,
T5 1M, T8 1M. Paper stack. T5/T8 use the same flags (one algorithm).

## Validation

```bash
python tests/test_episode_local_gate.py
```

## Limitations

Seed 6 only. T4/T7 resume paper Success-BC, not plain DCC.
The local-gate current-sparse jobs are a different buffer.
