# Paper sparse SAC R/R + P/P × 10-seed rerun

Date: 2026-09-09  
Status: implemented and launched on Torch HPC.

## Motivation

The DCC paper's non-contrastive baseline family is sparse goal-conditioned
SAC with HER (Appendix E.1). After the native-success wrapper repair, those
cells have to be rerun from scratch on the same ten seeds as the contrastive
9-cell grid.

## Variant used in the paper

Appendix E.1 and Table 4 fix the SAC *reward variant*, not a CKA mixture:

$$
r = \mathbf{1}\!\left[\|m(s')-g\|_2 < \tau_{\mathrm{reach}}\right] - 1
\in \{-1,0\},
\qquad
\tau_{\mathrm{reach}}=0.05,
\qquad
\gamma_{\mathrm{HER}}=(1-\mathbf{1}[r=0])(1-d)\gamma.
$$

This is `--step_penalty_reward` (the `{0,+1}` shape is an ablation only).
The two continual cells are:

- SAC (R/R): actor and critic reset at every task boundary
- SAC (P/P): actor and critic carried forward

No CKA cell. Target entropy is $-2.0$, Polyak $\rho=0.995$, width 1024,
depth 4, residual LayerNorm+Swish, `use_task_id=false`. Seeds are 5..14
to match the contrastive rerun (the paper table used 3–5 seeds; the
legacy SAC pull used seeds 1/2/3).

## Code and configuration changes

- `sac/flags.py`: `--sawyer_success_mode`, `--goal_conditioning_mode`,
  `--rl_metrics_occasional_multiplier`.
- `sac/training.py`: passes `native_info` into every environment factory
  and logs rank/NRC on the paper cadence.
- `sac/checkpointing.py`: `_success_{mode}` suffix when the success mode
  is not the historical `corrected` default, so this rerun cannot resume
  a legacy wrapper checkpoint.
- `experiment_configs_paper_sparse_sac_10seed.py`: 2 cells × 10 seeds.
- `DRAFT_paper_sparse_sac_10seed.sh`: four processes per L40S, 48h
  `afterany` continuation chain.

W&B project `continual_gcrl_paper`, groups
`PAPER-SPARSE-SAC-10SEED-{reset-reset|persistent-persistent}`.

Checkpoint root:

```
/scratch/yd2247/sgcrl/logs/paper_sparse_sac_checkpoints/10seed
```

## Launch command

```bash
sbatch DRAFT_paper_sparse_sac_10seed.sh
```

Resume unfinished array tasks:

```bash
ARRAY=$(python scripts/paper_sparse_sac_status.py --incomplete-array-ids)
sbatch --array="$ARRAY" DRAFT_paper_sparse_sac_10seed.sh
```

## Logged metrics

Same evaluator cadence as the contrastive paper cells (100k env steps),
with actor/Q feature rank, NRC, entropy, and weight norms every 200k.
`learner/her_success_rate` and `learner/reward_mean` confirm the
step-penalty HER signal.

## Validation

```bash
python tests/test_paper_sparse_sac_10seed.py
python -m pytest tests/test_sac_flags.py tests/test_sac_checkpointing.py -q
```

## Known limitations

- Mid-task replay is not serialized; a 48h kill restarts the current task
  from the previous task-boundary checkpoint.
- Batch size remains the historical Acme 256 / 64-update setting.
- Only R/R and P/P are launched, matching Appendix E.1.
