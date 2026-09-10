# Paper 9-baseline × 10-seed rerun after the native-success wrapper repair

Date: 2026-09-09  
Status: implemented and launched on Torch HPC.

## Motivation

The DCC paper compares a 3×3 contrastive transfer grid over actor and
critic modes `{reset, persistent, CKA}`. Earlier table numbers used
seeds 5/6/7 under the legacy Sawyer success wrapper. After the
`native_info` repair, those checkpoints are not reusable, so the nine
baseline cells are rerun from scratch on ten seeds.

Representation figures in the paper need feature rank, entropy, weight
norms, and neural-collapse metrics throughout each task. The previous
native-success promotion matrix disabled those logs to save wall time.
This rerun keeps them on, while packing four learners per L40S so the
GPU is not left idle during CPU MuJoCo rollouts.

## Mathematical objective

Each cell reuses single-task contrastive GCRL and differs only at task
boundaries. For actor parameters `θ` and critic parameters `φ`:

- Reset (R): reinitialise the network, optimiser, and replay at every
  task.
- Persistent (P): carry parameters, targets, and optimiser state;
  discard only the per-task replay.
- CKA (C): `θ_k = θ_base + Σ_j α_{k,j} v_j + v_k` with pool capacity
  `K_max = 5`, matching Appendix E.3.

Logged representation metrics follow Appendix F: effective rank at
`τ=0.99`, `∥θ∥_2`, feature entropy, and NRC1/NRC2.

## Code and configuration changes

- `run_continual_contrastive.py`: `--rl_metrics_occasional_multiplier`
  (default 5). Paper runs set it to 2, so rank/NRC/dormancy log every
  200k steps (~40 samples/task at 8M).
- `experiment_configs_paper_9baseline_10seed.py`: 9 cells × seeds
  5..14 = 90 runs. `sawyer_success_mode=native_info`,
  `network_width=1024`, `k_max=5`, `eval_every=100000`,
  `log_rl_metrics=true`. Heavy simulator diagnostics are off.
  Mixture-norm / pool-cosine logs are enabled only on CKA sides.
- `DRAFT_paper_9baseline_10seed.sh`: four processes per L40S
  (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.22`) on `l40s_public`, 48h wall
  time, and an `afterany` continuation chain (max 8 hops). Training
  still auto-resumes from the latest `task_{k}.pkl`.
- `scripts/paper_9baseline_status.py`: prints per-run completion and
  unfinished array IDs.

Checkpoint root (isolated from legacy and native-success promotion
runs):

```
/scratch/yd2247/sgcrl/logs/paper_9baseline_checkpoints/10seed
```

W&B project `continual_gcrl_paper`, groups
`PAPER-9BASELINE-10SEED-{actor}-{critic}`.

## Launch command

```bash
sbatch DRAFT_paper_9baseline_10seed.sh
```

If a chain breaks, resubmit only unfinished array tasks:

```bash
ARRAY=$(python scripts/paper_9baseline_status.py --incomplete-array-ids)
sbatch --array="$ARRAY" DRAFT_paper_9baseline_10seed.sh
```

## Logged metrics

Per evaluator step (100k env steps): success, return, actor/critic
weight norms, feature entropy, Gini.

Every two evaluator steps (200k): actor and critic effective rank,
NRC1, NRC2, dormant ratio.

CKA cells additionally log `cka/*_mixture_norm` and end-of-task pool
cosine matrices.

Runtime profiling (`actor_seconds`, `learner_seconds`,
`evaluation_seconds`, `rl_metrics_seconds`) is on so GPU idle time
can be checked without extra simulator work.

## Validation

```bash
python tests/test_paper_9baseline_10seed.py
```

The tests cover the 9×10 matrix, native-info / width-1024 / metrics
flags, checkpoint identity, completion detection, launcher packing,
and the new occasional-multiplier flag.

## Known limitations

- Mid-task replay is still not serialised. A 48h kill restarts the
  *current* task from the previous task-boundary checkpoint, not from
  the last env step. The continuation chain exists to bound that loss
  to one incomplete task per hop.
- Batch size remains the historical 256 / 64-update Acme setting, not
  the paper appendix's 1024. Width 1024 and the residual backbone
  match Appendix C.
- This launch is the nine contrastive baselines only. Sparse SAC R/R,
  SAC P/P, and DCC are not in this array.
- Four packed learners share 16 CPUs. If a node OOMs, resubmit with
  `TASKS_PER_GPU=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.30`.
