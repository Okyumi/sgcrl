# Jubail 10-seed DCC ablation vs Success-BC (no dynamics head)

Date: 2026-09-12  
Status: first submit (`17936020`) failed in 1–2s; relaunched after fixing
startup.

## Motivation

Torch is at GPU QOS, so the remaining paper cells move to Jubail. The
paper comparison requested here is a two-cell DCC ablation on the full
ten-task Continual World V2 Sawyer sequence, seeds 5–14:

1. **Ablation.** Decomposed critic, reset actor, **no dynamics head**
   and **no Success-BC**.
2. **Proposed.** The same decomposed critic with the augmented
   behavioural-cloning term (`success_bc_weight=0.1`). Still **no
   dynamics head**.

Both cells keep the paper representation logs: evaluation success,
per-task best/final success, curriculum averages, actor/critic feature
rank and entropy, neural collapse, dormant-neuron ratios, weight norms,
and the usual training metadata.

A 10-task × 8M-step curriculum exceeds Jubail’s 48h wall clock, so each
array task chains an `afterany` continuation. Training auto-resumes from
the latest `task_{k}.pkl`.

## Mathematical objective

The critic remains the DCC decomposition

\[
z_{s,a}^{(k)}=\phi_{\mathrm{shared}}(s,a)+\phi_{\mathrm{task}}^{(k)}(s,a),\qquad
f^{(k)}(s,a,g)=\mathrm{sim}(z_{s,a}^{(k)},\psi(g)).
\]

The masked-dynamics auxiliary is off for both cells (`μ=0`). When
`μ=0`, the learner JIT skips the `h_dyn` forward, backward, and Adam
step, so the dynamics head is not trained.

The proposed actor is Eq. 7 in the paper:

\[
J_{\mathrm{actor}}(\theta)=J_{\mathrm{CRL}}(\theta)
+\lambda_{\mathrm{succ}}
\mathbb{E}_{\tau\sim\mathcal{D}_{\mathrm{succ}},(s_t,a_t)\sim\tau}
[\log\pi_\theta(a_t\mid s_t,g)],
\]

with `λ_succ=0.1`, `episode_sparse_reward` labels, ring-buffer capacity
4096, and BC batch size 64. The ablation uses `λ_succ=0`.

## Code / config

- `contrastive/continual_learning_decomposed.py`: skip the dynamics step
  when `dyn_aux_weight=0`.
- `experiment_configs_paper_dcc_success_bc_jubail.py`: 2 cells × seeds
  5..14 = 20 runs. `dyn_aux_weight=0`, `native_info`, width 1024,
  `eval_every=100000`, `log_rl_metrics=true`, occasional multiplier 2.
  Heavy simulator diagnostics are off.
- `DRAFT_paper_dcc_success_bc_jubail.sh`: two processes per A100
  (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`) on `partition=nvidia`, 48h
  wall time, `afterany` continuation chain (max 8 hops). Mail is
  `FAIL` only so chain hops do not spam END / GPU-idle notices.
- `scripts/paper_dcc_success_bc_jubail_status.py`
- `tests/test_paper_dcc_success_bc_jubail.py`

Shared with the 9-baseline paper rerun: `sawyer_success_mode=native_info`,
`post_task_eval_scope=current`, `k_max=5`, residual 1024×4.

Checkpoint root:

```
/scratch/yd2247/sgcrl/logs/paper_dcc_success_bc_jubail_checkpoints/10seed
```

W&B project `continual_gcrl_paper`, groups
`PAPER-DCC-NODYN-10SEED-ablation` and
`PAPER-DCC-NODYN-10SEED-success-bc`.

## First-submit failure (2026-09-12)

Array `17936020_[0-9]` all failed in 1–2s on `cn003` with exit `1:0` and
**no stdout/stderr files**. Two startup bugs:

1. `#SBATCH --output` pointed at
   `logs/paper_dcc_success_bc_jubail/`, which did not exist yet. SLURM
   cannot open the log file if that directory is missing, so the script
   never ran.
2. The wrapper called `python` before `conda activate`, which would have
   been the next immediate failure on a GPU node (`python: command not
   found`).

Fix: create the log/checkpoint directories before `sbatch`, and load the
Jubail conda env before any Python. Tests run only on chain hop 0.

## Launch command

```bash
sbatch DRAFT_paper_dcc_success_bc_jubail.sh
```

Resume unfinished array tasks:

```bash
ARRAY=$(python scripts/paper_dcc_success_bc_jubail_status.py --incomplete-array-ids)
sbatch --array="$ARRAY" DRAFT_paper_dcc_success_bc_jubail.sh
```

## Logged metrics

Per evaluator step (100k env steps): success, return, actor/critic
weight norms, feature entropy, Gini.

Every two evaluator steps (200k): actor and critic effective rank,
NRC1, NRC2, dormant ratio.

Success-BC additionally logs `retention/buffer_size`, `retention/bc_loss`,
`retention/bc_active`, and `retention/source_success_fraction`.

Runtime profiling (`actor_seconds`, `learner_seconds`,
`evaluation_seconds`, `rl_metrics_seconds`) is on.

## Validation

```bash
python tests/test_paper_dcc_success_bc_jubail.py
python experiment_configs_paper_dcc_success_bc_jubail.py --list
```

## Known limitations

- Mid-task replay is still not serialised. A 48h kill restarts the
  *current* task from the previous task-boundary checkpoint, not from
  the last env step. The continuation chain bounds that loss to one
  incomplete task per hop.
- The unused `h_dyn` parameters remain in the Haiku pytree so
  checkpoint layout stays compatible with other DCC cells; they receive
  no forward or gradient when `μ=0`.
- Two packed learners share 12 CPUs on mixed 40GB/80GB A100s. If a 40GB
  node OOMs, resubmit with `TASKS_PER_GPU=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`.
