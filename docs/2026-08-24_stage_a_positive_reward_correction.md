# Stage-A2 Positive-Reward Correction

Date: 2026-08-24  
Status: implemented for Torch HPC; rerun not submitted

## Motivation

All four original Stage-A runs finished and satisfied the presence-only
automated checks. W&B nevertheless showed that
`benchmark_success_fraction` was exactly $1.0$ at every event, even though
evaluation success was between $0$ and $0.2$. The rank collector had used
`zero_reward`, which interprets reward $0$ as success under a $-1/0$
convention.

That convention belongs to the learner's internally shaped HER target. The
actual Sawyer wrappers used by `SuccessObserver` emit reward $0$ on failure
and $1$ on success. Consequently, the original Stage-A success bonus was a
constant and its oracle-success metrics were invalid. The progress rankings
remain useful because adding a constant did not change their ordering.

## Corrected definition

For a counterfactual rollout of horizon $H$, benchmark success is now

$$
z_H=\max_{1\leq t\leq H}\mathbf 1[r_t>0].
$$

The independently computed mechanism proxy is

$$
\hat z_H=\max_{1\leq t\leq H}
\mathbf 1[\lVert x_{\mathrm{mechanism},t}-g\rVert_2\leq\epsilon],
$$

with $\epsilon=0.02$ for handle-press-side (Task 5) and $\epsilon=0.05$
for window-close (Task 8), matching `env_utils.py`.

## Exact changes

- Every Stage A--D configuration now uses
  `counterfactual_rank_success_mode=positive_reward` and
  `action_landscape_success_mode=positive_reward`.
- Task 5 uses success threshold $0.02$; Task 8 uses $0.05$.
- The corrected Stage-A2 groups are
  `CFR-STAGE-A2-positive-reward-task5` and
  `CFR-STAGE-A2-positive-reward-task8`; the old runs cannot be mixed in.
- Stage A2 is reduced from 100k to 30k steps, with 10k diagnostic intervals.
  It validates instrumentation and is not a performance comparison.
- The Stage-A evaluator now requires per-seed signal availability and
  benchmark/proxy agreement of at least $0.99$, with false-positive and
  false-negative rates at most $0.01$.
- The Torch launcher uses separate `A2_positive_reward` log and checkpoint
  directories and verifies the emitted success-mode and W&B-group settings
  before entering the main runner.
- The MuJoCo preflight checks that both task wrappers emit raw 0/1 rewards and
  that a zero-action reset is interpreted as failure by `positive_reward`.

## W&B metrics and promotion gate

Stage A2 must record, for both seeds on both tasks:

- `learner/counterfactual_rank/benchmark_success_available_fraction`;
- `learner/counterfactual_rank/success_predicate_agreement`;
- `learner/counterfactual_rank/proxy_false_positive_fraction`;
- `learner/counterfactual_rank/proxy_false_negative_fraction`;
- `learner/counterfactual_rank/heldout_post/fixed_state_score_std`;
- `learner/action_landscape/action_repeat`;
- `learner/action_landscape/aligned_score_vs_progress_spearman`.

Only measurement validity is a Stage-A2 promotion condition. Causal ranking
quality is evaluated in Stage C.

## Torch launch

```bash
cd /scratch/yd2247/sgcrl
git pull --ff-only origin section3_done

COUNTERFACTUAL_STAGE=A sbatch DRAFT_counterfactual_stages.sh
```

After all four runs finish:

```bash
python scripts/evaluate_counterfactual_stages.py --stage A
```

Do not submit Stage B until this command passes for both tasks.

## Validation and limitations

Dependency-light tests cover both reward conventions, task-specific
thresholds, isolated W&B groups, config expansion, and shell-to-runner flag
propagation. Python and shell syntax are checked locally. Exact MuJoCo restore
and reward behavior require the Torch `contrastive_rl` environment and run as
launcher preflights.

Agreement with the mechanism proxy validates the current Sawyer wrappers; it
is not a general assumption for benchmarks whose success predicate includes
additional contact, force, or temporal conditions.
