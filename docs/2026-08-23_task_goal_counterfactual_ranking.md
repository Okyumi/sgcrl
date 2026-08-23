# Task-Goal Same-State Counterfactual Ranking for DCC

Date: 2026-08-23  
Status: implemented as a four-cell, 1M-step falsification experiment

## Why this experiment is next

Stage 2 rejected the observational raw-outcome fix. Its replay labels were almost always positive, its score was retained when actions were shuffled, and its fixed-state action ordering remained unrelated to measured rollout progress. Each replay state appeared with only one behavior action, so a head could predict state or trajectory difficulty without learning the causal effect of choosing a different action.

This experiment removes that confound directly. It restores the same simulator state, tries multiple candidate actions, evaluates every candidate against the environment's original task goal under a common continuation policy, and trains only on within-state preferences.

## Exact insertion point in DCC

The original DCC representation and losses remain unchanged:

$$
q_{\mathrm{DCC}}(s,a,g)
=
\left(h_{\phi}(b_{\mathrm{shared}}(s,a))
+\phi_{\mathrm{task}}(s,a)\right)^\top\psi(g).
$$

The new component uses the existing task-local action-effect MLP slot $u_{\eta}(s,g,a)$, configured to return one scalar. It is inserted at the actor-score stage, after DCC's representation update and before the actor loss. In this falsification cell the actor uses only the counterfactual head:

$$
J_{\pi}
=
\mathbb{E}_{s,g_{\mathrm{task}}}
\left[u_{\eta}(s,g_{\mathrm{task}},a_{\pi})
+\alpha\mathcal{H}(\pi(\cdot\mid s,g_{\mathrm{task}}))\right].
$$

The original DCC score is deliberately excluded from the actor objective. This prevents the diagnosed miscalibrated DCC action landscape from masking whether correct counterfactual action ordering is sufficient.

## Counterfactual labels

For anchor state $s_i$ and original task goal $g_i$, the collector constructs candidate actions from four families: policy, local policy perturbations, nearest replay actions, and uniform actions. Each candidate $a_{ij}$ is held for $C=5$ steps, followed by the same deterministic policy continuation, for a total horizon $H=100$.

Let $d_m(s,g)$ be the mechanism-coordinate distance. Candidate progress and success are

$$
p_{ij}
=
\max\left(0,
d_m(s_i,g_i)-\min_{1\leq t\leq H}d_m(s_{ij,t},g_i)\right),
$$

$$
z_{ij}
=
\mathbb{1}\left[
\min_{1\leq t\leq H}d_m(s_{ij,t},g_i)\leq\epsilon
\right],
$$

and the ranking outcome is

$$
y_{ij}=p_{ij}+\lambda_s z_{ij}.
$$

The implementation does not interpret a positive Meta-World dense reward as success. It uses only the task-goal mechanism threshold, avoiding the previous all-positive label failure.

For candidates $j$ and $k$ from the same anchor, an informative preference exists when $|y_{ij}-y_{ik}|\geq\delta$. The head is optimized with

$$
\mathcal{L}_{\mathrm{rank}}
=
\frac{1}{|\mathcal{P}|}
\sum_{(i,j,k)\in\mathcal{P}}
\log\left(
1+\exp\left[
-\operatorname{sign}(y_{ij}-y_{ik})
\frac{u_{\eta}(s_i,g_i,a_{ij})-u_{\eta}(s_i,g_i,a_{ik})}{\tau}
\right]
\right)
+\lambda_2\|\eta\|_2^2.
$$

Because every comparison holds $(s_i,g_i)$ fixed, state difficulty and goal identity cannot explain the label. The head is gated off in the actor until at least one informative rank update occurs.

## Implementation map

| File | Change |
|---|---|
| `contrastive/counterfactual_ranking.py` | Exact-state collector, four candidate families, original-task-goal outcomes, grouped buffer, score summaries, self-test |
| `contrastive/continual_learning_decomposed.py` | Scalar rank-head mode, pairwise optimizer, inactive-until-trained gate, task-goal actor input, score API, checkpoint-compatible state |
| `run_continual_contrastive.py` | Flags, original-goal replay extra, isolated environment, periodic training events, immediate W&B logging, cleanup |
| `contrastive/continual_config.py` | Collection, buffer, optimizer, and loss settings |
| `experiment_configs_counterfactual_ranking.py` | Tasks 5/8, seeds 5/6, 1M steps; four cells |
| `DRAFT.sh` | Environment-variable and CLI forwarding plus preflight checks |
| `DRAFT_counterfactual_ranking.sh` | Torch Slurm array launcher |
| `scripts/evaluate_counterfactual_ranking.py` | Two-seed promotion gates |
| `scripts/evaluate_outcome_falsification.py` | Corrected historical W&B metric keys |

All prior modes retain their original defaults. The extra simulator and counterfactual work are created only when `counterfactual_rank_interval_steps > 0`.

## Experiment grid

| Array index | Task | Seed | Steps | W&B group |
|---:|---|---:|---:|---|
| 0 | Sawyer handle press side (Task 5) | 5 | 1M | `CFRDCC-taskgoal-H100-chunk5-task5` |
| 1 | Sawyer handle press side (Task 5) | 6 | 1M | `CFRDCC-taskgoal-H100-chunk5-task5` |
| 2 | Sawyer window close (Task 8) | 5 | 1M | `CFRDCC-taskgoal-H100-chunk5-task8` |
| 3 | Sawyer window close (Task 8) | 6 | 1M | `CFRDCC-taskgoal-H100-chunk5-task8` |

Each collection event uses four anchors and sixteen candidate actions per anchor. It occurs after the first learner step and then every 50,000 environment steps. The task-local buffer holds 128 complete anchor groups; each event performs 25 rank updates with 16 sampled groups.

## W&B metric keys

The old gate script omitted the logger namespace and used an incomplete causal metric name. The corrected historical keys are:

| Quantity | Correct key |
|---|---|
| Same-state causal rollout Spearman | `learner/action_landscape/score_vs_rollout_mechanism_progress_spearman` |
| Fixed-state outcome-head score std | `learner/outcome/fixed_state_action_std` |
| Outcome-head shuffled-action retention | `learner/outcome/action_shuffle_retention` |

The new experiment logs immediately at each collection event:

- `learner/counterfactual_rank/informative_anchor_fraction`
- `learner/counterfactual_rank/near_interaction_fraction`
- `learner/counterfactual_rank/candidate_progress_std`
- `learner/counterfactual_rank/task_success_fraction`
- `learner/counterfactual_rank/pair_count`
- `learner/counterfactual_rank/pairwise_accuracy`
- `learner/counterfactual_rank/score_vs_task_progress_spearman`
- `learner/counterfactual_rank/top_action_regret`
- `learner/counterfactual_rank/fixed_state_score_std`
- `learner/counterfactual_rank/updates_total`

The ordinary causal probe remains enabled at 250,000-step intervals, so the learned head is also judged on separately collected same-state rollouts.

## Pre-registered interpretation

| Observation after 1M steps | Conclusion | Next action |
|---|---|---|
| Informative-anchor fraction below 0.25 or progress std below 0.002 | Candidate interventions do not generate useful outcome variation | Improve interaction anchors and test short action sequences before changing the learner |
| Labels vary, but pairwise accuracy below 0.65 or rank-batch correlation below 0.30 | Optimization or capacity is inadequate | Tune rank updates, temperature, and head capacity only |
| Rank batch fits, but independent causal correlation stays below 0.30 | The head memorizes collected anchors or one initial action is insufficient | Rank action chunks or options and use receding-horizon selection |
| Both correlations pass, but success AUC and retention fail | A correct local ranker is insufficient for closed-loop long-horizon control | Add policy-support regularization or execute ranked chunks with MPC-style replanning |
| Both tasks pass all gates | Counterfactual ordering is a credible control signal | Proceed to longer runs and then continual retention |

Promotion requires, on both tasks and across both finished seeds:

- success AUC gain of at least 0.05 over matched CRTR;
- final-20%-to-peak retention of at least 0.80;
- independent causal rollout Spearman at least 0.30;
- collected rank-batch Spearman at least 0.30;
- pairwise accuracy at least 0.65;
- informative-anchor fraction at least 0.25;
- candidate progress standard deviation at least 0.002;
- fixed-state score standard deviation at least 0.001.

## Launch and evaluation

On Torch:

```bash
sbatch DRAFT_counterfactual_ranking.sh
```

After all four jobs finish:

```bash
python scripts/evaluate_counterfactual_ranking.py
```

The evaluator returns a nonzero exit status unless every gate passes on both tasks.

## Validation completed

- All modified Python files compile.
- `DRAFT.sh`, `DRAFT_counterfactual_ranking.sh`, and the previous outcome launcher pass shell syntax checks.
- The configuration script expands to exactly four task/seed cells.
- The buffer, score metrics, contact controller, and corrected W&B keys pass dependency-light tests.
- Existing outcome-falsification dependency-light checks still pass.

The full MuJoCo restore-step preflight runs inside the Torch Conda environment before training, where Meta-World and simulator dependencies are available.

## Limitations

- Exact simulator restoration is a diagnostic/research intervention, not a deployable model-free assumption.
- The anchor controller uses the known Sawyer hand and mechanism coordinate layout only to reach contact; it does not provide the successful task action.
- Each event performs up to 6,400 isolated simulator steps, in addition to rank updates and the independent causal probe.
- Ranking only the first five repeated actions may still be too local for tasks requiring temporally extended contact or direction changes. That is an explicit falsification outcome, not a reason to start a longer run prematurely.
