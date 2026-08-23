# Outcome-Calibrated Sequence DCC: staged 1M-step falsification

Date: 2026-08-22  
Branch: `section3_done`  
Tasks: Task 5 (`sawyer_handle_press_side`) and Task 8
(`sawyer_window_close`)

## Motivation

The 3M-step IWR-DCC, Advantage-DCC, and Bridge-DCC pilots did not improve
matched-horizon CRTR performance.  The action-effect head reached high cosine
agreement with its target, but its same-state score ranking remained nearly
uncorrelated with measured 100-step mechanism progress.  Task-8 success also
rose transiently and then collapsed.  The next experiment must therefore
separate three conjectures instead of combining them:

1. The DCC score numerically or directionally overwhelms a useful local head.
2. The head target itself is wrong because one-step movement in learned
   $\psi$ geometry is not task progress.
3. A useful policy is briefly discovered but forgotten without explicit
   retention.

The full numerical diagnosis motivating these stages is recorded in
`docs/2026-08-22_bridge_dcc_3m_refined_diagnosis.md`.

## Common representation learner

All stages retain CRTR/DCC representation learning with in-trajectory
repetition $r=12$ and the decomposed DCC dynamics auxiliary.  DCC is therefore
an auxiliary representation learner in this experiment; only the actor credit
path changes between stages.

## Stage 1: effect-only actor with the existing $\psi$ target

The task-local head remains the implemented Bridge-DCC vector head:

$$
u_\omega(s_t,a_t) \approx
\gamma\,\hat\psi(s_{t+1})-\hat\psi(s_t),
$$

where $\hat\psi(x)=\psi(x)/\lVert\psi(x)\rVert_2$.  The local action score is

$$
A_\omega(s,a,g)=
\tanh\!\left(
\frac{u_\omega(s,a)^\top\hat\psi(g)}{\tau}
\right).
$$

Unlike Advantage-DCC, the Stage-1 actor does **not** include the DCC score:

$$
J_1(\pi)=
\mathbb E_{a\sim\pi(\cdot\mid s,g)}
\left[A_\omega(s,a,g)+\alpha\mathcal H(\pi(\cdot\mid s,g))\right].
$$

Insertion point: `actor_loss_fn` in
`contrastive/continual_learning_decomposed.py`.  This stage changes only the
actor aggregation (`action_effect_actor_mode=effect_only`); the head target,
replay, DCC loss, dynamics loss, and network output dimension remain unchanged.

Interpretation: if Stage 1 passes while Advantage-DCC failed, DCC/head gradient
competition or score-scale domination was causal.  If Stage 1 still has poor
same-state ranking, the learned $\psi$-movement target is rejected.

## Stage 2: raw finite-horizon outcome credit

For a relabeled goal $g$, define mechanism distance

$$
d_M(s,g)=\lVert s_{4:7}-g_{4:7}\rVert_2.
$$

The non-bootstrapped $H$-step targets are

$$
y_t^{(H)}(g)=d_M(s_t,g)-
\min_{1\le h\le H}d_M(s_{t+h},g),
$$

and

$$
c_t^{(H)}(g)=
\max_{1\le h\le H}
\mathbf 1\!\left[d_M(s_{t+h},g)\le\epsilon\right].
$$

The first probe uses $H=25$ and $\epsilon=0.05$.  A goal-conditioned
task-local MLP receives $(s,g,a)$ and emits a standardized progress prediction
$\hat y_\omega$ and success logit $\hat c_\omega$.  Its loss is

$$
\mathcal L_{\mathrm{outcome}}=
\lambda_y\operatorname{Huber}(\hat y_\omega,\widetilde y_t^{(H)})+
\lambda_c\operatorname{BCEWithLogits}(\hat c_\omega,c_t^{(H)}),
$$

where $\widetilde y_t^{(H)}$ uses task-local EMA mean/variance with a
$0.01$ standard-deviation floor.  There is no Bellman bootstrap.  The actor
maximizes only

$$
J_2(\pi)=\mathbb E
\left[
\hat y_\omega(s,g,a)+
\lambda_s\sigma(\hat c_\omega(s,g,a))+
\alpha\mathcal H(\pi)
\right].
$$

Insertion points:

- `run_continual_contrastive.py`: derives labels from the sampled replay
  episode before row shuffling.
- `contrastive/outcome_credit.py`: shared TensorFlow label implementation and
  a cluster-environment self-test.
- `contrastive/decomposed_networks.py`: switches the task-local head input to
  $(s,g,a)$ and output dimension to two only in `raw_horizon` mode.
- `contrastive/continual_learning_decomposed.py`: trains the two-output head,
  updates target normalization, and supplies its score to the actor.

Interpretation: Stage 2 passing after Stage 1 fails supports the conjecture
that $\psi$ geometry, rather than DCC/head aggregation, supplied the wrong
notion of progress.

## Stage 3: raw outcome credit plus success retention

Stage 3 leaves Stage 2 unchanged and adds a task-local ring buffer
$\mathcal B^+$ of transitions whose **original task goal** is reached within
$H=25$.  The buffer stores the original $(s,g_{\mathrm{task}},a)$ rather than
the relabeled goal.  The actor loss adds

$$
\mathcal L_{\mathrm{BC}}=
-\mathbb E_{(s,g,a)\sim\mathcal B^+}
\log\pi(a\mid s,g),
$$

with total weight $\lambda_{\mathrm{BC}}=0.1$, buffer capacity $4096$, and BC
batch size $64$.  The ring buffer is part of the learner state and resets with
the task-local actor/head.

Interpretation: Stage 3 is supported only if Stage 2 learns positive action
ranking but still loses late success, and Stage 3 improves late/peak retention.

## Exact 1M experiment sequence

`experiment_configs_outcome_falsification.py` contains 12 cells:

| Indices | Stage | Task/seed grid | Budget |
|---:|---|---|---:|
| 0--3 | Effect-only actor, current $\psi$ target | Tasks 5/8, seeds 5/6 | 1M |
| 4--7 | Raw $H=25$ outcome head | Tasks 5/8, seeds 5/6 | 1M |
| 8--11 | Raw outcome + success retention | Tasks 5/8, seeds 5/6 | 1M |

Every cell evaluates every 50K steps and runs the causal same-state landscape
probe every 250K steps, producing four probe events in a 1M run.  The causal
probe uses a 100-step rollout so the head is not evaluated only against its own
25-step label horizon.

Submit one stage at a time:

```bash
cd /scratch/yd2247/sgcrl
git pull --ff-only origin section3_done
FALSIFICATION_STAGE=1 sbatch DRAFT_outcome_falsification.sh
```

After the Stage-1 jobs finish, evaluate the registered gates:

```bash
python scripts/evaluate_outcome_falsification.py --stage 1
```

Repeat with `FALSIFICATION_STAGE=2` or `3` only when the preceding result
justifies that test.  The evaluator exits nonzero when any gate fails.

## Promotion gates

Both tasks must pass across seeds 5/6:

| Gate | Threshold |
|---|---:|
| Success AUC gain over matched first-1M CRTR | at least $+0.05$ |
| Late success / peak success | at least $0.80$ |
| Same-state score vs 100-step mechanism-progress Spearman | at least $0.30$ |
| Fixed-$(s,g)$ head score standard deviation across actions | at least $0.001$ |
| Head score retention after action shuffling | at most $0.50$ |

The gate command compares with `DCC-intrajectory-negatives-task5` and
`DCC-intrajectory-negatives-task8`, restricted to seeds 5/6 and the first 20
50K evaluations.

Only one winning stage should be expanded to seeds 5/6/7 and 8M steps:

```bash
OUTCOME_WINNER_STAGE=2 sbatch DRAFT_outcome_promotion.sh
```

`experiment_configs_outcome_promotion.py` refuses to enumerate runs unless
`OUTCOME_WINNER_STAGE` is one of 1, 2, or 3.

## Logged metrics

New dense metrics:

- `outcome/progress_loss`
- `outcome/success_loss`
- `outcome/progress_target_mean`
- `outcome/progress_target_std`
- `outcome/progress_prediction_std`
- `outcome/progress_pearson`
- `outcome/action_shuffle_delta`
- `learner/outcome/action_shuffle_retention`
- `learner/outcome/fixed_state_action_std`
- `outcome/success_rate`
- `action_effect/head_score_std`
- `action_effect/head_to_dcc_ratio`
- `retention/bc_loss`
- `retention/bc_active`
- `retention/buffer_size`

Existing causal metrics remain the decisive ranking evidence, especially
`learner/action_landscape/score_vs_rollout_mechanism_progress_spearman`, action-family outcome
means, candidate-score standard deviation, and top-score regret.

## Validation and preflight

The repository checks cover Python compilation, shell syntax, exact 12-cell
enumeration, stage isolation, promotion expansion, and configuration emission.
The launchers set both `ACTION_LANDSCAPE_SELF_TEST=true` and
`OUTCOME_FALSIFICATION_SELF_TEST=true`.  These tests run **after** the
`contrastive_rl` Conda environment is activated, avoiding the prior system
Python `ModuleNotFoundError` failure.

## Known limitations

- The raw mechanism target uses Sawyer coordinates `4:7`; it is deliberately a
  causal Task-5/Task-8 diagnostic, not yet a benchmark-general reward design.
- The head attributes an $H$-step outcome to the first action and does not yet
  model an action chunk.  Chunked control is the next conditional experiment
  only if the $H$-step label is informative but first-action ranking fails.
- Success retention uses a distance proxy to the original task goal, not the
  environment's hidden success predicate.
- The success buffer retains only actions present in replay; it cannot create
  successful contact transitions when exploration never discovers them.
- A 1M pass is a falsification signal, not a final performance estimate.  Only
  the pre-registered winner is eligible for the 8M promotion.
