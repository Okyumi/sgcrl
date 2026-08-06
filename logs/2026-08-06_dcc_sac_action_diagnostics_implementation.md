# DCC-SAC, AC-DCC, and shortcut-diagnostic implementation log

Date: 2026-08-06

## Motivation

The previous additive RBC-DCC critic failed because the Bellman residual
overwhelmed the contrastive base.  More importantly, ordinary DCC can solve
its goal-classification loss through a shortcut that barely depends on the
action.  High batch-classification accuracy is therefore not sufficient
evidence that the learned score is useful for control.

This implementation treats action insensitivity as the primary hypothesis.

## Modes

### decomposed

Unchanged DCC training objective.  It now supports optional periodic
shortcut/action diagnostics.  The default diagnostic interval is zero, so
legacy DCC runs keep the same training path.

### dcc_sac

DCC and twin SAC Q are independent.

DCC is trained with its original InfoNCE and dynamics objectives.  Q receives
raw [state, goal, action] inputs and is trained from canonical HER reward and
discount semantics.  There is no TD path into any DCC parameter.

The actor retains the DCC objective.  A Q correction is added only when:

1. the Q warm-up is complete;
2. EMA absolute TD error is below its configured threshold;
3. EMA normalized twin disagreement is below its configured threshold; and
4. the relevant quantities are finite.

For each state and goal, Q is centered and scaled across candidate actions.
The normalized correction is clipped and multiplied by a ramped gate and a
small beta.  At gate zero, the actor objective is ordinary DCC exactly.

The gate establishes numerical stability, not semantic usefulness.  Q action
variance and DCC-Q ranking agreement are therefore logged separately and must
be inspected before increasing beta.

### dcc_sac_separate

A required ablation.  DCC is trained in parallel, but the actor uses standard
SAC Q only.  This mode measures whether the proposed fusion is doing more than
sparse SAC plus an unused representation learner.

### action_dcc

Reward-free Action-Contrastive DCC (AC-DCC).  In addition to the original
goal-contrastive matrix, it constructs an action matrix for a replay subset:

* hold the current state and achieved next-state goal fixed;
* vary only the action across actions from other replay transitions;
* classify the actual replay action as the positive.

This directly trains the fixed-(state, goal) action-ranking direction needed
by the actor.  It does not use environment reward, HER reward, or a Bellman
target.

### action_dcc_sac

AC-DCC plus the same stable, normalized Q correction used by dcc_sac.  This
mode should be run only after the two simpler mechanisms are understood.

## Periodic diagnostic metrics

The diagnostic path runs outside the hot learner update and is controlled by
shortcut_diagnostic_interval.

Shortcut retention:

* shortcut/categorical_accuracy
* shortcut/action_shuffled_categorical_accuracy
* shortcut/zero_action_categorical_accuracy
* shortcut/action_shuffle_retention
* shortcut/zero_action_retention
* shortcut/logit_saturation_fraction
* shortcut/positive_negative_margin

Action sensitivity and ranking:

* action/dcc_shuffle_delta_rms
* action/dcc_shuffle_delta_abs
* action/dcc_candidate_std_policy
* action/dcc_candidate_std_uniform
* action/dcc_action_grad_norm
* action/dcc_progress_spearman
* action/q_candidate_std_policy
* action/q_candidate_std_uniform
* action/q_progress_spearman
* action/dcc_q_candidate_spearman
* q/twin_disagreement_periodic

AC-DCC training metrics:

* acdcc/action_contrast_loss
* acdcc/action_contrast_accuracy
* acdcc/action_contrast_margin
* acdcc/action_score_std
* acdcc/action_contrast_weight

DCC-SAC stability and tail metrics include Q/target mean, standard deviation,
minimum and maximum, mean/max TD error, absolute and normalized twin
disagreement, EMA stability statistics, HER success, gate value, effective
beta, and normalized correction magnitude.

The candidate-action standard deviations and DCC-Q agreement are genuinely
fixed-(state, goal) diagnostics.  The two `progress_spearman` fields are
cheaper replay-batch proxies: they correlate scores with observed one-step
progress across replay transitions and are not counterfactual environment
rollouts from the same state.  A causal action-ranking study still requires
checkpoint rollouts from resettable states.

## HER predicate used by Q modes

The canonical helper does not call each environment's native success
conditions.  It computes one strict Euclidean predicate,

`norm(achieved_next - relabeled_goal) < her_reward_threshold`.

Thus every coordinate in the selected goal projection contributes jointly,
but this is not a logical AND over task-specific predicates.  A transition
receives the success reward only when the joint vector lies inside the radius;
otherwise it keeps the failure reward.  This approximation must be validated
against the native success definition for tasks 5 and 8.

## Recommended first experiments

Run individual task 5 and task 8 before the continual curriculum.  Start with
one seed as a runtime smoke, then use matched seeds 5, 6, and 7.

Required matched modes:

1. decomposed with diagnostics;
2. dcc_sac;
3. dcc_sac_separate;
4. action_dcc;
5. action_dcc_sac only after modes 2 and 4 are stable.

Use:

* actor_mode=reset
* combine_mode=add
* goal_encoder_mode=shared
* shortcut_diagnostic_interval=1000
* post_task_eval_scope=current
* her_reward_threshold=0.05 for Q modes
* step_penalty_reward=true for Q modes

Interpret high action-shuffle or zero-action retention as evidence that the
contrastive classifier remains shortcut-solvable.  The desired AC-DCC
signature is lower shortcut retention together with higher fixed-state action
score variance and better success AUC, not merely higher classification
accuracy.

## Evaluation-cost control

The legacy task-boundary behavior remains post_task_eval_scope=all_seen.
Use current for individual probes or when past-task sweeps are unnecessary,
and none to disable the boundary sweep entirely.  Intra-task previous-task
evaluation remains disabled by default.

## Validation gate

Before large runs:

1. parse all modified Python and shell files;
2. run the dependency-light tests;
3. run one JAX learner update for each new mode;
4. confirm DCC-SAC gate remains zero during warm-up;
5. confirm TD gradients do not change DCC parameters;
6. confirm action_dcc produces no Q contribution;
7. inspect HER success, Q tails, TD EMA, twin EMA, and diagnostic cadence;
8. save and reload a task-boundary checkpoint.
