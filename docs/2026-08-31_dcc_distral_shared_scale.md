# DCC shared-scale pilot motivated by Distral

## Claim being tested

Distral writes a task policy as a shared policy term plus a task-specific
term. Its coefficient alpha controls how strongly the shared policy prior
affects the task policy. For DCC with inner-product scores,

`f_k(s,a,g) = alpha * f_shared(s,a,g) + f_task(s,a,g)`

has the same two-column score form. This motivates a useful but limited
claim: alpha may control how strongly the shared critic guides the current
task actor.

This is not an exact implementation of Distral. DCC does not directly add a
KL loss between the task policy and a learned shared policy. Its actor is a
continuous tanh-Gaussian policy trained to maximize a critic score, and both
critic branches keep learning. The experiment therefore tests a
"Distral-like prior strength" interpretation, not mathematical equivalence of
the two algorithms.

## Efficient first experiment

Run only the first six curriculum tasks, ending with Task 5
`sawyer_handle_press_side`. This is the task where the submitted DCC paper
reported its clearest gain over all nine non-DCC contrastive variants. The
first five tasks give the shared branch a real transfer history, unlike a
single-task run.

First train Tasks 0-4 once with normal DCC for matched seeds `{5, 6, 7}`.
Then load each exact Task-4 checkpoint six times and train only Task 5 with
fixed alpha in `{0, 0.25, 0.5, 0.75, 1, 1.5}`. This holds the learned shared
representation fixed before the alpha intervention. It costs about 33 million
environment steps instead of 108 million. Three seeds share each GPU.

Alpha is applied inside every plain-DCC critic scoring path. Alpha zero keeps
the dynamics auxiliary training the shared body, but removes the shared
contrastive branch from the score. The shared goal encoder is still carried
across tasks, so alpha zero is not a complete no-transfer baseline.

Launch the shared prefix first:

```bash
sbatch DRAFT_dcc_shared_scale_task5_prefix.sh
```

After all three prefix runs finish Task 4, launch the Task-5 branches:

```bash
sbatch DRAFT_dcc_shared_scale_task5.sh
```

## Metrics

The normal evaluation success metrics remain the primary outcome. Compare
Task-5 best success, success curve area, and steps needed to cross fixed
success levels. Tasks 0-4 are controls for large performance regressions.

The learner also records:

- `decomp/shared_scale`: requested alpha;
- `decomp/shared_norm`, `decomp/task_norm`, and `decomp/combined_norm`;
- `decomp/scaled_shared_to_task_norm`: actual representation balance after
  alpha is applied;
- `decomp/shared_task_cosine`: whether the two branches agree or cancel;
- `decomp/shared_goal_score_abs` and `decomp/task_goal_score_abs`;
- `decomp/shared_score_fraction`: fraction of raw matched-goal score magnitude
  supplied by the scaled shared branch;
- critic loss, categorical and binary accuracy, dynamics loss, actor loss,
  entropy, and evaluation success already logged by the runner.

## Evidence for or against the hypothesis

Evidence in favor requires both representation and behavior to move together:

1. increasing alpha should increase `shared_score_fraction` or
   `scaled_shared_to_task_norm` across seeds;
2. Task-5 learning speed or final success should change consistently with
   that effective shared contribution; and
3. an intermediate alpha should give a useful trade-off if the shared branch
   helps but can also over-constrain the new task.

The claim is not supported if the task branch simply rescales itself so the
effective shared fraction stays unchanged, or if success is unchanged despite
a large change in shared contribution. If success changes while the effective
shared contribution does not, the likely cause is optimization scale rather
than a policy-prior effect.

Only after this pilot shows a stable pattern should Task 8 be used as a second
confirmation. A trainable scalar alpha is the next low-risk extension. A
state-goal-dependent alpha should wait until the fixed sweep shows that
different amounts of sharing are actually useful.
