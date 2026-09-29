# Handoff: DCC shared-scale (`alpha`) experiment

Date: 2026-09-29

## One-sentence goal

Test whether the coefficient on DCC's shared critic score behaves like the
strength of Distral's shared-policy prior, first on a seven-task continual
pilot and then, only if the pilot is useful, on nine or all ten tasks.

This document is written for the agent implementing and launching the next
experiment. It also records what has already been implemented so that the
same work is not repeated.

## Important task numbering

The code uses zero-based task IDs, while people often call the first task
"Task 1". The current sequence is:

| Human name | Code ID | Environment |
|---|---:|---|
| Task 1 | 0 | `sawyer_hammer` |
| Task 2 | 1 | `sawyer_push_wall` |
| Task 3 | 2 | `sawyer_faucet_close` |
| Task 4 | 3 | `sawyer_push_back` |
| Task 5 | 4 | `sawyer_stick_pull` |
| Task 6 | 5 | `sawyer_handle_press_side` |
| Task 7 | 6 | `sawyer_push` |
| Task 8 | 7 | `sawyer_shelf_place` |
| Task 9 | 8 | `sawyer_window_close` |
| Task 10 | 9 | `sawyer_peg_unplug_side` |

Therefore `--num_tasks=7` means human Tasks 1-7, and `--num_tasks=9`
means human Tasks 1-9. Use these exact terms in run names and reports.

## The algorithm change that already exists

DCC has a shared state-action representation and a task-specific
state-action representation. The original score is

```text
f(s,a,g) = alpha * <z_shared(s,a), z_goal(g)>
           +       <z_task(s,a),   z_goal(g)>.
```

The repository now supports three modes through
`--shared_repr_normalization`:

1. `none` (original DCC)

   ```text
   f = alpha * <z_shared, z_goal> + <z_task, z_goal>
   ```

2. `unit_mix` (fixed total score budget)

   ```text
   f = [alpha * <unit(z_shared), unit(z_goal)>
        +       <unit(z_task),   unit(z_goal)>] / (alpha + 1)
   ```

3. `unit_distral` (primary Distral test)

   ```text
   f = alpha * <unit(z_shared), unit(z_goal)>
       +       <unit(z_task),   unit(z_goal)>
   ```

Here `unit(x) = x / max(||x||_2, epsilon)` is applied per sample.

Use `unit_distral` for the main experiment. It prevents the network from
undoing alpha merely by shrinking or enlarging a branch, while preserving
the exact `alpha * shared + task` form used in the Distral connection. Do not
divide by `alpha + 1` in the primary test: that changes both the prior
exponent and the total score scale. Keep `unit_mix` as a separate control,
not as the main result.

The implementation is already in:

- `contrastive/decomposed_networks.py`: constructs the normalized score;
- `contrastive/continual_learning_decomposed.py`: logs branch magnitudes and
  effective coefficients;
- `run_continual_contrastive.py`: exposes `none`, `unit_mix`, and
  `unit_distral`;
- `contrastive/continual_config.py`: stores `shared_repr_scale` and the
  normalization mode;
- `tests/test_dcc_distral_unit_shared_scale.py`: checks the equation and
  configuration wiring.

The original behavior is unchanged because the default mode remains `none`.

## Why this is connected to Distral

For a fixed critic, write the normalized shared and task scores as

```text
f_shared_hat(s,a,g) = <unit(z_shared), unit(z_goal)>
f_task_hat(s,a,g)   = <unit(z_task),   unit(z_goal)>
f_alpha             = alpha * f_shared_hat + f_task_hat.
```

The maximum-entropy actor objective is

```text
J(pi) = E_pi[f_alpha - tau * log pi(a|s,g)].
```

Define the policy induced by the shared score:

```text
pi_0(a|s,g) proportional to exp(f_shared_hat(s,a,g) / tau).
```

Substituting this into the actor objective gives

```text
J(pi) = E_pi[f_task_hat]
        - alpha * tau * KL(pi || pi_0)
        + (1 - alpha) * tau * H(pi)
        + a constant.
```

Equivalently, the ideal soft policy has the form

```text
pi*(a|s,g) proportional to
    pi_0(a|s,g)^alpha * exp(f_task_hat(s,a,g) / tau).
```

This is the precise sense in which alpha resembles the strength of a Distral
policy prior: larger alpha gives the shared policy more influence, while the
task branch supplies the task-specific correction.

This is not a claim that DCC and Distral are identical. The derivation assumes
a fixed critic and an unrestricted soft policy. In practice, DCC learns the
critic and actor together, and its shared object is a representation learned
with contrastive and dynamics losses rather than Distral's explicit policy
distillation objective. The experiment tests whether the useful Distral
intuition survives those differences.

For `0 <= alpha <= 1`, the mapping is easiest to interpret:

```text
KL coefficient      = alpha * tau
entropy coefficient = (1 - alpha) * tau.
```

Keep `alpha=1.5` because it was in the original request, but describe it as a
stronger-than-standard-Distral extrapolation.

## What the previous Task-6 pilot did and did not test

The existing pilot files are:

- `experiment_configs_dcc_distral_unit_shared_scale_task5.py`;
- `DRAFT_dcc_distral_unit_shared_scale_task5.sh`;
- `docs/2026-09-04_dcc_distral_faithful_unit_scale_task5.md`.

Despite `task5` in those file names, they use code ID 5, which is human Task
6: `sawyer_handle_press_side`.

That pilot first learned code IDs 0-4 with the original `alpha=1` setup, then
branched from the same checkpoint and trained code ID 5 with different alpha
values. This is a good, cheap test of how strongly an already-learned shared
representation should be used on one new task. It is not a from-scratch test
of how alpha changes the shared representation learned over a curriculum.

The older unnormalized sweep showed that alpha can change performance, but
the result was confounded: as alpha increased, the network reduced the raw
shared output norm. The model partly cancelled the requested scale change.
`unit_distral` was added specifically to remove this compensation.

## Primary experiment: from scratch through seven tasks

Run a new curriculum from random initialization for every `(alpha, seed)`.
Do not use `--start_task` and do not resume from the old Tasks 0-4 checkpoint.
Use one constant alpha for the entire curriculum. This tests both how alpha
changes shared knowledge while it is learned and how that knowledge is used
on later tasks.

### Sweep

```text
alpha in {0, 0.25, 0.5, 0.75, 1, 1.5}
seed  in {5, 6, 7}
num_tasks = 7
steps_per_task = 1,000,000
base_steps = 1,000,000
```

This is 18 independent curricula and 126 million environment steps. Start
with this seven-task pilot. Do not launch nine or ten tasks until the pilot
passes its manipulation checks and shows that the comparison is informative.

### Keep everything else fixed

Use plain DCC:

```text
actor_mode = reset
critic_mode = decomposed
shared_repr_normalization = unit_distral
dyn_aux_weight = 1.0
network_width = 1024
critic_depth = 4
actor_depth = 4
phi_task_width = 256
phi_task_depth = 4
combine_mode = add
energy_fn = inner_product
eval_every = 50,000
eval_episodes = 10
sawyer_success_mode = native_info
goal_conditioning_mode = full_state
use_task_id = false
post_task_eval_scope = current
```

Do not add behavior cloning, an action-effect or advantage head, Bellman Q,
counterfactual supervision, extra negative repeats, interaction-weighted
relabeling, or a persistent actor. Those are different hypotheses.

Use this W&B group:

```text
DCC-DISTRAL-UNIT-ALPHA-CONTINUAL7-1M
```

Run names and local output paths must contain both alpha and seed, for example
`continual7_alpha0p25_seed6`. Never let two settings share a checkpoint or log
directory.

## Required original-DCC control

The normalized alpha sweep answers whether alpha controls the relative shared
contribution when branch magnitudes cannot compensate. It does not, by
itself, tell us whether normalization changed the algorithm.

Add a matched control with:

```text
alpha = 1
shared_repr_normalization = none
```

All other settings and seeds must match. If an existing corrected-wrapper run
has exactly these settings, reuse it. Otherwise launch three control runs.
The important comparison is:

```text
unit_distral, alpha=1  versus  original DCC, alpha=1.
```

If these differ greatly, discuss normalization as an algorithm change rather
than attributing every difference to prior strength.

## Files the implementing agent should add

Create:

```text
experiment_configs_dcc_distral_unit_shared_scale_continual.py
DRAFT_dcc_distral_unit_shared_scale_continual.sh
tests/test_dcc_distral_unit_shared_scale_continual.py
```

Use the existing Task-6 pilot config and launcher as a template, with these
changes:

1. Remove `start_task` and `resume_checkpoint_dir` completely.
2. Default to `num_tasks=7` and allow an environment variable such as
   `ALPHA_NUM_TASKS` with only `7`, `9`, or `10` accepted.
3. Allow `ALPHA_STEPS_PER_TASK`, defaulting to `1000000`.
4. Enumerate all six alpha values and seeds 5, 6, and 7.
5. Pass `--shared_repr_normalization=unit_distral` explicitly.
6. Give every alpha/seed pair a unique run, checkpoint, and log directory.
7. Run both the existing unit-Distral test and the new config/launcher test
   before starting training.
8. Do not edit the legacy wrapper or silently change default DCC behavior.

The launcher should support commands of this form:

```bash
# Seven-task pilot (recommended first)
ALPHA_NUM_TASKS=7 sbatch DRAFT_dcc_distral_unit_shared_scale_continual.sh

# Nine-task follow-up, only after reviewing the pilot
ALPHA_NUM_TASKS=9 sbatch DRAFT_dcc_distral_unit_shared_scale_continual.sh

# Full ten-task follow-up
ALPHA_NUM_TASKS=10 sbatch DRAFT_dcc_distral_unit_shared_scale_continual.sh
```

If the Slurm file uses one array element per alpha and launches the three
seeds in parallel, its array range is `0-5`. If it uses one array element per
run, its range is `0-17`. The config test must verify which convention is
used so that no setting is skipped or repeated.

## Tests that must pass before launch

The new test should verify:

- exactly 18 primary configurations are produced;
- every alpha has exactly seeds 5, 6, and 7;
- every run starts from task 0 with no resume checkpoint;
- `num_tasks` is 7 by default and accepts only 7, 9, or 10;
- the score mode is always `unit_distral`;
- all plain-DCC settings above are fixed;
- all run/checkpoint paths are unique;
- the launcher passes alpha and normalization mode explicitly;
- the launcher array covers every configuration exactly once;
- invalid task counts and invalid alpha values fail early;
- original `shared_repr_normalization=none` behavior remains unchanged.

Also run the existing relevant tests:

```bash
python tests/test_dcc_distral_unit_shared_scale.py
python tests/test_dcc_distral_unit_shared_scale_continual.py
```

## Metrics required to answer the hypothesis

Performance alone is not enough. We need both a manipulation check and a
behavioral result.

### Manipulation check

Record at least:

```text
decomp/shared_scale
decomp/shared_coefficient
decomp/task_coefficient
decomp/shared_norm_raw
decomp/task_norm_raw
decomp/shared_norm_effective
decomp/task_norm_effective
decomp/scaled_shared_to_task_norm
decomp/shared_goal_score
decomp/task_goal_score
decomp/shared_score_fraction
decomp/branch_cosine
```

Use the exact W&B key names currently produced by the learner if they differ
slightly from this list; do not create duplicate aliases unnecessarily.

Under `unit_distral`, the effective shared-to-task norm ratio should be alpha,
up to numerical error. That is the basic proof that the experiment actually
changed shared influence as intended. Raw branch norms may still change, but
they should no longer cancel the forward-pass coefficient.

Also record the usual critic loss/accuracy, dynamics loss, actor loss and
entropy. These help distinguish a useful prior-strength effect from critic
instability or a simple temperature effect.

### Performance summaries

For every task and run, save:

- the complete evaluation-success learning curve;
- success AUC over a fixed one-million-step window;
- success at fixed checkpoints, especially 250k, 500k, 750k, and 1M;
- best success;
- mean of the last three evaluations;
- time or steps to reach 50%, 80%, and 90% success, when reached;
- performance on earlier tasks after later tasks if a compatible retained-task
  evaluation is already available. Do not change the training protocol solely
  to obtain this metric.

Compare matched seeds. Report the mean and uncertainty across seeds, but also
show individual seed curves so that one lucky seed is visible.

## How to decide what the result means

### Strong support

The hypothesis is strongly supported if:

1. the effective norm-ratio check tracks alpha;
2. increasing alpha causes a consistent, ordered change in learning behavior
   across matched seeds and several tasks;
3. larger alpha helps where transfer from previous tasks is useful, while a
   smaller alpha helps where the old shared solution causes interference; and
4. the result is not explained only by critic collapse, actor entropy, or the
   normalization control.

The best alpha does not need to be the same for every task. In fact, stable
task-dependent preferences would motivate a later trainable `alpha(s,g)`.

### Partial support

The hypothesis is partially supported if the manipulation check passes and
alpha reliably changes performance, but the direction is not monotonic or is
strongly task-dependent. Then alpha is a real shared-influence control, but
not a single universal prior-strength hyperparameter.

### Evidence against the practical hypothesis

The useful claim is not supported if the norm-ratio check passes but curves
are statistically indistinguishable, chaotic across seeds, or explained by
critic/temperature instability. That would mean the algebraic Distral
connection is still valid for a fixed critic, but alpha is not acting as a
useful policy-prior control in the learned continual system.

## Recommended sequence

1. Implement and test the seven-task from-scratch launcher.
2. Run the three original-DCC `alpha=1, none` controls if exact runs do not
   already exist.
3. Launch the 18-run seven-task `unit_distral` sweep.
4. Check the effective norm ratio before waiting for every run to finish.
5. Analyze per-task curves and matched-seed summaries.
6. Continue to nine tasks only if the seven-task result is interpretable.
7. Use all ten tasks only for the final claim.
8. Consider a trainable state/goal-dependent alpha only after fixed alpha
   shows stable task-dependent preferences.

The old branch-from-checkpoint pilot and this new experiment answer different
questions. Keep both: the old pilot tests use of an already learned prior on
one task; the new run tests learning and using that prior across a curriculum.
