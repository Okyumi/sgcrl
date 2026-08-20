# DCC same-state action-landscape diagnostic

## Question

Task 5 (`sawyer_handle_press_side`) and Task 8
(`sawyer_window_close`) can fail despite high contrastive goal-categorization
accuracy. This experiment separates two non-exclusive explanations:

1. **Bad in-distribution landscape:** DCC already ranks replay-supported
   alternative actions incorrectly for task-relevant multi-step progress.
2. **Actor exploitation:** DCC ranks replay-supported actions reasonably, but
   the actor selects off-support actions with unusually high critic scores and
   poor real outcomes.

The old `action/dcc_progress_spearman` cannot distinguish these explanations:
it correlates scores and progress across different replay states. The new
probe changes the action while holding the simulator state and fixed task goal
constant.

## Causal intervention

At each diagnostic event, a separate evaluator environment is advanced under
the current policy to an anchor state. The probe snapshots MuJoCo state,
mocap/controller buffers, wrapper counters, cached observations, and RNG
states. From that identical snapshot it tests four action families:

- deterministic/stochastic actions from the current policy;
- local perturbations around the deterministic policy action;
- actions from replay states nearest to the anchor state;
- uniformly random actions.

Each candidate is applied once, followed by the same deterministic
continuation policy for 24 more steps (25 total). The probe records DCC score,
one-step and rollout progress for the full goal, rollout progress for the
task-relevant mechanism coordinates (unified state indices `4:7`), and task
success. The continuation RNG is reset for every candidate so only the first
action differs.

## Primary interpretation

| Evidence | Interpretation |
|---|---|
| Low/negative `replay_score_vs_rollout_mechanism_spearman` or high `replay_top_score_regret` | The critic landscape is already wrong on replay-supported actions. |
| Reasonable replay Spearman/regret, high `policy_score_percentile`, low `policy_outcome_percentile`, and large `policy_replay_support_distance` | The actor is exploiting high-scoring off-support regions. |
| Both signatures | A miscalibrated landscape is being amplified by actor exploitation. |
| Neither signature but success remains poor | The 25-step horizon/anchor distribution is insufficient, or the policy fails for a separate exploration reason. |

The mechanism-progress metrics are primary. Full-goal metrics include the
hand and gripper target fields and may therefore reproduce the shortcut that
the experiment is auditing.

## Shortcut-coordinate audit

Every 50 learner calls, the existing categorical probe now separately
shuffles goal-side hand (`0:3`), gripper (`3:4`), and mechanism (`4:7`)
coordinates across batch rows. The corresponding categorical-accuracy drops
measure which block the critic uses to retrieve future goals. These mixed
goals are off-manifold, so the result is a cue-reliance audit, not by itself a
causal explanation of policy failure.

## Experiments and launch

The active `experiment_configs.py` cells are:

| Indices | Task | Seeds | Algorithm | Dynamics |
|---|---|---:|---|---:|
| 12--14 | handle press side (Task 5) | 5, 6, 7 | plain DCC | 1.0 |
| 15--17 | window close (Task 8) | 5, 6, 7 | plain DCC | 1.0 |

Run on Torch HPC:

```bash
sbatch DRAFT_action_landscape.sh
```

The wrapper first runs restore--step reproducibility checks on both
environments and aborts if the installed MuJoCo/wrapper stack cannot reproduce
an identical transition. Diagnostic checkpoints use a separate directory so
plain-DCC checkpoints cannot cause false auto-resume.

No loss or actor update is changed, and all new runtime paths are disabled by
default.
