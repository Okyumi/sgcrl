# DCC in-trajectory negatives for Sawyer Task 5 and Task 8

Date: 2026-08-20

## Motivation and source

*Scaling the Horizon of Contrastive Reinforcement Learning* reports that
ordinary contrastive RL becomes unreliable on long-horizon tasks because its
critic can separate easy cross-trajectory negatives without learning the
within-trajectory temporal distinctions needed for long-horizon control. Its
StableCRL recipe samples multiple state--future-goal pairs from each replay
trajectory, making the other goals from the same trajectory hard negatives.
The reported default is a repetition factor of `r=12`; the paper observes most
of the gain by `r>=6` and diminishing returns beyond 12.

The paper is available from the
[authors' site](https://david-yan1.github.io/pdfs/builderbench.pdf) and
[OpenReview](https://openreview.net/forum?id=knI0KISsbw). A public StableCRL
repository was not found as of this date. The implementation therefore follows
the repeated-trajectory batching principle in the official
[CRTR repository](https://github.com/Princeton-RL/CRTR), which the StableCRL
paper identifies as the source of in-trajectory negatives; the corresponding
[CRTR paper](https://arxiv.org/abs/2508.13113) provides the method background.

## Implementation

The feature is opt-in through `--in_trajectory_negative_repeats=N`. `N=1` is
the default and takes the existing replay pipeline unchanged. For `N>1`, each
sampled replay episode produces `N` independently sampled anchor states. Each
anchor keeps SGCRL's existing discounted, strictly-future goal relabeling. The
ordinary DCC `B x B` InfoNCE matrix then treats the other rows from the same
episode as in-trajectory negatives; the critic and actor losses do not change.

Sawyer uses a critic batch size of 256, which is not divisible by 12. The new
pipeline samples 22 episodes per critic batch, keeps 21 complete groups of 12
and four rows from the final episode, and therefore retains exactly 256 rows.
This preserves the previous batch size and optimizer settings.

Checkpoint and log identities include `_itn12`; single-task checkpoint paths
also include the environment name. Consequently, Task 5 and Task 8 runs with
the same seed cannot overwrite each other, and neither can collide with any
legacy `r=1` run.

## Experiment matrix

The six existing persistent-actor DCC configurations remain unchanged at
indices 0--5. The following cells were appended:

| Indices | Environment | Seeds | DCC dynamics | Actor | Repeats |
|---|---|---:|---:|---|---:|
| 6--8 | `sawyer_handle_press_side` (Task 5) | 5, 6, 7 | 1.0 | reset | 12 |
| 9--11 | `sawyer_window_close` (Task 8) | 5, 6, 7 | 1.0 | reset | 12 |

These single-task probes intentionally change only negative sampling. DCC
dynamics remains enabled, actor reset behavior remains the Sawyer baseline,
and the existing adaptive entropy objective is unchanged. The full StableCRL
recipe also reports a lower entropy coefficient (`0.01`), but that is excluded
here to isolate the requested in-trajectory negatives.

The probes enable shortcut/action diagnostics every 1,000 learner updates and
save probe data. Compare final success and success AUC against the existing DCC
Task 5/8 baselines, while checking categorical accuracy, binary accuracy,
action sensitivity, and the shortcut diagnostics.

## Validation

- Python modules compile and launchers pass `bash -n`.
- `experiment_configs.py --total` reports 12 cells and preserves the first six
  legacy cells at `r=1`.
- The dependency-light suite covers grouping, future-goal sampling, checkpoint
  identity, launcher forwarding, and preservation of persistent-actor cells.

A short cluster smoke run should still precede the full 8-million-step jobs
because local validation does not exercise the complete
TensorFlow/Reverb/Meta-World GPU stack.
