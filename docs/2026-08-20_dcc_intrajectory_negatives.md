# DCC in-trajectory negatives for Sawyer Task 5 and Task 8

Date: 2026-08-20

## Source and method

*Scaling the Horizon of Contrastive Reinforcement Learning* reports that easy
cross-trajectory negatives can let a contrastive critic avoid learning the
within-trajectory temporal distinctions required for long-horizon control. Its
StableCRL recipe draws several state--future-goal pairs per trajectory so the
other goals from that trajectory become hard negatives. The reported default
is `r=12`.

Sources: [paper](https://david-yan1.github.io/pdfs/builderbench.pdf),
[OpenReview](https://openreview.net/forum?id=knI0KISsbw), and the official
[CRTR repository](https://github.com/Princeton-RL/CRTR) that supplies the
repeated-trajectory sampling principle.

The implementation adds `--in_trajectory_negative_repeats`. Its default `1`
uses the existing replay pipeline unchanged. For `r>1`, each replay episode
produces independently sampled anchors with SGCRL's discounted, strictly-future
goal relabeling. DCC's normal `B x B` InfoNCE matrix supplies the negatives.

For Sawyer's batch size 256 and `r=12`, each batch contains 21 full episode
groups of 12 plus four rows from a final episode. Checkpoint/log identities
include `_itn12` and single-task checkpoint paths also include the environment.

## Experiments

The original persistent-actor cells remain indices 0--5. New cells are:

| Indices | Environment | Seeds | Dynamics | Actor | `r` |
|---|---|---:|---:|---|---:|
| 6--8 | `sawyer_handle_press_side` (Task 5) | 5, 6, 7 | 1.0 | reset | 12 |
| 9--11 | `sawyer_window_close` (Task 8) | 5, 6, 7 | 1.0 | reset | 12 |

These probes isolate negative sampling: DCC dynamics stays enabled and the
existing adaptive entropy objective is unchanged. The complete StableCRL
recipe also reports entropy coefficient `0.01`, which is deliberately excluded
from this first comparison.

Run a short TensorFlow/Reverb/Meta-World cluster smoke test before the full
8-million-step jobs.

## Torch HPC launch

`DRAFT.sh` remains the canonical Torch launcher and still runs the full active
configuration matrix by default. `DRAFT_intrajectory.sh` selects only indices
6--11 and maps three experiments to each L40S GPU:

```bash
sbatch DRAFT_intrajectory.sh
```

Array task 0 runs Task 5 seeds 5/6/7; array task 1 runs Task 8 seeds 5/6/7.
