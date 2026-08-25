# Task-5/Task-8 historical DCC checkpoint reevaluation

Date: 2026-08-25
Status: implemented; ready to run against the six existing C2 checkpoints.

## Question

Were Task 5 and Task 8 genuinely unsolved by the historical DCC actors, or did
the legacy full-3D wrapper merely report successful trajectories as failures?

Because DCC is reward-free, retraining after changing the wrapper reward would
not answer this question. The evaluator therefore loads each historical
`composed_policy` and performs deterministic evaluation only. No learner,
replay buffer, adder, training actor, or gradient update is created.

## Paired measurement

Every trajectory is scored under both definitions:

$$
y_{legacy}=\mathbb{1}[\lVert m-g\rVert_2 < \epsilon],
$$

$$
y_5=\mathbb{1}[|z_{handle}-z_{goal}|\leq0.02],\qquad
y_8=\mathbb{1}[|x_{window}-x_{goal}|\leq0.05].
$$

Both values come from the same observations in the same episode. The program
also verifies at every step that its reconstructed legacy flag equals the
reward emitted by the legacy wrapper. Any disagreement aborts the result.

The matrix is Task 5/Task 8 by seeds 5/6/7, with 100 episodes per checkpoint.
The documented C2 checkpoint root is
`/scratch/yd2247/sgcrl/logs/continual_checkpoints`. On disk, the original C2
seeds 5/6/7 live in the pre-disambiguation directory

`actor_reset_critic_decomposed_tid_False_heads_True/seed_{seed}/task_{k}.pkl`

(`k` is 5 for handle-press-side and 8 for window-close). That identity is
reset actor, decomposed critic, no task ID. The later
`..._dyn1.000_pt256x4` suffix was added after those runs; it does **not**
contain Task-5/Task-8 checkpoints for seeds 5/6/7. Actor network width and
depth are inferred directly from the saved policy parameters, including the
historical 1024-wide actor.

## Run

```bash
sbatch DRAFT_task58_checkpoint_reeval.sh
```

The launcher uses one EGL GPU only because this cluster's `mujoco_py` setup
requires it. Two evaluation processes share the allocation. There is no GPU
training workload.

If the checkpoints live under a different root:

```bash
CHECKPOINT_ROOT=/path/to/continual_checkpoints \
  sbatch DRAFT_task58_checkpoint_reeval.sh
```

After the three array jobs finish:

```bash
python scripts/summarize_task58_checkpoint_reevaluation.py \
  logs/task58_checkpoint_reeval/results/*.json \
  --output logs/task58_checkpoint_reeval/summary.json
```

Primary outputs are `legacy_success_rate`, `task_axis_success_rate`, their
paired gain, and `axis_rescued_success_rate` for each checkpoint.

## Interpretation

- A large positive axis gain means the old evaluator hid successful behavior.
- A near-zero gain with low axis success means the actors genuinely failed;
  changing wrapper reward cannot improve reward-free DCC training.
- A partial gain means both occurred: the metric was too strict, but the
  learned policy still needs improvement.

This reevaluation diagnoses measurement only. It does not make a new causal
claim about goal representation, exploration, or action credit.
