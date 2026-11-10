# Vanilla CRL with critic gate and success BC on Jubail

## Motivation and experimental definition

This is the actual undecomposed CRL control for DCC. It uses the existing
`ContinualContrastiveLearner` and `make_networks`, with a single state action
encoder and a single goal encoder. There is no shared/task critic decomposition,
no task specific 256-wide branch, and no critic or actor transfer across tasks.
Both actor and critic reset, including their optimizer state, at each boundary;
the success buffer also starts empty. This is **not** the earlier no-transfer
DCC ablation, which retained the decomposed architecture. Its parameter count is
therefore lower than full DCC; it is not a parameter-count-matched control.

The paper seed set is 5–14 inclusive, taken from
`experiment_configs_full_critic_gate_5seed.py` (despite that legacy filename).
Each seed trains all ten tasks in `CONTINUAL_TASK_SEQUENCE`, 8 million environment
steps per task. The actor and vanilla critic use residual networks of width 1024
and depth 4. Other settings follow that full-DCC config: two collection actors,
native-info Sawyer success, full-state goals, HER actor goals, adaptive entropy,
100k evaluation interval, 10 evaluation episodes, success buffer capacity 4096,
BC batch 64, lambda 0.1, local gate noise 0.1, eight action/state probes.

## Objective

The ordinary contrastive CPC critic objective is unchanged. For policy loss,

`L_actor = E[alpha log pi(a|s,g) - f(s,a,g)]
           + 0.1 E_(s,a,g in D_success)[stop_gradient(w(s,g)) * -log pi(a|s,g)]`.

Here `f(s,a,g) = phi(s,a)^T psi(g)` and
`w = sigma_state / (sigma_state + sigma_action)`; when the denominator is at
most 1e-6, the gate is one. State probes add Gaussian noise scaled by 0.1 times
the success minibatch's coordinatewise state standard deviation (floored at
1e-6); the goal and sampled policy action stay fixed. Action probes are uniform
in [-1,1]. BC is zero until the success buffer contains data.

The success label is `episode_sparse_reward`, matching the selected full-DCC
comparison: retain sampled transitions from source episodes with any positive
native success reward, paired with the **original commanded goal**, not the
HER-relabeled goal. This includes the approach trajectory, not only successful
terminal states. The ring buffer is filled from SGD minibatches, as in DCC, so
replayed transitions can be inserted more than once.

The original vanilla learner's update ordering is preserved: its actor uses
critic parameters from before that SGD critic update. The DCC learner currently
uses updated critic parameters. Thus these are the existing vanilla CRL update
semantics with gate and BC added, rather than a rewrite of vanilla training.

## Code changes

- `contrastive/vanilla_success_bc.py`: JAX success ring buffer, local gate and BC
  loss; gate gradients cannot alter the actor or critic.
- `contrastive/continual_learning.py`: optional buffer in training state, insertion
  per SGD minibatch, actor BC term and retention metrics. BC defaults to zero,
  preserving the old loss/RNG path. This new path explicitly requires reset/reset,
  local gating, no recency weighting or warmup, and no success-score objective.
- `run_continual_contrastive.py`: supplies action dimension; adds
  `max_tasks_per_invocation` (default zero, unlimited); saves task checkpoints
  atomically **after** boundary evaluation and W&B finish. Failed or partial
  writes cannot become completed-task checkpoints. Records `task_complete`.
- `experiment_configs_vanilla_crl_gate_bc.py`: seeds, production/smoke settings,
  CLI arguments; removes irrelevant decomposed-critic options.
- `scripts/vanilla_crl_gate_bc.py`: noninteractive W&B preflight, launch and status.
- `scripts/submit_vanilla_crl_gate_bc.sh` and
  `scripts/jubail_vanilla_crl_gate_bc.sbatch`: Jubail setup and Slurm continuation.

## Launch from the Mac SSH terminal on Jubail

Use the existing `sgcrl` checkout and environment. No API key is stored in code.

```bash
cd /scratch/yd2247/sgcrl
git switch section3_done
git pull --ff-only origin section3_done
bash scripts/submit_vanilla_crl_gate_bc.sh
```

The submission command creates a 2-hour, two-task smoke job (20k steps each,
separate checkpoints and W&B group), and a ten-element production array with
`afterok` dependency on the smoke job. Each array index maps to seed `index + 5`.
The smoke first runs numerical tests and an actual JIT learner update using real
networks on synthetic successful transitions, then tests Sawyer training,
evaluation, task reset and online W&B logging. Failed smoke dependencies cancel
the production array instead of launching expensive runs.

Each production allocation requests one A100, eight CPUs, 64 GB, and 48 hours.
It runs **one complete task**, queues an `afterany` successor **before training**,
and exits at the task boundary. The successor automatically detects completed
tasks and advances. Completion of all ten tasks cancels the pending successor.
Ordinary Python errors cancel continuation; scheduler termination leaves it
queued. There are at most 20 allocations per seed per submission (ten required,
ten spare retries). A per-seed lock prevents simultaneous duplicate training.
Failed submission of a successor aborts before training. The commit and tracked
working tree are checked at each allocation so code versions cannot silently mix.

**This is task-boundary continuation, not a full mid-task replay checkpoint.**
If an individual task exceeds 48 hours or a node dies, that unfinished task is
restarted from its beginning. Earlier completed tasks remain saved. Consequently,
every individual 8M task must fit in one allocation; a task which repeatedly
exceeds 48 hours needs a full mid-task replay/RNG/optimizer checkpoint extension
or a longer queue. The overall 80M curriculum can span many allocations. Slurm
queue availability determines when a successor starts; immediate start is not
guaranteed. GPU smoke and actual allocation duration cannot be verified locally.

Do not change tracked files or the checkout commit while these jobs are pending.
To resume a stopped/error-corrected batch, inspect the logs, resolve the cause,
then rerun the submission command. Existing complete task checkpoints are reused.
Use a fresh `VANILLA_RUN_ROOT` for changes to scientific hyperparameters to avoid
mixing experiments, for example:

```bash
VANILLA_RUN_ROOT="$PWD/logs/vanilla_crl_gate_bc_v2" bash scripts/submit_vanilla_crl_gate_bc.sh
```

## W&B and monitoring

The jobs enforce online mode and `https://api.wandb.ai`. They use the existing
W&B login, `WANDB_API_KEY`, or the existing `$HOME/.wandb_api_key` file. An
unauthenticated run fails at preflight instead of prompting inside Slurm. If
login is missing, on the Jubail login node:

```bash
module load cuda/11.8.0 conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
wandb login --host https://api.wandb.ai
bash scripts/submit_vanilla_crl_gate_bc.sh
```

Project: `nyuad_mmvc/continual_gcrl_paper`.
Production group: `PAPER-VANILLA-CRL-GATE-BC-10SEED`.
Smoke group: the same with `-SMOKE`. One W&B run per task; retries create new runs.
Only include completed production runs (`task_complete=true`) in final tables;
if duplicate completed attempts exist, select one per seed/task consistently.
AUC/best/final can be computed from the usual evaluator success history and
boundary `eval/<environment>`/`eval/mean_success`; do not mix the two different
evaluation estimators. Final here means the end of that task's training, not a
retest after finishing task 9. Reset/reset has no retained policy for past tasks.

Additional learner metrics: `retention/bc_loss`, `weighted_bc_loss`, `bc_active`,
`bc_weight`, `bc_critic_gate_mean`, `bc_critic_sigma_a`, `bc_critic_sigma_s`,
`bc_effective_weight`, `buffer_size`, `source_success_fraction` (under the
`learner/retention/` W&B prefix).

```bash
squeue -u "$USER"
python scripts/vanilla_crl_gate_bc.py --mode=status --root=logs/vanilla_crl_gate_bc
ls -lt logs/vanilla_crl_gate_bc/slurm
```

To stop this experiment including future continuations:

```bash
touch logs/vanilla_crl_gate_bc/STOP
scancel --user="$USER" --name=crl_gate_bc
```

Remove that STOP file deliberately before resubmitting. Merely cancelling one
running allocation can release its `afterany` successor.

## Validation

Local CPU checks passed: five numerical/config tests, four mocked Slurm workflow
tests, and two checkpoint/status tests. They cover success filtering and ring
wrap, original goals, gate extremes/zero derivative, empty-buffer zero loss,
nonzero policy BC gradient and zero critic BC gradient, seed/flag validity,
prequeue ordering, failure cancellation, scheduler rejection, completed-task
skip, atomic checkpoint interruption, and contiguous progress. The existing
full-DCC config test also passed. Python compilation, bash syntax and diff
whitespace checks passed. No HPC job was launched from this workspace. The real
Acme/JAX network integration test and two-task Sawyer smoke are queued by the
user's submission command because the local environment lacks the full stack.

```bash
JAX_PLATFORM_NAME=cpu python tests/test_vanilla_success_bc.py
python tests/test_vanilla_job_chain.py
JAX_PLATFORM_NAME=cpu python tests/test_vanilla_checkpoint_resume.py
python tests/test_full_critic_gate_5seed.py
# On Jubail, automated by the smoke job:
JAX_PLATFORM_NAME=cpu python tests/test_vanilla_learner_integration.py
```
