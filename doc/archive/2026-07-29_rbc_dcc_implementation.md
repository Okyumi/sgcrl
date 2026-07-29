# RBC-DCC implementation

## Scope

`critic_mode=rbc_decomposed` is a new opt-in path. The existing
`decomposed`, `persistent`, `reset`, and `cka` learners and their state types
remain separate.

RBC-DCC reuses the DCC representation:

$$
z_{\mathrm{shared}}=h_\phi(b_{\mathrm{shared}}(s,a)),\qquad
z_{\mathrm{task}}=\phi_{\mathrm{task}}(s,a),\qquad
z_g=\psi(g),
$$

and the matched contrastive score

$$
f_C(s,a,g)=(z_{\mathrm{shared}}+z_{\mathrm{task}})^\top z_g.
$$

The twin scalar values are

$$
Q_i(s,a,g)=
\operatorname{softplus}(\rho_i) f_C(s,a,g)+b_i+
\Delta_i(z_{\mathrm{shared}},z_{\mathrm{task}},z_g).
$$

The two residual heads are independent and their final layers are initialized
to zero. Calibration slopes initialize to one and biases to zero.

## Gradient routing

For the Bellman and actor paths, transferred encoder **parameters** are
stop-gradient copies. This blocks TD updates to `b_shared`, `h_phi`, and `psi`
without stopping the derivative of the score with respect to the action.
Stopping the whole score tensor would incorrectly remove the contrastive base
from actor action selection.

`phi_task` receives one optimizer update from the combined InfoNCE and Bellman
gradient. The dynamics gradient is added only to `b_shared`; `h_dyn` receives
only the dynamics gradient.

## HER and Bellman semantics

RBC-DCC calls `sac.her.her_reward_and_discount`, the same canonical helper as
standalone SAC:

- Future-state geometric goal relabeling.
- Achieved goal from the next state.
- Strict Euclidean reach threshold.
- Default threshold `0.05`.
- Default step-penalty reward `-1/0`.
- Bootstrap discount zero at goal reach and at environment termination.

The target is

$$
y=r+\gamma d_{\mathrm{HER}}
\left[\min_i\bar Q_i(s',a',g)-\alpha\log\pi(a'|s',g)\right].
$$

Target copies include the shared, task, goal, residual, and calibration
parameters and are Polyak-updated.

## Task boundaries

Carried across tasks:

- `b_shared`, `h_phi`, `h_dyn`, and `psi`.
- Their optimizer states.

Reset at every task:

- `phi_task`.
- Twin residual heads and calibration parameters.
- All target parameters.
- Actor, actor optimizer, entropy temperature, and replay.

The first implementation supports only `actor_mode=reset`,
`combine_mode=add`, and `goal_encoder_mode=shared`.

## Checkpoints and logging

RBC checkpoint paths include a deterministic fingerprint of every RBC-defining
setting, including HER threshold and reward shape. Each RBC task log directory
contains `resolved_config.json`.

Standalone SAC checkpoints now include `her_reward_threshold` in their path.
Legacy SAC paths without a threshold are detected and reported as ambiguous
instead of being loaded silently.

The former `reward_pos_rate` metric is replaced by `her_success_rate`, which is
correct for both `-1/0` and `0/+1` reward shapes.

## Validation status

Dependency-light source and checkpoint tests can run outside the training
environment. Network, gradient-routing, JIT update, replay, checkpoint-resume,
and two-task smoke tests must be run in the pinned `contrastive_rl`
environment containing JAX, Haiku, Acme, Reverb, TensorFlow, and Meta-World.

Recommended validation commands:

```bash
pytest -q tests/test_sac_*.py tests/test_rbc_*.py

python run_continual_sac.py \
  --task_sequence=sawyer_push,sawyer_window_close \
  --num_tasks=2 --base_steps=20000 --steps_per_task=20000 \
  --actor_mode=reset --critic_mode=reset --nouse_wandb

python run_continual_contrastive.py \
  --critic_mode=rbc_decomposed --actor_mode=reset \
  --num_tasks=2 --base_steps=20000 --steps_per_task=20000 \
  --combine_mode=add --goal_encoder_mode=shared \
  --her_reward_threshold=0.05 --step_penalty_reward \
  --nouse_wandb --eval_episodes=1

python run_continual_contrastive.py \
  --critic_mode=decomposed --actor_mode=reset \
  --num_tasks=2 --base_steps=20000 --steps_per_task=20000 \
  --combine_mode=add --goal_encoder_mode=shared \
  --nouse_wandb --eval_episodes=1
```
