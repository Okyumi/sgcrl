"""Training loop for the continual SAC + HER baseline.

All heavy imports (``reverb``, ``tensorflow``, ``acme``) live here so that
``run_continual_sac.py --help``, :mod:`sac.flags` and the unit tests stay
importable on a machine without the full training stack.

Structure mirrors ``run_continual_contrastive.py``: :func:`train_single_task`
owns one task's replay buffer / learner / environment loops, :func:`run` walks
the task sequence, checkpoints at each boundary and runs the cross-task
evaluation sweep.
"""
import os
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
import optax

from acme import environment_loop, specs, types
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib, actors
from acme.jax import networks as networks_lib, utils as jax_utils, variable_utils
from acme.utils import counting

import reverb
from reverb import rate_limiters
import tensorflow as tf
import tree

from contrastive import config as contrastive_config
from contrastive import rl_metrics
from contrastive import utils as contrastive_utils
from contrastive.continual_config import ContinualConfig
from contrastive.knowledge_pool import KnowledgePool, _pytree_zeros_like
from default import make_default_logger

from sac import checkpointing, flags as sac_flags, her, metrics as sac_metrics, tasks
from sac.learning import ContinualSACLearner
from sac.networks import (
    apply_policy_and_sample,
    apply_policy_k_sample_argmax,
    make_sac_networks,
)

try:
  import wandb
except ImportError:
  wandb = None

# W&B metric families and the payload key each is plotted against.  Without
# these declarations wandb forces every family onto one monotonic global step
# and drops the entries whose step went backwards.
#
# The first three axes come from acme: both ``EnvironmentLoop`` and the learner
# increment a ``counting.Counter`` under the key ``steps``, which ``WandbLogger``
# namespaces with the logger label -- so the axis is ``<label>/steps`` (SGD
# steps for the learner, cumulative env steps for the two loops).  The rest are
# payloads this module builds itself.
_WANDB_STEP_METRICS = (
    ('learner', 'learner/steps'),
    ('actor', 'actor/steps'),
    ('evaluator', 'evaluator/steps'),
    ('rl_metrics', 'rl_metrics/env_steps'),
    ('intra_eval', 'intra_eval/env_steps'),
    ('eval', 'eval/num_tasks_seen'),
    ('actor_reset', 'actor_reset/env_steps'),
)


def _networks_for(env_spec, obs_dim, config):
  """SAC networks for ``env_spec``: twin scalar Q, CRL-identical actor."""
  return make_sac_networks(
      env_spec, obs_dim=obs_dim,
      twin_q=config.twin_q,
      hidden_layer_sizes=config.hidden_layer_sizes,
      use_residual=config.use_residual,
      network_width=config.network_width,
      critic_depth=config.critic_depth,
      actor_depth=config.actor_depth)


class _FixedVarSource:
  """Variable source that always serves one frozen parameter set."""

  def __init__(self, params):
    self._params = params

  def get_variables(self, names):
    return [self._params for _ in names]


def evaluate_on_task(eval_env_name, eval_task_id, policy_params, q_params,
                     config, continual_cfg, seed, num_episodes, k_sample_k=0,
                     use_task_id=False):
  """Success rate of ``policy_params`` on one task over ``num_episodes``."""
  _tid, _ntasks = tasks.task_id_args(
      use_task_id, eval_task_id, continual_cfg.num_tasks)
  eval_env, eval_obs_dim = contrastive_utils.make_environment(
      eval_env_name, config.start_index, config.end_index,
      seed + eval_task_id + 9999,
      fixed_start_end=tasks.fixed_goal(eval_env_name),
      task_id=_tid, num_tasks=_ntasks)

  networks = _networks_for(
      specs.make_environment_spec(eval_env), eval_obs_dim, config)

  if k_sample_k > 0:
    # K-sample-argmax scores candidate actions by min(Q1,Q2) instead of the
    # contrastive inner product used by the CRL driver.
    eval_policy = apply_policy_k_sample_argmax(networks, k=k_sample_k)
    eval_params = (policy_params, q_params)
  else:
    eval_policy = apply_policy_and_sample(networks, eval_mode=True)
    eval_params = policy_params

  var_client = variable_utils.VariableClient(
      _FixedVarSource(eval_params), '', device='cpu')
  eval_actor = actors.GenericActor(
      actor_core_lib.batched_feed_forward_to_actor_core(eval_policy),
      jax.random.PRNGKey(seed + eval_task_id + 5000), var_client, backend='cpu')

  eval_loop = environment_loop.EnvironmentLoop(
      eval_env, eval_actor, observers=[contrastive_utils.SuccessObserver()])

  successes = 0
  for _ in range(num_episodes):
    result = eval_loop.run_episode()
    if result.get('success', 0) > 0.5:
      successes += 1
  try:
    eval_env.close()
  except Exception:  # pylint: disable=broad-except
    pass
  return successes / max(num_episodes, 1)


def _make_flatten_fn(config, her_reward_threshold, step_penalty_reward):
  """Build the ``tf.function`` that turns an episode into HER transitions.

  Goal sampling is inherited unchanged from the CRL driver: a geometric
  distribution over *future* states within the same trajectory.  The SAC
  addition is the TD signal, which must be consistent with the relabeled goal
  and is therefore recomputed here by :func:`sac.her.her_reward_and_discount`.
  """
  ops = her.tensorflow_ops()

  @tf.function
  def flatten_fn(sample):
    seq_len = tf.shape(sample.data.observation)[0]
    arange = tf.range(seq_len)
    is_future = tf.cast(arange[:, None] < arange[None], tf.float32)
    discount = config.discount ** tf.cast(
        arange[None] - arange[:, None], tf.float32)
    probs = is_future * discount
    goal_index = tf.random.categorical(
        logits=tf.math.log(probs), num_samples=1)[:, 0]
    state = sample.data.observation[:-1, :config.obs_dim]
    next_state = sample.data.observation[1:, :config.obs_dim]
    goal = sample.data.observation[:, :config.obs_dim]
    goal = contrastive_utils.obs_to_goal_2d(
        goal, start_index=config.start_index, end_index=config.end_index)
    goal = tf.gather(goal, goal_index[:-1])
    new_obs = tf.concat([state, goal], axis=1)
    new_next_obs = tf.concat([next_state, goal], axis=1)

    achieved_next = contrastive_utils.obs_to_goal_2d(
        next_state, start_index=config.start_index, end_index=config.end_index)
    # `sample.data.discount` is float64 from the env spec; her_reward_and_
    # discount casts it, so the float32 multiply below never trips TF's
    # strict dtype matching.
    her_reward, her_discount = her.her_reward_and_discount(
        achieved_next, goal, sample.data.discount[:-1],
        threshold=her_reward_threshold,
        step_penalty_reward=step_penalty_reward,
        ops=ops)

    transition = types.Transition(
        observation=new_obs, action=sample.data.action[:-1],
        reward=her_reward, discount=her_discount,
        next_observation=new_next_obs,
        extras={'next_action': sample.data.action[1:]})
    shift = tf.random.uniform((), 0, seq_len, tf.int32)
    return tree.map_structure(lambda t: tf.roll(t, shift, axis=0), transition)

  return flatten_fn


def _make_iterator(config, replay_client, flatten_fn):
  """reverb -> tf.data pipeline yielding batched HER transitions."""
  num_parallel_calls = 1

  def _make_dataset(unused):
    ds = reverb.TrajectoryDataset.from_table_signature(
        server_address=replay_client.server_address,
        table=config.replay_table_name,
        max_in_flight_samples_per_worker=100)
    ds = ds.map(flatten_fn)

    def _transpose_fn(t):
      dims = tf.range(tf.shape(tf.shape(t))[0])
      perm = tf.concat([[1, 0], dims[2:]], axis=0)
      return tf.transpose(t, perm)

    ds = ds.batch(config.batch_size, drop_remainder=True)
    ds = ds.map(lambda tr: tree.map_structure(_transpose_fn, tr))
    return ds.unbatch().unbatch()

  dataset = tf.data.Dataset.from_tensors(0).repeat()
  dataset = dataset.interleave(
      _make_dataset, cycle_length=num_parallel_calls,
      num_parallel_calls=num_parallel_calls, deterministic=False)
  dataset = dataset.batch(
      config.batch_size * config.num_sgd_steps_per_step, drop_remainder=True)

  @tf.function
  def add_info(data):
    info = reverb.SampleInfo(key=0, probability=0.0, table_size=0, priority=0.0)
    return reverb.ReplaySample(info=info, data=data)

  dataset = dataset.map(add_info, num_parallel_calls=tf.data.AUTOTUNE,
                        deterministic=False)
  return dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()


def _split_actor_head_body(theta_base, v_k):
  """Fold body deltas into the base, keep only head deltas in ``v_k``.

  ``adapt_heads_only`` (the CKA-RL default) pools only the actor output head:
  body updates are absorbed into ``theta_base`` and their pool entries zeroed,
  so the knowledge pool mixes head directions only.
  """
  out_base_leaves, out_vk_leaves = [], []
  flat_base, treedef = jax.tree_util.tree_flatten_with_path(theta_base)
  flat_vk, _ = jax.tree_util.tree_flatten_with_path(v_k)
  for (path, base_val), (_, vk_val) in zip(flat_base, flat_vk):
    path_str = '/'.join(str(p) for p in path)
    is_head = 'Normal' in path_str or 'normal' in path_str.lower()
    if is_head:
      out_base_leaves.append(base_val)
      out_vk_leaves.append(vk_val)
    else:
      out_base_leaves.append(base_val + vk_val)
      out_vk_leaves.append(jnp.zeros_like(vk_val))
  return treedef.unflatten(out_base_leaves), treedef.unflatten(out_vk_leaves)


def train_single_task(
    task_id: int,
    env_name: str,
    config: contrastive_config.ContrastiveConfig,
    continual_cfg: ContinualConfig,
    seed: int,
    theta_base: Optional[networks_lib.Params],
    pool: KnowledgePool,
    prev_q_params: Optional[networks_lib.Params],
    prev_target_q_params: Optional[networks_lib.Params],
    prev_q_optimizer_state,
    task_sequence,
    flag_values,
    q_base: Optional[networks_lib.Params] = None,
    critic_pool: Optional[KnowledgePool] = None,
):
  """Train on one task; return the continual state carried to the next task."""
  f = flag_values
  critic_mode = f.critic_mode
  np.random.seed(seed + task_id)

  # ---- environment -------------------------------------------------------
  fixed_goal = tasks.fixed_goal(env_name)
  _tid, _ntasks = tasks.task_id_args(
      f.use_task_id, task_id, continual_cfg.num_tasks)
  env, obs_dim = contrastive_utils.make_environment(
      env_name, config.start_index, config.end_index,
      seed + task_id, fixed_start_end=fixed_goal,
      task_id=_tid, num_tasks=_ntasks)

  config.obs_dim = obs_dim
  config.max_episode_steps = getattr(env, '_step_limit') + 1
  env_spec = specs.make_environment_spec(env)

  max_steps = tasks.steps_for_task(
      task_id, continual_cfg.base_steps, continual_cfg.steps_per_task)

  # Scalar (twin) Q instead of contrastive phi/psi encoders.  The actor
  # architecture is identical to CRL, so the CKA head/body split below works
  # unchanged.
  networks = _networks_for(env_spec, obs_dim, config)

  # ---- replay buffer (reverb) -------------------------------------------
  replay_table = reverb.Table(
      name=config.replay_table_name,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      max_size=config.max_replay_size // config.max_episode_steps,
      rate_limiter=rate_limiters.MinSize(
          config.min_replay_size // config.max_episode_steps),
      signature=adders_reverb.EpisodeAdder.signature(env_spec, {}))
  replay_server = reverb.Server([replay_table], port=None)
  replay_client = reverb.Client(f'localhost:{replay_server.port}')

  iterator = _make_iterator(
      config, replay_client,
      _make_flatten_fn(config, float(f.her_reward_threshold),
                       bool(f.step_penalty_reward)))

  # ---- learner -----------------------------------------------------------
  log_dir = os.path.join(
      f.log_dir, f'continual_{config.alg_name}',
      checkpointing.config_key(critic_mode, f.use_task_id, f.adapt_heads_only,
                               f.actor_mode, f.step_penalty_reward,
                               f.her_reward_threshold),
      f'task{task_id}_{env_name}_s{seed}')
  os.makedirs(log_dir, exist_ok=True)

  logger_kwargs = dict(save_data=True, save_dir=log_dir, add_uid=config.add_uid,
                       use_wandb=config.use_wandb, time_delta=10.0,
                       wandb_auto_step=True)
  learner_logger = make_default_logger(
      'learner', asynchronous=True,
      serialize_fn=jax_utils.fetch_devicearray,
      steps_key='learner_steps', **logger_kwargs)

  learner = ContinualSACLearner(
      networks=networks,
      rng=jax.random.PRNGKey(seed + task_id * 1000),
      q_optimizer=optax.adam(learning_rate=config.learning_rate, eps=1e-7),
      vk_optimizer=optax.adam(
          learning_rate=config.actor_learning_rate, eps=1e-7),
      beta_optimizer=optax.adam(learning_rate=1e-3),
      alpha_scale_optimizer=optax.adam(learning_rate=1e-3),
      iterator=iterator,
      counter=counting.Counter(),
      logger=learner_logger,
      config=config,
      continual_config=continual_cfg,
      task_id=task_id,
      theta_base=theta_base,
      pool=pool,
      prev_q_params=prev_q_params,
      prev_target_q_params=prev_target_q_params,
      prev_q_optimizer_state=prev_q_optimizer_state,
      critic_mode=critic_mode,
      adapt_heads_only=f.adapt_heads_only,
      encoder_from_base=f.encoder_from_base,
      q_base=q_base,
      critic_pool=critic_pool,
      step_penalty_reward=f.step_penalty_reward,
  )

  # ---- behaviour actor ---------------------------------------------------
  actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
      apply_policy_and_sample(networks))
  variable_client = variable_utils.VariableClient(
      learner, 'policy', device='cpu')
  adder = adders_reverb.EpisodeAdder(
      client=replay_client,
      priority_fns={config.replay_table_name: None},
      max_sequence_length=config.max_episode_steps + 1)
  actor_cls = (contrastive_utils.InitiallyRandomActor
               if config.use_random_actor else actors.GenericActor)
  actor = actor_cls(actor_core, jax.random.PRNGKey(seed + task_id + 100),
                    variable_client, adder, backend='cpu')

  def _observers():
    return [
        contrastive_utils.SuccessObserver(),
        contrastive_utils.DistanceObserver(
            obs_dim=config.obs_dim, start_index=config.start_index,
            end_index=config.end_index),
    ]

  env_loop = environment_loop.EnvironmentLoop(
      env, actor, counter=counting.Counter(),
      logger=make_default_logger('actor', steps_key='actor_steps',
                                 **logger_kwargs),
      observers=_observers())

  # ---- in-task evaluator (deterministic policy) --------------------------
  eval_variable_client = variable_utils.VariableClient(
      learner, 'policy', device='cpu')
  eval_actor = actors.GenericActor(
      actor_core_lib.batched_feed_forward_to_actor_core(
          apply_policy_and_sample(networks, eval_mode=True)),
      jax.random.PRNGKey(seed + task_id + 200), eval_variable_client,
      backend='cpu')
  eval_env, _ = contrastive_utils.make_environment(
      env_name, config.start_index, config.end_index,
      seed + task_id + 300, fixed_start_end=fixed_goal,
      task_id=_tid, num_tasks=_ntasks)
  eval_loop = environment_loop.EnvironmentLoop(
      eval_env, eval_actor, counter=counting.Counter(),
      logger=make_default_logger('evaluator', steps_key='actor_steps',
                                 **logger_kwargs),
      observers=_observers())

  # ---- prefill -----------------------------------------------------------
  first_batch = config.batch_size * config.num_sgd_steps_per_step
  prefill_steps = max(config.min_replay_size,
                      first_batch + config.max_episode_steps)
  print(f'  Prefilling replay ({prefill_steps} steps)...', flush=True)
  prefill_done, prefill_eps = 0, 0
  while prefill_done < prefill_steps:
    result = env_loop.run_episode()
    env_loop._logger.write(result)  # pylint: disable=protected-access
    prefill_done += int(result['episode_length'])
    prefill_eps += 1
  print(f'  Prefill complete ({prefill_done} steps, {prefill_eps} episodes).',
        flush=True)

  # ---- training loop -----------------------------------------------------
  env_steps_done = 0
  train_steps = max_steps - config.min_replay_size
  log_every_steps = 10000
  next_log_at = log_every_steps
  eval_every = f.eval_every
  eval_enabled = f.eval_episodes > 0 and eval_every > 0
  next_eval_at = eval_every if eval_enabled else float('inf')
  next_evaluator_at = eval_every if eval_enabled else float('inf')
  # rl_metrics cadence mirrors the CRL driver: cheap metrics every
  # `metrics_every` env steps, expensive ones (SVD, NRC) every 5x that.
  metrics_every = eval_every if eval_every > 0 else 50000
  next_metrics_frequent = metrics_every if f.log_rl_metrics else float('inf')
  next_metrics_occasional = (
      5 * metrics_every if f.log_rl_metrics else float('inf'))
  episodes_done = 0
  auto_reset_active = (task_id == 0 and f.actor_auto_reset)
  actor_reset_count = 0
  actor_reset_rng = jax.random.PRNGKey(seed + 9999)
  log_wandb = f.use_wandb and wandb is not None

  print(f'  Training for {train_steps} env steps...', flush=True)
  if auto_reset_active:
    print(f'  Actor auto-reset enabled: warmup={f.actor_reset_warmup}, '
          f'threshold={f.actor_reset_dormant_threshold}, '
          f'max_resets={f.actor_reset_max}.', flush=True)

  while env_steps_done < train_steps:
    result = env_loop.run_episode()
    env_loop._logger.write(result)  # pylint: disable=protected-access
    env_steps_done += int(result['episode_length'])
    episodes_done += 1

    if episodes_done == 1:
      print('  First learner step (includes JIT compilation)...', flush=True)
    learner.step()
    if episodes_done == 1:
      print('  JIT compilation done.', flush=True)

    if env_steps_done >= next_log_at:
      if log_wandb:
        last_metrics = getattr(learner, 'last_metrics', None)
        if last_metrics:
          payload = {f'learner/{k}': float(v)
                     for k, v in last_metrics.items()
                     if k not in ('steps', 'learner_steps', 'walltime')}
          payload['learner/env_steps'] = env_steps_done
          wandb.log(payload)
      print(f'  Task {task_id} [{env_name}]: '
            f'{env_steps_done}/{train_steps} env steps '
            f'({episodes_done} episodes)', flush=True)
      next_log_at = env_steps_done + log_every_steps

    if env_steps_done >= next_evaluator_at:
      eval_variable_client.update_and_wait()
      successes, returns = [], []
      for _ in range(f.eval_episodes):
        ep_result = eval_loop.run_episode()
        successes.append(ep_result.get('success', 0))
        returns.append(float(ep_result.get('episode_return', 0)))
      eval_success_rate = float(np.mean(successes))
      eval_mean_return = float(np.mean(returns))
      if log_wandb:
        wandb.log({'evaluator/success_rate': eval_success_rate,
                   'evaluator/mean_return': eval_mean_return,
                   'evaluator/env_steps': env_steps_done})
      print(f'  [eval @ {env_steps_done}] success={eval_success_rate:.1%} '
            f'return={eval_mean_return:.1f}', flush=True)
      next_evaluator_at = env_steps_done + eval_every

    # Representation-level metrics.  `compute_sac_metrics` reads the critic's
    # penultimate activations (pre-Dense(1)) via networks.critic_hidden_repr_fn
    # and reuses every CRL primitive under critic_q1/* and critic_q2/*.  Actor
    # metrics are identical to CRL (shared architecture).  Auto-reset
    # piggybacks on the actor/dormant_ratio computed here.
    if env_steps_done >= next_metrics_frequent:
      if env_steps_done >= next_metrics_occasional:
        level = 'occasional'
        next_metrics_occasional = env_steps_done + 5 * metrics_every
      else:
        level = 'frequent'
      next_metrics_frequent = env_steps_done + metrics_every

      try:
        transitions = learner.last_transitions
        if transitions is not None:
          bs = config.batch_size
          obs_sample = jnp.array(transitions.observation[:bs])
          act_sample = jnp.array(transitions.action[:bs])
          current_actor = learner.get_variables(['policy'])[0]
          m = sac_metrics.compute_sac_metrics(
              networks, current_actor, learner.q_params,
              obs_sample, act_sample, level=level)
          if log_wandb:
            payload = {f'rl_metrics/{k}': v for k, v in m.items()}
            payload['rl_metrics/env_steps'] = env_steps_done
            wandb.log(payload)

          if (auto_reset_active
              and actor_reset_count < f.actor_reset_max
              and env_steps_done >= f.actor_reset_warmup):
            actor_dr = m.get('actor/dormant_ratio')
            if actor_dr is None:
              # Frequent-level cycle: dormant_ratio only runs on occasional.
              actor_feats = rl_metrics.extract_actor_features(
                  networks, current_actor, obs_sample)
              if actor_feats is not None:
                actor_dr = rl_metrics.dormant_ratio(actor_feats)
            if (actor_dr is not None
                and actor_dr > f.actor_reset_dormant_threshold):
              actor_reset_rng, reset_key = jax.random.split(actor_reset_rng)
              actor_reset_count += 1
              print(f'  [auto-reset @ {env_steps_done}] actor '
                    f'dormant_ratio={actor_dr:.3f} > '
                    f'{f.actor_reset_dormant_threshold} — resetting actor '
                    f'(#{actor_reset_count}).', flush=True)
              learner.reset_actor(reset_key)
              if log_wandb:
                wandb.log({'actor_reset/triggered': 1,
                           'actor_reset/dormant_ratio_at_reset': actor_dr,
                           'actor_reset/count': actor_reset_count,
                           'actor_reset/env_steps': env_steps_done})
      except Exception as e:  # pylint: disable=broad-except
        print(f'  [rl_metrics] Warning: {e}', flush=True)

    if f.intra_eval_previous_tasks and env_steps_done >= next_eval_at:
      next_eval_at = env_steps_done + eval_every
      current_policy = learner.get_variables(['policy'])[0]
      print(f'  [intra-eval @ {env_steps_done} steps] '
            f'Evaluating tasks 0..{task_id}...', flush=True)
      intra_results = {
          task_sequence[eval_tid]: evaluate_on_task(
              task_sequence[eval_tid], eval_tid, current_policy,
              learner.q_params, config, continual_cfg, seed,
              num_episodes=f.eval_episodes, k_sample_k=f.k_sample_k,
              use_task_id=f.use_task_id)
          for eval_tid in range(task_id + 1)
      }
      intra_mean = float(np.mean(list(intra_results.values())))
      print(f'  [intra-eval] Mean success: {intra_mean:.1%}', flush=True)
      if log_wandb:
        payload = {f'intra_eval/{n}': s for n, s in intra_results.items()}
        payload['intra_eval/mean_success'] = intra_mean
        payload['intra_eval/env_steps'] = env_steps_done
        wandb.log(payload)

  print(f'  Task {task_id} training complete ({env_steps_done} env steps, '
        f'{episodes_done} episodes).', flush=True)

  # ---- extract continual state for the next task -------------------------
  composed_policy = learner.get_variables(['policy'])[0]
  v_k = learner.v_k

  if task_id == 0:
    # Nothing to mix yet: fold v_k into the base and pool a zero vector so
    # the pool length still tracks the number of tasks seen.
    out_theta_base = jax.tree_util.tree_map(
        lambda b, v: b + v, learner.theta_base, v_k)
    pool.append(_pytree_zeros_like(out_theta_base))
  elif f.adapt_heads_only:
    out_theta_base, v_k_head_only = _split_actor_head_body(
        learner.theta_base, v_k)
    head_leaves = jax.tree_util.tree_leaves(v_k_head_only)
    n_head = sum(1 for v in head_leaves if jnp.any(v != 0))
    print(f'  [pool] head params: {n_head}, '
          f'body params (zeroed): {len(head_leaves) - n_head}', flush=True)
    pool.append(v_k_head_only)
  else:
    out_theta_base = theta_base
    pool.append(v_k)
  pool.merge_if_needed()

  out_critic_pool = (critic_pool if critic_pool is not None
                     else KnowledgePool(k_max=continual_cfg.k_max))
  out_q_base = q_base
  if critic_mode == 'cka':
    if task_id == 0:
      out_q_base = learner.q_params
      out_critic_pool.append(_pytree_zeros_like(out_q_base))
    else:
      out_critic_pool.append(learner.w_k_critic)
    out_critic_pool.merge_if_needed()

  out_q_params = learner.q_params
  out_target_q_params = learner.target_q_params
  out_q_optimizer_state = learner.q_optimizer_state

  replay_server.stop()
  for closeable in (env, eval_env):
    try:
      closeable.close()
    except Exception:  # pylint: disable=broad-except
      pass
  del learner, variable_client, eval_variable_client

  return (out_theta_base, out_q_params, out_target_q_params,
          out_q_optimizer_state, pool, composed_policy,
          out_q_base, out_critic_pool)


def _init_wandb(params, task_id, env_name, num_tasks, continual_cfg, seed,
                flag_values):
  """Start a per-task W&B run and declare the per-family step metrics."""
  f = flag_values
  wandb.init(
      entity=f.wandb_entity or None,
      project=f.wandb_project,
      group=f.wandb_group,
      mode=f.wandb_mode,
      config=sac_flags.wandb_run_config(
          params, task_id, env_name, num_tasks, continual_cfg.k_max,
          flag_values=f),
      name=f'task{task_id}_{env_name}_s{seed}',
      reinit=True,
  )
  for family, axis in _WANDB_STEP_METRICS:
    wandb.define_metric(axis)
    wandb.define_metric(f'{family}/*', step_metric=axis)


def run(flag_values):
  """Walk the task sequence: train, checkpoint, evaluate on all tasks seen."""
  f = flag_values
  seed = f.seed
  task_sequence, num_tasks = sac_flags.resolve_tasks(f)
  if f.single_task:
    print(f'  [single-task mode] Training on {f.single_task} only.', flush=True)
  continual_cfg = sac_flags.build_continual_config(num_tasks, f)
  params = sac_flags.build_contrastive_params(f)

  theta_base = None
  pool = KnowledgePool(k_max=continual_cfg.k_max)
  prev_q = prev_tgt_q = prev_q_opt = q_base = None
  critic_pool = KnowledgePool(k_max=continual_cfg.k_max)

  ckpt_kwargs = dict(critic_mode=f.critic_mode, use_task_id=f.use_task_id,
                     adapt_heads_only=f.adapt_heads_only,
                     actor_mode=f.actor_mode,
                     step_penalty_reward=f.step_penalty_reward,
                     her_reward_threshold=f.her_reward_threshold)

  # ---- resume ------------------------------------------------------------
  start_task = f.start_task
  if start_task == 0 and f.auto_resume:
    resume_at = checkpointing.find_resume_task(
        f.checkpoint_dir, num_tasks, seed, **ckpt_kwargs)
    if resume_at is None:
      print('  [auto-resume] No existing checkpoints found. '
            'Starting from task 0.', flush=True)
    else:
      start_task = resume_at
      print(f'  [auto-resume] Found checkpoint for task {start_task - 1} '
            f'-> resuming from task {start_task}.', flush=True)

  if start_task > 0:
    if start_task >= num_tasks:
      print(f'  All {num_tasks} tasks already completed. Nothing to do.',
            flush=True)
      return
    ckpt = checkpointing.load_ckpt(
        f.checkpoint_dir, start_task - 1, seed, **ckpt_kwargs)
    theta_base = ckpt['theta_base']
    pool.load_state_dict(ckpt['pool_vectors'])
    prev_q = ckpt['q_params']
    prev_tgt_q = ckpt['target_q_params']
    prev_q_opt = ckpt.get('q_optimizer_state')
    if f.critic_mode == 'cka':
      q_base = ckpt.get('q_base')
      critic_pool_vecs = ckpt.get('critic_pool_vectors')
      if critic_pool_vecs is not None:
        critic_pool.load_state_dict(critic_pool_vecs)

  fresh_pool = lambda: KnowledgePool(k_max=continual_cfg.k_max)

  for task_id in range(start_task, num_tasks):
    env_name = task_sequence[task_id]
    params['env_name'] = env_name

    print(f'\n{"=" * 60}', flush=True)
    print(f'Task {task_id}/{num_tasks - 1}: {env_name}', flush=True)
    print(f'Phase: {"BASE" if task_id == 0 else "CONTINUAL"} | Steps: '
          f'{tasks.steps_for_task(task_id, continual_cfg.base_steps, continual_cfg.steps_per_task)}'
          f' | Pool: {len(pool)}/{continual_cfg.k_max}', flush=True)
    print(f'Critic: {f.critic_mode} | Task ID: {f.use_task_id} | '
          f'Heads only: {f.adapt_heads_only} | '
          f'Encoder base: {f.encoder_from_base}', flush=True)
    print(f'Actor mode: {f.actor_mode} | Eval: {f.eval_episodes}ep, '
          f'K={f.k_sample_k} | Reward: '
          f'{"step-penalty" if f.step_penalty_reward else "sparse01"} '
          f'(tau={f.her_reward_threshold})', flush=True)
    print(f'{"=" * 60}\n', flush=True)

    config = contrastive_config.ContrastiveConfig(**params)

    if f.use_wandb and wandb is not None:
      _init_wandb(params, task_id, env_name, num_tasks, continual_cfg, seed, f)

    in_theta_base, in_pool = tasks.pre_task_actor_state(
        f.actor_mode, task_id, theta_base, pool, fresh_pool)

    (theta_base, prev_q, prev_tgt_q, prev_q_opt, pool, composed_policy,
     q_base, critic_pool) = train_single_task(
        task_id=task_id,
        env_name=env_name,
        config=config,
        continual_cfg=continual_cfg,
        seed=seed,
        theta_base=in_theta_base,
        pool=in_pool,
        prev_q_params=prev_q,
        prev_target_q_params=prev_tgt_q,
        prev_q_optimizer_state=prev_q_opt,
        task_sequence=task_sequence,
        flag_values=f,
        q_base=q_base,
        critic_pool=critic_pool,
    )

    theta_base, pool = tasks.post_task_actor_state(
        f.actor_mode, theta_base, pool, composed_policy, fresh_pool)

    ckpt_data = {
        'theta_base': theta_base,
        'pool_vectors': pool.state_dict(),
        'q_params': prev_q,
        'target_q_params': prev_tgt_q,
        'q_optimizer_state': prev_q_opt,
        'composed_policy': composed_policy,
        'task_id': task_id,
        'env_name': env_name,
    }
    if f.critic_mode == 'cka':
      ckpt_data['q_base'] = q_base
      ckpt_data['critic_pool_vectors'] = critic_pool.state_dict()
    checkpointing.save_ckpt(
        f.checkpoint_dir, task_id, seed, ckpt_data, **ckpt_kwargs)

    # ---- cross-task evaluation -------------------------------------------
    if f.eval_episodes > 0:
      print('\n  Evaluating on all tasks seen so far...', flush=True)
      eval_results = {}
      for eval_tid in range(task_id + 1):
        eval_env_name = task_sequence[eval_tid]
        sr = evaluate_on_task(
            eval_env_name, eval_tid, composed_policy, prev_q, config,
            continual_cfg, seed, num_episodes=f.eval_episodes,
            k_sample_k=f.k_sample_k, use_task_id=f.use_task_id)
        eval_results[eval_env_name] = sr
        print(f'    Task {eval_tid} [{eval_env_name}]: {sr:.1%}', flush=True)
      mean_sr = float(np.mean(list(eval_results.values())))
      print(f'    Mean success: {mean_sr:.1%}', flush=True)
      if f.use_wandb and wandb is not None:
        payload = {f'eval/{name}': sr for name, sr in eval_results.items()}
        payload['eval/mean_success'] = mean_sr
        payload['eval/num_tasks_seen'] = task_id + 1
        wandb.log(payload)

    if f.use_wandb and wandb is not None:
      wandb.finish()

  print(f'\nAll {num_tasks} tasks complete.', flush=True)
