r"""Sequential continual goal-conditioned contrastive RL driver.

This is the recommended entrypoint for the continual RL experiments.
Unlike the LaunchPad-based version, this script runs each task sequentially
in a single process, making it straightforward to pass state between tasks.

For each task:
  1. Create environment, networks, replay, and learner.
  2. Collect data and train for steps_per_task environment steps.
  3. Extract θ_base (task 0), v_k, critic state, and update the pool.
  4. Save checkpoint.
  5. Proceed to next task.

Usage:
  python run_continual_contrastive.py \
      --seed=42 --num_tasks=10 --steps_per_task=1000000 \
      --alg=contrastive_cpc --k_max=5

For a quick test (2 tasks, 10k steps each):
  python run_continual_contrastive.py \
      --seed=42 --num_tasks=2 --steps_per_task=10000 --base_steps=10000 \
      --alg=contrastive_cpc --k_max=2
"""
import functools
import os
import pickle
import time
from typing import Optional

from absl import app
from absl import flags
import numpy as np

import jax
import jax.numpy as jnp
import optax

import acme
from acme import specs, types, environment_loop
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib, actors
from acme.jax import networks as networks_lib, variable_utils, utils as jax_utils
from acme.utils import counting, loggers
from acme.wrappers import gym_wrapper, step_limit

import reverb
from reverb import rate_limiters
import tensorflow as tf
import tree

import contrastive
from contrastive import config as contrastive_config
from contrastive import networks as contrastive_networks
from contrastive import utils as contrastive_utils
from contrastive.continual_config import ContinualConfig, CONTINUAL_TASK_SEQUENCE
from contrastive.continual_learning import (
    ContinualContrastiveLearner, ContinualTrainingState,
)
from contrastive.knowledge_pool import KnowledgePool, _pytree_zeros_like
from default import make_default_logger

import env_utils

# ---- flags ----------------------------------------------------------------
FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string('alg', 'contrastive_cpc', 'Algorithm variant.')
flags.DEFINE_integer('num_tasks', 10, 'Number of tasks.')
flags.DEFINE_integer('steps_per_task', 1_000_000, 'Env steps per continual task.')
flags.DEFINE_integer('base_steps', 1_000_000, 'Env steps for base task.')
flags.DEFINE_integer('k_max', 5, 'Max pool size before merging.')
flags.DEFINE_string('log_dir', 'logs/', 'Base log directory.')
flags.DEFINE_string('checkpoint_dir', 'logs/continual_checkpoints',
                    'Directory for cross-task checkpoints.')
flags.DEFINE_bool('use_wandb', False, 'Log to W&B.')
flags.DEFINE_bool('add_uid', False, 'Add UID to log dirs.')
flags.DEFINE_integer('start_task', 0, 'Resume from this task (loads ckpt from task-1).')
flags.DEFINE_integer('eval_every', 50_000, 'Evaluate every N env steps.')
flags.DEFINE_integer('time_delta_minutes', 5, 'Checkpoint frequency (minutes).')
flags.DEFINE_integer('num_actors', 1, 'Number of parallel actors (1 for sequential).')

# Fixed goals for all continual tasks
FIXED_GOALS = {
    'sawyer_hammer': np.array([0.24, 0.74, 0.11]),
    'sawyer_push_wall': np.array([0.05, 0.85, 0.015]),
    'sawyer_faucet_close': np.array([-0.14, 0.82, 0.13]),
    'sawyer_push_back': np.array([0.06, 0.62, 0.02]),
    'sawyer_stick_pull': np.array([0.41, 0.54, 0.02]),
    'sawyer_handle_press_side': np.array([-0.07, 0.68, 0.07]),
    'sawyer_push': np.array([0.02, 0.89, 0.02]),
    'sawyer_shelf_place': np.array([0.02, 0.89, 0.30]),
    'sawyer_window_close': np.array([0., 0.80, 0.2]),
    'sawyer_peg_unplug_side': np.array([0.01, 0.66, 0.13]),
}


# ---- checkpoint utilities ------------------------------------------------

def _ckpt_path(ckpt_dir, task_id, seed):
  return os.path.join(ckpt_dir, f'seed_{seed}', f'task_{task_id}.pkl')


def save_ckpt(ckpt_dir, task_id, seed, data):
  path = _ckpt_path(ckpt_dir, task_id, seed)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  # Convert JAX arrays to numpy for pickling
  data_np = jax.tree_map(
      lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
      data)
  with open(path, 'wb') as f:
    pickle.dump(data_np, f)
  print(f'  [ckpt] Saved → {path}')


def load_ckpt(ckpt_dir, task_id, seed):
  path = _ckpt_path(ckpt_dir, task_id, seed)
  with open(path, 'rb') as f:
    data = pickle.load(f)
  # Convert back to JAX arrays
  data_jax = jax.tree_map(
      lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
      data)
  print(f'  [ckpt] Loaded ← {path}')
  return data_jax


# ---- single task training loop -------------------------------------------

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
):
  """Train on a single task and return (theta_base, learner) for the next task."""

  np.random.seed(seed + task_id)

  # ---- environment -------------------------------------------------------
  fixed_goal = FIXED_GOALS[env_name]
  env, obs_dim = contrastive_utils.make_environment(
      env_name, config.start_index, config.end_index,
      seed + task_id, fixed_start_end=fixed_goal)

  config.obs_dim = obs_dim
  config.max_episode_steps = getattr(env, '_step_limit') + 1
  env_spec = specs.make_environment_spec(env)

  if task_id == 0:
    max_steps = continual_cfg.base_steps
  else:
    max_steps = continual_cfg.steps_per_task

  # ---- networks ----------------------------------------------------------
  networks = contrastive.make_networks(
      env_spec, obs_dim=obs_dim,
      repr_dim=config.repr_dim, repr_norm=config.repr_norm,
      twin_q=config.twin_q, use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes)

  # ---- replay buffer (reverb) -------------------------------------------
  samples_per_insert_tolerance = (
      config.samples_per_insert_tolerance_rate * config.samples_per_insert)
  min_replay_traj = config.min_replay_size // config.max_episode_steps
  max_replay_traj = config.max_replay_size // config.max_episode_steps
  error_buffer = min_replay_traj * samples_per_insert_tolerance

  replay_table = reverb.Table(
      name=config.replay_table_name,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      max_size=max_replay_traj,
      rate_limiter=rate_limiters.SampleToInsertRatio(
          min_size_to_sample=min_replay_traj,
          samples_per_insert=config.samples_per_insert,
          error_buffer=error_buffer),
      signature=adders_reverb.EpisodeAdder.signature(env_spec, {}))

  replay_server = reverb.Server([replay_table], port=None)
  replay_client = reverb.Client(f'localhost:{replay_server.port}')

  # ---- dataset iterator --------------------------------------------------
  @tf.function
  def flatten_fn(sample):
    seq_len = tf.shape(sample.data.observation)[0]
    arange = tf.range(seq_len)
    is_future = tf.cast(arange[:, None] < arange[None], tf.float32)
    discount = config.discount ** tf.cast(arange[None] - arange[:, None], tf.float32)
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
    transition = types.Transition(
        observation=new_obs, action=sample.data.action[:-1],
        reward=sample.data.reward[:-1], discount=sample.data.discount[:-1],
        next_observation=new_next_obs,
        extras={'next_action': sample.data.action[1:]})
    shift = tf.random.uniform((), 0, seq_len, tf.int32)
    transition = tree.map_structure(lambda t: tf.roll(t, shift, axis=0), transition)
    return transition

  num_parallel_calls = config.num_parallel_calls or tf.data.AUTOTUNE

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
    ds = ds.unbatch().unbatch()
    return ds

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
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  iterator = dataset.as_numpy_iterator()

  # Prefetch to device
  device = jax.devices()[0]
  iterator = jax_utils.prefetch(iterator, buffer_size=2, device=device)

  # ---- learner -----------------------------------------------------------
  log_dir = os.path.join(
      FLAGS.log_dir, f'continual_{config.alg_name}',
      f'task{task_id}_{env_name}_s{seed}')
  os.makedirs(log_dir, exist_ok=True)

  learner_logger = make_default_logger(
      'learner', save_data=True, save_dir=log_dir,
      add_uid=config.add_uid, use_wandb=config.use_wandb,
      time_delta=10.0, asynchronous=True,
      serialize_fn=jax_utils.fetch_devicearray,
      steps_key='learner_steps')

  rng = jax.random.PRNGKey(seed + task_id * 1000)

  q_optimizer = optax.adam(learning_rate=config.learning_rate, eps=1e-7)
  vk_optimizer = optax.adam(learning_rate=config.actor_learning_rate, eps=1e-7)
  beta_optimizer = optax.adam(learning_rate=1e-3)
  alpha_scale_optimizer = optax.adam(learning_rate=1e-3)

  learner = ContinualContrastiveLearner(
      networks=networks,
      rng=rng,
      q_optimizer=q_optimizer,
      vk_optimizer=vk_optimizer,
      beta_optimizer=beta_optimizer,
      alpha_scale_optimizer=alpha_scale_optimizer,
      iterator=iterator,
      counter=counting.Counter(),
      logger=learner_logger,
      obs_to_goal=functools.partial(
          contrastive_utils.obs_to_goal_2d,
          start_index=config.start_index,
          end_index=config.end_index),
      config=config,
      continual_config=continual_cfg,
      task_id=task_id,
      theta_base=theta_base,
      pool=pool,
      prev_q_params=prev_q_params,
      prev_target_q_params=prev_target_q_params,
      prev_q_optimizer_state=prev_q_optimizer_state,
  )

  # ---- actor (for data collection) ---------------------------------------
  policy_network = contrastive_networks.apply_policy_and_sample(networks)
  actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy_network)
  variable_client = variable_utils.VariableClient(learner, 'policy', device='cpu')

  adder = adders_reverb.EpisodeAdder(
      client=replay_client,
      priority_fns={config.replay_table_name: None},
      max_sequence_length=config.max_episode_steps + 1)

  if config.use_random_actor:
    actor = contrastive_utils.InitiallyRandomActor(
        actor_core, jax.random.PRNGKey(seed + task_id + 100),
        variable_client, adder, backend='cpu')
  else:
    actor = actors.GenericActor(
        actor_core, jax.random.PRNGKey(seed + task_id + 100),
        variable_client, adder, backend='cpu')

  # ---- observers ---------------------------------------------------------
  observers = [
      contrastive_utils.SuccessObserver(),
      contrastive_utils.DistanceObserver(
          obs_dim=config.obs_dim,
          start_index=config.start_index,
          end_index=config.end_index),
  ]

  # ---- training loop (actor-learner loop) --------------------------------
  actor_logger = make_default_logger(
      'actor', save_data=True, save_dir=log_dir,
      add_uid=config.add_uid, use_wandb=config.use_wandb,
      time_delta=10.0, steps_key='actor_steps')

  env_loop = environment_loop.EnvironmentLoop(
      env, actor, counter=counting.Counter(),
      logger=actor_logger, observers=observers)

  # Prefill replay
  print(f'  Prefilling replay ({config.min_replay_size} steps)...')
  env_loop.run(num_steps=config.min_replay_size)

  # Training
  steps_done = 0
  train_steps = max_steps - config.min_replay_size
  print(f'  Training for {train_steps} steps...')

  learner_steps_per_actor_step = 1  # simplified ratio
  while steps_done < train_steps:
    # Actor step
    env_loop.run(num_steps=1)
    steps_done += 1

    # Learner step
    learner.step()

    # Periodic logging
    if steps_done % 10000 == 0:
      print(f'  Task {task_id} [{env_name}]: {steps_done}/{train_steps} steps')

  print(f'  Task {task_id} training complete.')

  # ---- extract state for next task ---------------------------------------
  if task_id == 0:
    # After base phase: θ_base = initial_params + v_0 (fully trained policy).
    # v_0 captures the training delta.  Fold it into θ_base so that the base
    # is the *trained* policy, matching the pseudocode.
    out_theta_base = jax.tree_map(
        lambda b, v: b + v, learner.theta_base, learner.v_k)
    # Per pseudocode, initialise the pool with a zero vector (not v_0).
    # The zero vector acts as a "no-op" entry that softmax can allocate
    # weight to, providing a form of regularisation.
    pool.append(_pytree_zeros_like(out_theta_base))
  else:
    out_theta_base = theta_base  # stays frozen
    # Append the learned knowledge vector v_k to the pool.
    pool.append(learner.v_k)

  pool.merge_if_needed()

  out_q_params = learner.q_params
  out_target_q_params = learner.target_q_params
  out_q_optimizer_state = learner.q_optimizer_state

  # Cleanup
  replay_server.stop()

  return (out_theta_base, out_q_params, out_target_q_params,
          out_q_optimizer_state, pool)


# ---- main ----------------------------------------------------------------

def main(_):
  seed = FLAGS.seed
  num_tasks = min(FLAGS.num_tasks, len(CONTINUAL_TASK_SEQUENCE))

  continual_cfg = ContinualConfig(
      num_tasks=num_tasks,
      steps_per_task=FLAGS.steps_per_task,
      base_steps=FLAGS.base_steps,
      k_max=FLAGS.k_max,
      checkpoint_dir=FLAGS.checkpoint_dir,
      seed=seed,
  )

  # Shared config
  alg = FLAGS.alg
  params = {
      'seed': seed,
      'use_random_actor': True,
      'entropy_coefficient': 0.0,
      'env_name': '',
      'max_number_of_steps': 0,
      'alg_name': alg,
      'fix_goals': True,
      'add_uid': FLAGS.add_uid,
      'log_dir': FLAGS.log_dir,
      'time_delta_minutes': FLAGS.time_delta_minutes,
      'use_wandb': FLAGS.use_wandb,
  }
  if alg == 'contrastive_cpc':
    params['use_cpc'] = True
  elif alg == 'c_learning':
    params['use_td'] = True
    params['twin_q'] = True
  elif alg == 'nce+c_learning':
    params['use_td'] = True
    params['twin_q'] = True
    params['add_mc_to_td'] = True
  else:
    raise NotImplementedError(f'Unknown algorithm: {alg}')

  # State
  theta_base = None
  pool = KnowledgePool(k_max=continual_cfg.k_max)
  prev_q = None
  prev_tgt_q = None
  prev_q_opt = None

  start_task = FLAGS.start_task
  if start_task > 0:
    ckpt = load_ckpt(FLAGS.checkpoint_dir, start_task - 1, seed)
    theta_base = ckpt['theta_base']
    pool.load_state_dict(ckpt['pool_vectors'])
    prev_q = ckpt['q_params']
    prev_tgt_q = ckpt['target_q_params']
    prev_q_opt = ckpt.get('q_optimizer_state')

  for task_id in range(start_task, num_tasks):
    env_name = CONTINUAL_TASK_SEQUENCE[task_id]
    params['env_name'] = env_name

    print(f'\n{"="*60}')
    print(f'Task {task_id}/{num_tasks - 1}: {env_name}')
    phase = 'BASE' if task_id == 0 else 'CONTINUAL'
    steps = continual_cfg.base_steps if task_id == 0 else continual_cfg.steps_per_task
    print(f'Phase: {phase} | Steps: {steps} | Pool: {len(pool)}/{continual_cfg.k_max}')
    print(f'{"="*60}\n')

    config = contrastive.ContrastiveConfig(**params)

    (theta_base, prev_q, prev_tgt_q, prev_q_opt, pool) = train_single_task(
        task_id=task_id,
        env_name=env_name,
        config=config,
        continual_cfg=continual_cfg,
        seed=seed,
        theta_base=theta_base,
        pool=pool,
        prev_q_params=prev_q,
        prev_target_q_params=prev_tgt_q,
        prev_q_optimizer_state=prev_q_opt,
    )

    # Save checkpoint
    save_ckpt(FLAGS.checkpoint_dir, task_id, seed, {
        'theta_base': theta_base,
        'pool_vectors': pool.state_dict(),
        'q_params': prev_q,
        'target_q_params': prev_tgt_q,
        'q_optimizer_state': prev_q_opt,
        'task_id': task_id,
        'env_name': env_name,
    })

  print(f'\nAll {num_tasks} tasks complete.')


if __name__ == '__main__':
  app.run(main)
