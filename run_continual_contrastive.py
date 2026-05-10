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
      --seed=42 --num_tasks=10 --steps_per_task=8000000 \
      --alg=contrastive_cpc --k_max=10

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
from contrastive.continual_config import (
    ContinualConfig, CONTINUAL_TASK_SEQUENCE, CONTINUAL_TASK_SEQUENCE_20,
)
from contrastive.continual_learning import (
    ContinualContrastiveLearner, ContinualTrainingState,
)
from contrastive.continual_learning_decomposed import (
    ContinualDecomposedLearner, DecomposedTrainingState,
)
from contrastive.decomposed_networks import make_decomposed_networks
from contrastive.knowledge_pool import (
    KnowledgePool, _pytree_zeros_like, cosine_summary_from_vectors,
    cosine_matrix_from_vectors,
)
from contrastive.negative_bank import NegativeBank
from contrastive import rl_metrics
from default import make_default_logger

import env_utils

# Conditional wandb import (only needed when --use_wandb is set)
try:
  import wandb
except ImportError:
  wandb = None

# ---- flags ----------------------------------------------------------------
FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string('alg', 'contrastive_cpc', 'Algorithm variant.')
flags.DEFINE_integer('num_tasks', 10, 'Number of tasks.')
flags.DEFINE_integer('steps_per_task', 8_000_000, 'Env steps per continual task.')
flags.DEFINE_integer('base_steps', 8_000_000, 'Env steps for base task.')
flags.DEFINE_integer('k_max', 10, 'Max pool size before merging.')
flags.DEFINE_string('checkpoint_dir', 'logs/continual_checkpoints',
                    'Directory for cross-task checkpoints.')
flags.DEFINE_bool('use_wandb', True, 'Log to W&B.')
flags.DEFINE_bool('add_uid', False, 'Add UID to log dirs.')
flags.DEFINE_integer('start_task', 0, 'Resume from this task (loads ckpt from task-1).')
flags.DEFINE_integer('eval_every', 50_000, 'Evaluate every N env steps.')
flags.DEFINE_integer('time_delta_minutes', 5, 'Checkpoint frequency (minutes).')
flags.DEFINE_integer('num_actors', 1, 'Number of parallel actors (1 for sequential).')
flags.DEFINE_bool('use_task_id', False, 'Append one-hot task ID to state and goal.')
flags.DEFINE_string('critic_mode', 'persistent',
                    'Critic evolution across tasks: "persistent" (never reset, carry forward), '
                    '"reset" (reinitialize critic each task), '
                    '"cka" (CKA-RL style base+vectors for critic too).')
flags.DEFINE_integer('eval_episodes', 10,
                     'Episodes per task for cross-task evaluation (0 to disable).')
flags.DEFINE_bool('intra_eval_previous_tasks', False,
                  'During training on the current task, periodically evaluate on '
                  'all previously learned tasks. Disabled by default because it '
                  'is expensive (creates envs for every past task at each eval interval).')
flags.DEFINE_bool('log_rl_metrics', True,
                  'Log representation metrics (weight norms, feature rank, '
                  'NRC, dormant ratio, intrinsic dimension). Enabled by default.')
flags.DEFINE_integer('k_sample_k', 0,
                     'K for K-sample-argmax evaluation (0 = deterministic mean).')
flags.DEFINE_bool('adapt_heads_only', True,
                  'Only adapt actor output head layers (CKA-RL default).')
flags.DEFINE_bool('encoder_from_base', False,
                  'Freeze shared encoder from base task.')
flags.DEFINE_bool('use_20_tasks', False,
                  'Use 20-task sequence (two passes of the 10-task sequence).')
flags.DEFINE_string('actor_mode', 'cka',
                    'Actor evolution across tasks: '
                    '"cka" (CKA-RL style base+vectors, default), '
                    '"reset" (reinitialize actor each task), '
                    '"persistent" (single network, continuously trained, no decomposition).')
# Scaling architecture (Wang et al., 2025: 1000-layer GCRL)
flags.DEFINE_bool('use_residual', True,
                  'Use ResidualMLP (LayerNorm+Swish+skip) instead of plain MLP.')
flags.DEFINE_integer('network_width', 256, 'Hidden dim for ResidualMLP.')
flags.DEFINE_integer('critic_depth', 4,
                     'Dense layers in critic residual blocks (multiple of 4).')
flags.DEFINE_integer('actor_depth', 4,
                     'Dense layers in actor residual blocks (multiple of 4).')
flags.DEFINE_string('energy_fn', 'inner_product',
                    'Critic energy function: inner_product (SGCRL) or l2 (1000-layer paper).')
flags.DEFINE_float('logsumexp_penalty', 0.01,
                   'Coefficient for logsumexp regularization in CPC loss.')
flags.DEFINE_string('single_task', '',
                    'If set, train on this single environment only '
                    '(e.g., sawyer_shelf_place). Overrides task sequence.')
# Automatic actor reset during task 0 (dormancy-triggered)
flags.DEFINE_bool('actor_auto_reset', False,
                  'Monitor actor health during task 0 and automatically reset '
                  'if dormant ratio exceeds threshold.  Resets actor weights + '
                  'optimizer; critic is unchanged.  Only active during task 0 '
                  'to preserve continual ablation integrity.  When the actor '
                  'learns well, no reset ever fires.  DISABLED by default to '
                  'avoid any reset interfering with ablation experiments.')
flags.DEFINE_float('actor_reset_dormant_threshold', 0.1,
                   'Dormant ratio threshold that triggers an automatic actor '
                   'reset.  0.1 = reset when >=10%% of trunk neurons are dormant '
                   '(activation score < 2.5%% of layer mean under Swish).')
flags.DEFINE_integer('actor_reset_warmup', 200000,
                     'Minimum env steps before the first dormancy check. '
                     'Gives the actor time to stabilise after initialisation '
                     'before judging it.')
flags.DEFINE_integer('actor_reset_max', 3,
                     'Maximum number of automatic actor resets per task-0 run. '
                     'Safety cap to prevent infinite reset loops.')

# Previous-replay negative bank (offline-to-online variant)
flags.DEFINE_string('neg_bank_mode', 'off',
                    'Previous-replay negative bank mode: '
                    '"off" (default, no bank), '
                    '"vanilla" (uniform random bank goals as extra negatives), '
                    '"hard_weighted" (per-anchor top-M hard negatives, weighted).')
flags.DEFINE_integer('neg_bank_per_task_capacity', 10000,
                     'Max goals stored per task in the negative bank.  When a '
                     'task finishes, up to this many HER-relabeled goals from '
                     'that task\'s replay buffer are added to the bank.')
flags.DEFINE_integer('neg_bank_n_per_step', 256,
                     'Number of bank goals sampled per learner step.  For '
                     'hard_weighted mode, this is the FINAL number of hard '
                     'negatives per anchor (not the candidate pool size).')
flags.DEFINE_integer('neg_bank_candidate_pool', 1024,
                     'Candidate pool size drawn from the bank each step '
                     '(hard_weighted mode only).  The top M by score per '
                     'anchor are kept.  Ignored in vanilla mode.')
flags.DEFINE_float('neg_bank_weight', 0.3,
                   'Logit weight for bank negatives (0-1).  Scales the bank '
                   'logit contribution relative to in-batch negatives, '
                   'preventing over-reliance on cross-task contrasts.')
flags.DEFINE_integer('neg_bank_max_tasks', 20,
                     'Max number of tasks retained in the bank (FIFO).')

# Decomposed-critic + diagnostic flags (proposal 1, plan section 6).
# All default to the existing ContinualConfig dataclass defaults so that
# omitting any of these from a submit script preserves bit-identical
# behaviour vs prior runs. Cluster cells set them via env-var pipelines
# in the submit scripts (draft_3.sh / draft_4.sh / DRAFT.sh /
# submit_continual_torch.sh) which read from experiment_configs.py.
flags.DEFINE_float('dyn_aux_weight', 1.0,
                   'Weight on the masked-dynamics auxiliary loss (mu) when '
                   'critic_mode="decomposed". 0.0 disables L_dyn (used for '
                   'the regression-check cell N5). Plan section 6.')
flags.DEFINE_integer('phi_task_width', 256,
                     'Width of the per-task additive encoder phi_task '
                     '(critic_mode="decomposed" only). Smaller than the '
                     'shared body. Plan section 6.')
flags.DEFINE_integer('phi_task_depth', 4,
                     'Depth of the per-task additive encoder phi_task '
                     '(critic_mode="decomposed" only). Must be a positive '
                     'multiple of 4 when use_residual=True (one residual '
                     'block at depth=4). Plan section 6.')
flags.DEFINE_bool('log_pool_cosine', True,
                  'Log per-task pool cosine-similarity matrices on the '
                  'actor / critic CKA pools. Cheap host-side metric. '
                  'Plan section 3.1 / D1-D4.')
flags.DEFINE_bool('log_mixture_norm', False,
                  'Log the per-step ratio || sum_j alpha_j v_j || / || v_k || '
                  'inside the CKA inner loop. One extra norm per active CKA '
                  'path when on; bit-identical when off. Plan section 3.2 / D5.')
flags.DEFINE_bool('log_probe_data', False,
                  'At the end of each task, dump the first batch_size '
                  '(obs, action) pairs from the replay iterator to a '
                  'probe_data_task{k}_seed{s}.npz file next to the '
                  'checkpoint. Consumed by eval_linear_probe.py. Plan '
                  'section 3.4 / D6.')

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

def _ckpt_path(ckpt_dir, task_id, seed, critic_mode='persistent',
               use_task_id=True, adapt_heads_only=True, actor_mode='cka'):
  """Checkpoint path keyed by all ablation-relevant config.

  Structure: {ckpt_dir}/actor_{mode}_critic_{mode}_tid_{bool}_heads_{bool}/seed_{seed}/task_{id}.pkl
  This ensures different ablation configurations never share checkpoints.
  """
  config_key = (f'actor_{actor_mode}_critic_{critic_mode}'
                f'_tid_{use_task_id}_heads_{adapt_heads_only}')
  return os.path.join(ckpt_dir, config_key, f'seed_{seed}',
                      f'task_{task_id}.pkl')


def save_ckpt(ckpt_dir, task_id, seed, data, critic_mode='persistent',
              use_task_id=True, adapt_heads_only=True, actor_mode='cka'):
  path = _ckpt_path(ckpt_dir, task_id, seed, critic_mode, use_task_id,
                     adapt_heads_only, actor_mode)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  # Convert JAX arrays to numpy for pickling
  data_np = jax.tree_util.tree_map(
      lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
      data)
  with open(path, 'wb') as f:
    pickle.dump(data_np, f)
  print(f'  [ckpt] Saved → {path}', flush=True)


def load_ckpt(ckpt_dir, task_id, seed, critic_mode='persistent',
              use_task_id=True, adapt_heads_only=True, actor_mode='cka'):
  path = _ckpt_path(ckpt_dir, task_id, seed, critic_mode, use_task_id,
                     adapt_heads_only, actor_mode)
  if not os.path.exists(path):
    raise FileNotFoundError(
        f'No checkpoint found at {path}. Make sure the previous run used '
        f'the same configuration (seed={seed}, actor_mode={actor_mode}, '
        f'critic_mode={critic_mode}, use_task_id={use_task_id}, '
        f'adapt_heads_only={adapt_heads_only}).')
  with open(path, 'rb') as f:
    data = pickle.load(f)
  # Convert back to JAX arrays
  data_jax = jax.tree_util.tree_map(
      lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
      data)
  print(f'  [ckpt] Loaded ← {path}', flush=True)
  return data_jax


# ---- cross-task evaluation -----------------------------------------------

def evaluate_on_task(
    eval_env_name, eval_task_id, policy_params, q_params, config,
    continual_cfg, seed, num_episodes, k_sample_k=0):
  """Run num_episodes on a task and return success rate."""
  fixed_goal = FIXED_GOALS[eval_env_name]
  _tid = eval_task_id if FLAGS.use_task_id else None
  _ntasks = continual_cfg.num_tasks if FLAGS.use_task_id else None
  eval_env, eval_obs_dim = contrastive_utils.make_environment(
      eval_env_name, config.start_index, config.end_index,
      seed + eval_task_id + 9999,
      fixed_start_end=fixed_goal,
      task_id=_tid, num_tasks=_ntasks)

  env_spec = specs.make_environment_spec(eval_env)
  networks = contrastive.make_networks(
      env_spec, obs_dim=eval_obs_dim,
      repr_dim=config.repr_dim, repr_norm=config.repr_norm,
      twin_q=config.twin_q, use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes,
      use_residual=config.use_residual,
      network_width=config.network_width,
      critic_depth=config.critic_depth,
      actor_depth=config.actor_depth,
      energy_fn=config.energy_fn)

  if k_sample_k > 0:
    eval_policy = contrastive_networks.apply_policy_k_sample_argmax(
        networks, k=k_sample_k)
    eval_params = (policy_params, q_params)
  else:
    eval_policy = contrastive_networks.apply_policy_and_sample(
        networks, eval_mode=True)
    eval_params = policy_params

  eval_actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
      eval_policy)

  class _FixedVarSource:
    def __init__(self, p):
      self._p = p
    def get_variables(self, names):
      return [self._p for _ in names]

  var_client = variable_utils.VariableClient(
      _FixedVarSource(eval_params), '', device='cpu')
  eval_actor = actors.GenericActor(
      eval_actor_core, jax.random.PRNGKey(seed + eval_task_id + 5000),
      var_client, backend='cpu')

  observer = contrastive_utils.SuccessObserver()
  eval_loop = environment_loop.EnvironmentLoop(
      eval_env, eval_actor, observers=[observer])

  successes = 0
  for _ in range(num_episodes):
    result = eval_loop.run_episode()
    if result.get('success', 0) > 0.5:
      successes += 1
  try:
    eval_env.close()
  except Exception:
    pass
  return successes / max(num_episodes, 1)


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
    critic_mode: str = 'persistent',
    actor_mode: str = 'cka',
    adapt_heads_only: bool = True,
    encoder_from_base: bool = False,
    task_sequence: tuple = CONTINUAL_TASK_SEQUENCE,
    q_base: Optional[networks_lib.Params] = None,
    critic_pool: Optional[KnowledgePool] = None,
    neg_bank=None,
    # ---- decomposed-critic carry (only used when critic_mode='decomposed') ---
    prev_b_shared_params: Optional[networks_lib.Params] = None,
    prev_b_shared_opt_state=None,
    prev_h_phi_params: Optional[networks_lib.Params] = None,
    prev_h_phi_opt_state=None,
    prev_h_dyn_params: Optional[networks_lib.Params] = None,
    prev_h_dyn_opt_state=None,
    prev_psi_params: Optional[networks_lib.Params] = None,
    prev_psi_opt_state=None,
):
  """Train on a single task and return (theta_base, learner) for the next task.

  When ``critic_mode == 'decomposed'``, the actor / pool plumbing is
  bypassed and the decomposed critic state is carried through the
  ``prev_b_shared_*`` / ``prev_h_phi_*`` / ``prev_h_dyn_*`` /
  ``prev_psi_*`` arguments instead. The returned tuple's actor/pool
  fields are placeholders in that case (see the bottom of this
  function).
  """

  np.random.seed(seed + task_id)

  # ---- early guards: incompatible flags for critic_mode='decomposed' ----
  # Fail fast before booting the replay server. The decomposed learner
  # also raises on use_td/twin_q internally, but failing here keeps the
  # error attributable to a single source.
  if critic_mode == 'decomposed':
    if config.use_td:
      raise ValueError(
          "critic_mode='decomposed' requires use_td=False (no TD path).")
    if config.twin_q:
      raise ValueError(
          "critic_mode='decomposed' requires twin_q=False.")
    if config.use_image_obs:
      raise ValueError(
          "critic_mode='decomposed' does not support use_image_obs=True.")
    if config.entropy_coefficient is not None:
      raise ValueError(
          "critic_mode='decomposed' requires adaptive entropy "
          "(config.entropy_coefficient=None).")
    if FLAGS.neg_bank_mode != 'off':
      raise ValueError(
          "critic_mode='decomposed' does not support neg_bank_mode != 'off'.")
    if actor_mode == 'cka':
      raise ValueError(
          "critic_mode='decomposed' is incompatible with actor_mode='cka' "
          "(proposal 1 keeps the actor reset; use actor_mode='reset').")
    if FLAGS.k_sample_k > 0:
      raise ValueError(
          "critic_mode='decomposed' does not currently support k_sample_k>0 "
          "in cross-task evaluation; the decomposed critic exposes a "
          "5-tuple bundle, not a single q_params pytree.")

  # ---- environment -------------------------------------------------------
  # Task ID is appended to both state and goal at the gym level
  # (via TaskIDGymWrapper in env_utils.py).  Observation layout:
  #   [state_spatial, task_one_hot, goal_spatial, task_one_hot]
  # obs_dim = STATE_DIM_UNIFIED + num_tasks, so state and goal have
  # identical dimensionality.  The contrastive critic sees the task ID
  # in both φ(s,a) and ψ(g).
  fixed_goal = FIXED_GOALS[env_name]
  _tid = task_id if FLAGS.use_task_id else None
  _ntasks = continual_cfg.num_tasks if FLAGS.use_task_id else None
  env, obs_dim = contrastive_utils.make_environment(
      env_name, config.start_index, config.end_index,
      seed + task_id, fixed_start_end=fixed_goal,
      task_id=_tid, num_tasks=_ntasks)

  config.obs_dim = obs_dim
  config.max_episode_steps = getattr(env, '_step_limit') + 1
  env_spec = specs.make_environment_spec(env)

  if task_id == 0:
    max_steps = continual_cfg.base_steps
  else:
    max_steps = continual_cfg.steps_per_task

  # ---- networks ----------------------------------------------------------
  # The actor is always built from `make_networks` (this gives us the
  # policy_network + sample / log_prob heads). When critic_mode is
  # 'decomposed' we additionally build the four-component decomposed
  # critic and pass it to the sibling learner; when not, the existing
  # q_network inside `networks` is used by ContinualContrastiveLearner.
  networks = contrastive.make_networks(
      env_spec, obs_dim=obs_dim,
      repr_dim=config.repr_dim, repr_norm=config.repr_norm,
      twin_q=config.twin_q, use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes,
      use_residual=config.use_residual,
      network_width=config.network_width,
      critic_depth=config.critic_depth,
      actor_depth=config.actor_depth,
      energy_fn=config.energy_fn)

  decomp_nets = None
  if critic_mode == 'decomposed':
    decomp_nets = make_decomposed_networks(
        env_spec, obs_dim=obs_dim,
        repr_dim=config.repr_dim,
        use_residual=config.use_residual,
        network_width=config.network_width,
        critic_depth=config.critic_depth,
        phi_task_width=getattr(continual_cfg, 'phi_task_width', 256),
        phi_task_depth=getattr(continual_cfg, 'phi_task_depth', 4),
        energy_fn=config.energy_fn,
        repr_norm=config.repr_norm,
    )

  # ---- replay buffer (reverb) -------------------------------------------
  # A fresh replay buffer is created per task so that experience from
  # previous tasks does not leak into the current task's training data.
  min_replay_traj = config.min_replay_size // config.max_episode_steps
  max_replay_traj = config.max_replay_size // config.max_episode_steps

  replay_table = reverb.Table(
      name=config.replay_table_name,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      max_size=max_replay_traj,
      # IMPORTANT (sequential continual runner):
      # During prefill we only insert and do not sample yet. Using
      # SampleToInsertRatio can block inserts after ~min_size_to_sample
      # episodes, causing prefill to hang. MinSize avoids this deadlock.
      rate_limiter=rate_limiters.MinSize(min_replay_traj),
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

  # Use a single interleave worker to avoid deadlocks with drop_remainder
  # batching during early sampling when the replay buffer is small.
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
  # No jax_utils.prefetch here: background device prefetching during the
  # replay prefill phase (before the learner starts consuming) causes
  # backpressure deadlocks.

  # ---- learner -----------------------------------------------------------
  config_tag = (f'actor_{FLAGS.actor_mode}_critic_{critic_mode}'
                f'_tid_{FLAGS.use_task_id}_heads_{FLAGS.adapt_heads_only}')
  log_dir = os.path.join(
      FLAGS.log_dir, f'continual_{config.alg_name}', config_tag,
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

  if critic_mode == 'decomposed':
    # Sibling learner: shares the actor with `make_networks` (we hand it
    # the policy_network + sample / log_prob fns) but maintains its own
    # 4-component critic + h_dyn head. CKA / persistent / reset paths
    # are untouched.
    learner = ContinualDecomposedLearner(
        decomp_nets=decomp_nets,
        policy_network=networks.policy_network,
        sample_fn=networks.sample,
        log_prob_fn=networks.log_prob,
        rng=rng,
        iterator=iterator,
        counter=counting.Counter(),
        logger=learner_logger,
        config=config,
        continual_config=continual_cfg,
        task_id=task_id,
        prev_b_shared_params=prev_b_shared_params,
        prev_b_shared_opt_state=prev_b_shared_opt_state,
        prev_h_phi_params=prev_h_phi_params,
        prev_h_phi_opt_state=prev_h_phi_opt_state,
        prev_h_dyn_params=prev_h_dyn_params,
        prev_h_dyn_opt_state=prev_h_dyn_opt_state,
        prev_psi_params=prev_psi_params,
        prev_psi_opt_state=prev_psi_opt_state,
    )
  else:
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
        critic_mode=critic_mode,
        actor_mode=actor_mode,
        adapt_heads_only=adapt_heads_only,
        encoder_from_base=encoder_from_base,
        q_base=q_base,
        critic_pool=critic_pool,
        neg_bank_mode=FLAGS.neg_bank_mode,
        neg_bank_n_per_step=FLAGS.neg_bank_n_per_step,
        neg_bank_weight=FLAGS.neg_bank_weight,
        neg_bank_hard_ratio=(FLAGS.neg_bank_candidate_pool // max(FLAGS.neg_bank_n_per_step, 1)),
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

  # ---- evaluator (deterministic policy) ----------------------------------
  eval_policy_network = contrastive_networks.apply_policy_and_sample(
      networks, eval_mode=True)
  eval_actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
      eval_policy_network)
  eval_variable_client = variable_utils.VariableClient(
      learner, 'policy', device='cpu')
  eval_actor = actors.GenericActor(
      eval_actor_core, jax.random.PRNGKey(seed + task_id + 200),
      eval_variable_client, backend='cpu')  # no adder — eval only

  eval_env, _ = contrastive_utils.make_environment(
      env_name, config.start_index, config.end_index,
      seed + task_id + 300, fixed_start_end=fixed_goal,
      task_id=_tid, num_tasks=_ntasks)
  eval_observers = [
      contrastive_utils.SuccessObserver(),
      contrastive_utils.DistanceObserver(
          obs_dim=config.obs_dim,
          start_index=config.start_index,
          end_index=config.end_index),
  ]
  evaluator_logger = make_default_logger(
      'evaluator', save_data=True, save_dir=log_dir,
      add_uid=config.add_uid, use_wandb=config.use_wandb,
      time_delta=10.0, steps_key='actor_steps')
  eval_loop = environment_loop.EnvironmentLoop(
      eval_env, eval_actor, counter=counting.Counter(),
      logger=evaluator_logger, observers=eval_observers)

  # ---- training loop (actor-learner loop) --------------------------------
  actor_logger = make_default_logger(
      'actor', save_data=True, save_dir=log_dir,
      add_uid=config.add_uid, use_wandb=config.use_wandb,
      time_delta=10.0, steps_key='actor_steps')

  env_loop = environment_loop.EnvironmentLoop(
      env, actor, counter=counting.Counter(),
      logger=actor_logger, observers=observers)

  # Prefill replay buffer.  We need enough data for the first learner
  # batch (batch_size * num_sgd_steps_per_step transitions) plus one
  # episode buffer, otherwise `next(iterator)` blocks and the
  # single-process actor-learner loop deadlocks.
  first_batch = config.batch_size * config.num_sgd_steps_per_step
  prefill_steps = max(config.min_replay_size,
                      first_batch + config.max_episode_steps)
  print(f'  Prefilling replay ({prefill_steps} steps)...', flush=True)
  prefill_done = 0
  prefill_eps = 0
  while prefill_done < prefill_steps:
    result = env_loop.run_episode()
    env_loop._logger.write(result)  # pylint: disable=protected-access
    prefill_done += int(result['episode_length'])
    prefill_eps += 1
  print(f'  Prefill complete ({prefill_done} steps, '
        f'{prefill_eps} episodes).', flush=True)

  # Training
  env_steps_done = 0
  train_steps = max_steps - config.min_replay_size
  log_every_steps = 10000  # print progress every N env steps
  next_log_at = log_every_steps
  eval_every = FLAGS.eval_every
  next_eval_at = eval_every if (FLAGS.eval_episodes > 0 and eval_every > 0) else float('inf')
  next_evaluator_at = eval_every if (FLAGS.eval_episodes > 0 and eval_every > 0) else float('inf')
  episodes_done = 0
  # Metric logging schedule: frequent (1x), occasional (5x)
  metrics_every = eval_every if eval_every > 0 else 50000
  next_metrics_frequent = metrics_every if FLAGS.log_rl_metrics else float('inf')
  next_metrics_occasional = 5 * metrics_every if FLAGS.log_rl_metrics else float('inf')
  # Automatic actor reset state (task 0 only)
  auto_reset_active = (task_id == 0 and FLAGS.actor_auto_reset)
  actor_reset_count = 0
  actor_reset_rng = jax.random.PRNGKey(seed + 9999)  # separate RNG stream

  # Negative-bank sampling (task > 0 only; empty bank at task 0 by design)
  bank_rng = np.random.default_rng(seed + 77777 + task_id * 31)
  bank_sample_size = (
      FLAGS.neg_bank_candidate_pool
      if FLAGS.neg_bank_mode == 'hard_weighted'
      else FLAGS.neg_bank_n_per_step)
  use_bank_this_task = (
      FLAGS.neg_bank_mode != 'off'
      and FLAGS.neg_bank_n_per_step > 0
      and task_id > 0
      and neg_bank is not None
      and neg_bank.size() > 0)

  print(f'  Training for {train_steps} env steps...', flush=True)
  if auto_reset_active:
    print(f'  Actor auto-reset enabled: warmup={FLAGS.actor_reset_warmup}, '
          f'threshold={FLAGS.actor_reset_dormant_threshold}, '
          f'max_resets={FLAGS.actor_reset_max}.', flush=True)
  if use_bank_this_task:
    print(f'  Negative bank enabled: mode={FLAGS.neg_bank_mode}, '
          f'M={FLAGS.neg_bank_n_per_step}, weight={FLAGS.neg_bank_weight}, '
          f'bank_size={neg_bank.size()} goals from '
          f'{neg_bank.num_tasks()} previous tasks.', flush=True)

  while env_steps_done < train_steps:
    # Actor step: run one full episode and count actual env steps.
    # NOTE: Acme's `EnvironmentLoop.run()` returns None (it only writes logs),
    # so we call `run_episode()` to get the per-episode metrics dict.
    result = env_loop.run_episode()
    # Mirror `EnvironmentLoop.run()` behavior: write the episode log.
    env_loop._logger.write(result)  # pylint: disable=protected-access
    episode_steps = int(result['episode_length'])
    env_steps_done += episode_steps
    episodes_done += 1

    # Sample bank negatives for this learner step.  Shape must be constant
    # across calls to avoid JIT recompilation.
    if use_bank_this_task:
      bank_sample = neg_bank.sample(n=bank_sample_size, rng=bank_rng)
      if bank_sample is not None:
        learner.set_bank_goals(jnp.asarray(bank_sample))

    # Learner step (first call triggers JAX JIT compilation, may be slow)
    if episodes_done == 1:
      print(f'  First learner step (includes JIT compilation)...', flush=True)
    learner.step()
    if episodes_done == 1:
      print(f'  JIT compilation done.', flush=True)

    # Log learner metrics to W&B with global env_steps as x-axis
    if FLAGS.use_wandb and wandb is not None and env_steps_done >= next_log_at:
      try:
        last_metrics = learner.last_metrics
        if last_metrics:
          wandb_learner = {f'learner/{k}': float(v)
                          for k, v in last_metrics.items()
                          if k not in ('steps', 'learner_steps', 'walltime')}
          wandb_learner['learner/env_steps'] = env_steps_done
          wandb.log(wandb_learner)
      except (AttributeError, Exception):
        pass  # learner may not have last_metrics yet

    # Periodic progress logging (to stdout, independent of TimeFilter)
    if env_steps_done >= next_log_at:
      print(f'  Task {task_id} [{env_name}]: '
            f'{env_steps_done}/{train_steps} env steps '
            f'({episodes_done} episodes)', flush=True)
      next_log_at = env_steps_done + log_every_steps

    # Periodic evaluation (deterministic policy on current task)
    if env_steps_done >= next_evaluator_at:
      eval_variable_client.update_and_wait()
      eval_successes = []
      eval_returns = []
      for _ in range(FLAGS.eval_episodes):
        ep_result = eval_loop.run_episode()
        eval_successes.append(ep_result.get('success', 0))
        eval_returns.append(float(ep_result.get('episode_return', 0)))
      eval_success_rate = np.mean(eval_successes)
      eval_mean_return = np.mean(eval_returns)
      if FLAGS.use_wandb and wandb is not None:
        wandb.log({
            'evaluator/success_rate': eval_success_rate,
            'evaluator/mean_return': eval_mean_return,
            'evaluator/env_steps': env_steps_done,
        })
      print(f'  [eval @ {env_steps_done}] success={eval_success_rate:.1%} '
            f'return={eval_mean_return:.1f}', flush=True)
      next_evaluator_at = env_steps_done + eval_every

    # ---- RL representation metrics ----------------------------------------
    if env_steps_done >= next_metrics_frequent:
      if env_steps_done >= next_metrics_occasional:
        level = 'occasional'
        next_metrics_occasional = env_steps_done + 5 * metrics_every
        next_metrics_frequent = env_steps_done + metrics_every
      else:
        level = 'frequent'
        next_metrics_frequent = env_steps_done + metrics_every

      try:
        transitions = learner.last_transitions
        if transitions is not None:
          current_actor = learner.get_variables(['policy'])[0]
          current_critic = learner.q_params
          # Use the last preprocessed batch from the learner (already has
          # goal relabeling applied). The batch has shape [B*N, ...] where
          # B=batch_size and N=num_sgd_steps_per_step. Take the first B.
          bs = config.batch_size
          obs_sample = jnp.array(transitions.observation[:bs])
          act_sample = jnp.array(transitions.action[:bs])
          m = rl_metrics.compute_all_metrics(
              networks, current_actor, current_critic,
              obs_sample, act_sample, obs_dim=obs_dim, level=level)
          if FLAGS.use_wandb and wandb is not None:
            wandb_m = {f'rl_metrics/{k}': v for k, v in m.items()}
            wandb_m['rl_metrics/env_steps'] = env_steps_done
            wandb.log(wandb_m)
          # ---- Automatic actor reset (dormancy-triggered, task 0 only) ----
          if (auto_reset_active
              and actor_reset_count < FLAGS.actor_reset_max
              and env_steps_done >= FLAGS.actor_reset_warmup):
            # Compute actor dormant ratio (cheap: one forward pass + mean).
            # Use the already-extracted actor features when available;
            # otherwise compute them on the fly.
            actor_dr = m.get('actor/dormant_ratio')
            if actor_dr is None:
              # Occasional-level metrics weren't computed this cycle;
              # compute dormant ratio directly.
              actor_feats = rl_metrics.extract_actor_features(
                  networks, current_actor, obs_sample)
              if actor_feats is not None:
                actor_dr = rl_metrics.dormant_ratio(actor_feats)
            if actor_dr is not None and actor_dr > FLAGS.actor_reset_dormant_threshold:
              actor_reset_rng, reset_key = jax.random.split(actor_reset_rng)
              print(f'  [auto-reset @ {env_steps_done}] '
                    f'actor dormant_ratio={actor_dr:.3f} > '
                    f'{FLAGS.actor_reset_dormant_threshold} — '
                    f'resetting actor (#{actor_reset_count + 1}).',
                    flush=True)
              learner.reset_actor(reset_key)
              actor_reset_count += 1
              if FLAGS.use_wandb and wandb is not None:
                wandb.log({
                    'actor_reset/triggered': 1,
                    'actor_reset/dormant_ratio_at_reset': actor_dr,
                    'actor_reset/count': actor_reset_count,
                    'actor_reset/env_steps': env_steps_done,
                })

      except Exception as e:
        print(f'  [rl_metrics] Warning: {e}', flush=True)

    # Intra-task periodic evaluation on all tasks seen so far
    if FLAGS.intra_eval_previous_tasks and env_steps_done >= next_eval_at:
      next_eval_at = env_steps_done + eval_every
      current_policy = learner.get_variables(['policy'])[0]
      current_q = learner.q_params
      print(f'  [intra-eval @ {env_steps_done} steps] '
            f'Evaluating tasks 0..{task_id}...', flush=True)
      intra_results = {}
      for eval_tid in range(task_id + 1):
        eval_env_i = task_sequence[eval_tid]
        sr = evaluate_on_task(
            eval_env_i, eval_tid, current_policy, current_q, config,
            continual_cfg, seed,
            num_episodes=FLAGS.eval_episodes,
            k_sample_k=FLAGS.k_sample_k)
        intra_results[eval_env_i] = sr
      intra_mean = np.mean(list(intra_results.values()))
      print(f'  [intra-eval] Mean success: {intra_mean:.1%}', flush=True)
      if FLAGS.use_wandb and wandb is not None:
        wandb_intra = {f'intra_eval/{n}': s for n, s in intra_results.items()}
        wandb_intra['intra_eval/mean_success'] = intra_mean
        wandb_intra['intra_eval/env_steps'] = env_steps_done
        wandb.log(wandb_intra)

  print(f'  Task {task_id} training complete '
        f'({env_steps_done} env steps, {episodes_done} episodes).', flush=True)

  # ---- snapshot composed policy for cross-task evaluation ----------------
  # Must happen before pool extraction which changes the composition.
  composed_policy = learner.get_variables(['policy'])[0]

  # ---- D6a: per-task probe data dump (linear-probe diagnostic) -----------
  # Saves a small (obs, action) sample alongside the checkpoint for the
  # `eval_linear_probe.py` task-classifier diagnostic. Off by default;
  # gated on continual_cfg.log_probe_data so existing runs are
  # bit-identical. The sample is drawn from the SAME iterator the
  # learner has been consuming (via flatten_fn), so it carries the same
  # HER goal-relabeling and the same obs layout (state||goal) the
  # learner saw during training. We take the first config.batch_size
  # rows of one batch (256 by default).
  def _dump_probe_data():
    if not getattr(continual_cfg, 'log_probe_data', False):
      return
    try:
      sample = next(iterator)
      tr = types.Transition(*sample.data)
      obs_np = np.asarray(tr.observation[:config.batch_size])
      act_np = np.asarray(tr.action[:config.batch_size])
      probe_path = os.path.join(
          os.path.dirname(_ckpt_path(
              FLAGS.checkpoint_dir, task_id, seed, critic_mode,
              FLAGS.use_task_id, adapt_heads_only, actor_mode)),
          f'probe_data_task{task_id}_seed{seed}.npz',
      )
      os.makedirs(os.path.dirname(probe_path), exist_ok=True)
      np.savez(probe_path,
               obs=obs_np, action=act_np,
               task_id=np.int32(task_id),
               obs_dim=np.int32(config.obs_dim))
      print(f'  [probe] Saved {obs_np.shape[0]} (obs, action) pairs to {probe_path}',
            flush=True)
    except Exception as e:
      print(f'  [probe] Warning: probe-data dump failed: {e}', flush=True)

  _dump_probe_data()

  # ---- extract state for next task ---------------------------------------
  if critic_mode == 'decomposed':
    # The decomposed actor is reset every task: there is no v_k pool to
    # extract, no theta_base to fold, no critic CKA pool to update. The
    # only carry is the four shared critic groups (b_shared / h_phi /
    # h_dyn / psi) plus their optimiser states. phi_task and the actor
    # are reinitialised inside ContinualDecomposedLearner at the next
    # task's construction.
    out_theta_base = None
    out_q_params = None
    out_target_q_params = None
    out_q_optimizer_state = None
    out_q_base = None
    out_critic_pool = critic_pool if critic_pool is not None else KnowledgePool(
        k_max=continual_cfg.k_max)
    out_b_shared_params = learner.b_shared_params
    out_b_shared_opt_state = learner.b_shared_opt_state
    out_h_phi_params = learner.h_phi_params
    out_h_phi_opt_state = learner.h_phi_opt_state
    out_h_dyn_params = learner.h_dyn_params
    out_h_dyn_opt_state = learner.h_dyn_opt_state
    out_psi_params = learner.psi_params
    out_psi_opt_state = learner.psi_opt_state
    print(
        f'  [decomposed] Carrying b_shared / h_phi / h_dyn / psi to task '
        f'{task_id + 1}; phi_task and actor reinitialised next task.',
        flush=True,
    )

    # ---- skip neg-bank goal extraction (guarded above; bank is off) ---
    task_goals_for_bank = None

    # Cleanup
    replay_server.stop()
    try:
      env.close()
    except Exception:
      pass
    try:
      eval_env.close()
    except Exception:
      pass
    del learner, variable_client, eval_variable_client

    return (
        out_theta_base, out_q_params, out_target_q_params,
        out_q_optimizer_state, pool, composed_policy,
        out_q_base, out_critic_pool, task_goals_for_bank,
        # Decomposed carry (None for non-decomposed paths).
        out_b_shared_params, out_b_shared_opt_state,
        out_h_phi_params, out_h_phi_opt_state,
        out_h_dyn_params, out_h_dyn_opt_state,
        out_psi_params, out_psi_opt_state,
    )

  v_k = learner.v_k

  if task_id == 0:
    # After base phase: θ_base = initial_params + v_0 (fully trained policy).
    # v_0 captures the training delta.  Fold it into θ_base so that the base
    # is the *trained* policy, matching the pseudocode.
    out_theta_base = jax.tree_util.tree_map(
        lambda b, v: b + v, learner.theta_base, v_k)
    # Fix C: do NOT seed the pool with a zero vector. The CKA pool is
    # genuinely empty at task 0 (the base IS the task-0 knowledge).
    # The new CKAPool's masked-softmax handles the empty case correctly
    # without any placeholder; a zero placeholder would permanently
    # dilute alpha contributions for all subsequent tasks (Bug 4 in the
    # audit).
  elif adapt_heads_only:
    # CKA-RL style: body is fine-tuned but NOT decomposed.
    # - Fold body portion of v_k into theta_base (encoder evolves)
    # - Store only head portion of v_k in the pool (CKA decomposition)
    from contrastive.networks import is_actor_head_path

    def _split_head_body(base_val, vk_val, path):
      path_str = '/'.join(str(p) for p in path)
      # Haiku flattens module paths into top-level keys like
      # 'Normal/linear'. DictKey('Normal/linear') stringifies as
      # "['Normal/linear']". Head detection is centralised in
      # contrastive.networks.is_actor_head_path (Fix E).
      if is_actor_head_path(path_str):
        return base_val, vk_val  # head: base unchanged, v_k goes to pool
      else:
        return base_val + vk_val, jnp.zeros_like(vk_val)  # body: fold into base

    out_base_leaves, out_vk_leaves = [], []
    flat_base, treedef = jax.tree_util.tree_flatten_with_path(learner.theta_base)
    flat_vk, _ = jax.tree_util.tree_flatten_with_path(v_k)
    for (path, b), (_, v) in zip(flat_base, flat_vk):
      new_b, new_v = _split_head_body(b, v, path)
      out_base_leaves.append(new_b)
      out_vk_leaves.append(new_v)
    out_theta_base = treedef.unflatten(out_base_leaves)
    v_k_head_only = treedef.unflatten(out_vk_leaves)
    # Diagnostic: count how many params went to head vs body
    n_head = sum(1 for v in out_vk_leaves if jnp.any(v != 0))
    n_body = sum(1 for v in out_vk_leaves if not jnp.any(v != 0))
    print(f'  [pool] head params: {n_head}, body params (zeroed): {n_body}',
          flush=True)
    pool.append(v_k_head_only)
  else:
    # Full-policy adaptation: theta_base stays frozen, full v_k goes to pool
    out_theta_base = theta_base
    pool.append(v_k)

  pool.merge_if_needed()

  # ---- CKA diagnostics: actor-pool cosine similarity --------------------
  # Cheap host-side metric (one matmul per pool). Off by default; enable
  # via continual_cfg.log_pool_cosine. See docs/2026-05-08_plan_proposal1_dyn_aux.md
  # section 3.1 / section 9 for the diagnostic experiment that consumes
  # this metric.
  if getattr(continual_cfg, 'log_pool_cosine', False):
    actor_summary = cosine_summary_from_vectors(pool.get_vectors())
    print(
        f'  [pool/cosine actor] task={task_id} '
        f"n_active={int(actor_summary['n_active'])} "
        f"mean_offdiag={actor_summary['mean_offdiag']:.4f} "
        f"max_offdiag={actor_summary['max_offdiag']:.4f} "
        f"min_offdiag={actor_summary['min_offdiag']:.4f}",
        flush=True,
    )
    if FLAGS.use_wandb and wandb is not None:
      wandb.log({
          f'pool_cosine_actor/{k}': v
          for k, v in actor_summary.items()
      }, step=env_steps_done)
    # Also persist the full matrix for paper figures.
    actor_mat = cosine_matrix_from_vectors(pool.get_vectors())
    if actor_mat.shape[0] > 0:
      mat_path = os.path.join(
          os.path.dirname(_ckpt_path(
              FLAGS.checkpoint_dir, task_id, seed, critic_mode,
              FLAGS.use_task_id, adapt_heads_only, actor_mode)),
          f'pool_cosine_actor_task{task_id}.npy',
      )
      os.makedirs(os.path.dirname(mat_path), exist_ok=True)
      np.save(mat_path, np.asarray(actor_mat))

  out_q_params = learner.q_params
  out_target_q_params = learner.target_q_params
  out_q_optimizer_state = learner.q_optimizer_state

  # Critic CKA: extract w_k and update critic pool
  out_q_base = q_base
  out_critic_pool = critic_pool if critic_pool is not None else KnowledgePool(
      k_max=continual_cfg.k_max)
  if critic_mode == 'cka':
    if task_id == 0:
      # After base phase: q_base is the trained critic. Fix C: do NOT
      # seed the critic pool with a zero vector — the empty pool is the
      # correct task-0 state and the masked-softmax in the new CKA path
      # handles it without a placeholder (see Bug 4 in the audit).
      out_q_base = out_q_params
    else:
      # Extract w_k_critic from the post-training CKA state.
      out_critic_pool.append(learner.w_k_critic)
      out_critic_pool.merge_if_needed()

    # ---- CKA diagnostics: critic-pool cosine similarity ---------------
    if getattr(continual_cfg, 'log_pool_cosine', False):
      critic_summary = cosine_summary_from_vectors(
          out_critic_pool.get_vectors())
      print(
          f'  [pool/cosine critic] task={task_id} '
          f"n_active={int(critic_summary['n_active'])} "
          f"mean_offdiag={critic_summary['mean_offdiag']:.4f} "
          f"max_offdiag={critic_summary['max_offdiag']:.4f} "
          f"min_offdiag={critic_summary['min_offdiag']:.4f}",
          flush=True,
      )
      if FLAGS.use_wandb and wandb is not None:
        wandb.log({
            f'pool_cosine_critic/{k}': v
            for k, v in critic_summary.items()
        }, step=env_steps_done)
      critic_mat = cosine_matrix_from_vectors(out_critic_pool.get_vectors())
      if critic_mat.shape[0] > 0:
        mat_path = os.path.join(
            os.path.dirname(_ckpt_path(
                FLAGS.checkpoint_dir, task_id, seed, critic_mode,
                FLAGS.use_task_id, adapt_heads_only, actor_mode)),
            f'pool_cosine_critic_task{task_id}.npy',
        )
        os.makedirs(os.path.dirname(mat_path), exist_ok=True)
        np.save(mat_path, np.asarray(critic_mat))

  # ---- Extract goals for the negative bank (BEFORE stopping the server) -
  # Draws a batch from the current task's replay buffer via the iterator,
  # which applies flatten_fn to produce HER-relabeled (obs, goal) pairs.
  # We keep only the goal portion (last obs_dim columns of observation).
  task_goals_for_bank = None
  if FLAGS.neg_bank_mode != 'off' and neg_bank is not None:
    try:
      n_target = FLAGS.neg_bank_per_task_capacity
      batches_needed = max(1, n_target // (config.batch_size
                                           * config.num_sgd_steps_per_step) + 1)
      goal_chunks = []
      for _ in range(batches_needed):
        s = next(iterator)
        t = types.Transition(*s.data)
        # t.observation: [B, obs_dim + goal_dim] -> take last goal_dim cols
        g = np.asarray(t.observation[:, config.obs_dim:])
        goal_chunks.append(g)
        if sum(c.shape[0] for c in goal_chunks) >= n_target:
          break
      task_goals_for_bank = np.concatenate(goal_chunks, axis=0)[:n_target]
      print(f'  [neg_bank] Extracted {task_goals_for_bank.shape[0]} goals '
            f'from task {task_id} replay buffer.', flush=True)
    except Exception as e:
      print(f'  [neg_bank] Warning: goal extraction failed: {e}', flush=True)
      task_goals_for_bank = None

  # Cleanup — release resources to avoid leaking Mujoco contexts
  replay_server.stop()
  try:
    env.close()
  except Exception:
    pass
  try:
    eval_env.close()
  except Exception:
    pass
  del learner, variable_client, eval_variable_client

  return (
      out_theta_base, out_q_params, out_target_q_params,
      out_q_optimizer_state, pool, composed_policy,
      out_q_base, out_critic_pool, task_goals_for_bank,
      # Decomposed carry slots: None on this branch (non-decomposed critic).
      None, None,  # b_shared params / opt_state
      None, None,  # h_phi params / opt_state
      None, None,  # h_dyn params / opt_state
      None, None,  # psi params / opt_state
  )


# ---- main ----------------------------------------------------------------

def main(_):
  seed = FLAGS.seed

  # Select task sequence
  if FLAGS.single_task:
    # Single-task mode: override sequence with just one environment
    task_sequence = (FLAGS.single_task,)
    num_tasks = 1
    print(f'  [single-task mode] Training on {FLAGS.single_task} only.',
          flush=True)
  elif FLAGS.use_20_tasks:
    task_sequence = CONTINUAL_TASK_SEQUENCE_20
    num_tasks = min(FLAGS.num_tasks, len(task_sequence))
  else:
    task_sequence = CONTINUAL_TASK_SEQUENCE
    num_tasks = min(FLAGS.num_tasks, len(task_sequence))

  continual_cfg = ContinualConfig(
      num_tasks=num_tasks,
      steps_per_task=FLAGS.steps_per_task,
      base_steps=FLAGS.base_steps,
      k_max=FLAGS.k_max,
      checkpoint_dir=FLAGS.checkpoint_dir,
      seed=seed,
      # Decomposed-critic + diagnostic flags (default to dataclass
      # defaults; submit scripts override per cell via
      # experiment_configs.py).
      dyn_aux_weight=FLAGS.dyn_aux_weight,
      phi_task_width=FLAGS.phi_task_width,
      phi_task_depth=FLAGS.phi_task_depth,
      log_pool_cosine=FLAGS.log_pool_cosine,
      log_mixture_norm=FLAGS.log_mixture_norm,
      log_probe_data=FLAGS.log_probe_data,
  )

  # Shared config
  alg = FLAGS.alg
  params = {
      'seed': seed,
      'use_random_actor': True,
      # entropy_coefficient=None enables adaptive (learned) alpha, matching
      # the scaling-crl study.  The SAC dual-gradient descent automatically
      # tunes alpha toward the target_entropy.  Previously this was 0.0
      # which disabled entropy entirely, contributing to high inter-seed
      # variance (bad actor inits couldn't recover through exploration).
      'entropy_coefficient': None,
      # target_entropy = -0.5 * action_dim.  For MetaWorld Sawyer (action_dim=4)
      # this equals -2.0, matching the scaling-crl convention.  The standard
      # SAC heuristic is -action_dim = -4; the 0.5 factor is less aggressive
      # and works well with contrastive critics.
      'target_entropy': -2.0,
      'env_name': '',
      'max_number_of_steps': 0,
      'alg_name': alg,
      'fix_goals': True,
      'add_uid': FLAGS.add_uid,
      'log_dir': FLAGS.log_dir,
      'time_delta_minutes': FLAGS.time_delta_minutes,
      'use_wandb': FLAGS.use_wandb,
      'use_residual': FLAGS.use_residual,
      'network_width': FLAGS.network_width,
      'critic_depth': FLAGS.critic_depth,
      'actor_depth': FLAGS.actor_depth,
      'energy_fn': FLAGS.energy_fn,
      'logsumexp_penalty': FLAGS.logsumexp_penalty,
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
  q_base = None  # frozen critic base (critic_mode='cka')
  critic_pool = KnowledgePool(k_max=continual_cfg.k_max)

  # Decomposed-critic carry (only used when --critic_mode=decomposed).
  prev_b_shared_params = None
  prev_b_shared_opt_state = None
  prev_h_phi_params = None
  prev_h_phi_opt_state = None
  prev_h_dyn_params = None
  prev_h_dyn_opt_state = None
  prev_psi_params = None
  prev_psi_opt_state = None

  # Previous-replay negative bank (offline-to-online variant)
  # goal_dim = config.obs_dim (state and goal have identical dimensionality
  # after the TaskIDGymWrapper is applied).
  neg_bank = None
  if FLAGS.neg_bank_mode != 'off':
    # Defer goal_dim determination until we know config.obs_dim.
    # At this point config hasn't been built yet; we'll pass None goal_dim
    # and create the bank lazily below.
    pass

  # ---- determine starting task (auto-resume) ----------------------------
  # If --start_task is explicitly set (> 0), use that.
  # If --start_task=0 (default), scan for existing checkpoints with the
  # same config and automatically resume from the latest completed task.
  start_task = FLAGS.start_task
  if start_task == 0:
    # Auto-resume: find the highest task_id with an existing checkpoint
    for probe_tid in range(num_tasks - 1, -1, -1):
      probe_path = _ckpt_path(
          FLAGS.checkpoint_dir, probe_tid, seed,
          critic_mode=FLAGS.critic_mode,
          use_task_id=FLAGS.use_task_id,
          adapt_heads_only=FLAGS.adapt_heads_only,
          actor_mode=FLAGS.actor_mode)
      if os.path.exists(probe_path):
        start_task = probe_tid + 1  # resume from the NEXT task
        print(f'  [auto-resume] Found checkpoint for task {probe_tid} '
              f'→ resuming from task {start_task}.', flush=True)
        break
    if start_task == 0:
      print(f'  [auto-resume] No existing checkpoints found. '
            f'Starting from task 0.', flush=True)

  if start_task > 0:
    if start_task >= num_tasks:
      print(f'  All {num_tasks} tasks already completed. Nothing to do.',
            flush=True)
      return
    ckpt = load_ckpt(FLAGS.checkpoint_dir, start_task - 1, seed,
                      critic_mode=FLAGS.critic_mode,
                      use_task_id=FLAGS.use_task_id,
                      adapt_heads_only=FLAGS.adapt_heads_only,
                      actor_mode=FLAGS.actor_mode)
    theta_base = ckpt['theta_base']
    pool.load_state_dict(ckpt['pool_vectors'])
    prev_q = ckpt['q_params']
    prev_tgt_q = ckpt['target_q_params']
    prev_q_opt = ckpt.get('q_optimizer_state')
    if FLAGS.critic_mode == 'cka':
      q_base = ckpt.get('q_base')
      critic_pool_vecs = ckpt.get('critic_pool_vectors')
      if critic_pool_vecs is not None:
        critic_pool.load_state_dict(critic_pool_vecs)
    if FLAGS.critic_mode == 'decomposed':
      prev_b_shared_params = ckpt.get('decomposed_b_shared_params')
      prev_b_shared_opt_state = ckpt.get('decomposed_b_shared_opt_state')
      prev_h_phi_params = ckpt.get('decomposed_h_phi_params')
      prev_h_phi_opt_state = ckpt.get('decomposed_h_phi_opt_state')
      prev_h_dyn_params = ckpt.get('decomposed_h_dyn_params')
      prev_h_dyn_opt_state = ckpt.get('decomposed_h_dyn_opt_state')
      prev_psi_params = ckpt.get('decomposed_psi_params')
      prev_psi_opt_state = ckpt.get('decomposed_psi_opt_state')

  for task_id in range(start_task, num_tasks):
    env_name = task_sequence[task_id]
    params['env_name'] = env_name

    print(f'\n{"="*60}', flush=True)
    print(f'Task {task_id}/{num_tasks - 1}: {env_name}', flush=True)
    phase = 'BASE' if task_id == 0 else 'CONTINUAL'
    steps = continual_cfg.base_steps if task_id == 0 else continual_cfg.steps_per_task
    print(f'Phase: {phase} | Steps: {steps} | Pool: {len(pool)}/{continual_cfg.k_max}', flush=True)
    print(f'Critic: {FLAGS.critic_mode} | Task ID: {FLAGS.use_task_id} | '
          f'Heads only: {FLAGS.adapt_heads_only} | '
          f'Encoder base: {FLAGS.encoder_from_base}', flush=True)
    print(f'Actor mode: {FLAGS.actor_mode} | '
          f'Eval: {FLAGS.eval_episodes}ep, K={FLAGS.k_sample_k} | '
          f'20-task: {FLAGS.use_20_tasks}', flush=True)
    print(f'{"="*60}\n', flush=True)

    config = contrastive.ContrastiveConfig(**params)

    # Initialise W&B run per task (matching lp_continual_contrastive.py).
    # WandbLogger in default.py assumes wandb.init() has already been called;
    # without this call all wandb.log() silently fail.
    if FLAGS.use_wandb and wandb is not None:
      wandb.init(
          project='continual_gcrl_paper',
          # group="C0: CKA-failure diagnostic",
          # group="C1: decomposed regression check and baseline",
          group="C2: decomposed single-cell sanity",
          config={**params, 'task_id': task_id, 'env_name': env_name,
                  'num_tasks': num_tasks, 'k_max': continual_cfg.k_max,
                  'critic_mode': FLAGS.critic_mode,
                  'use_task_id': FLAGS.use_task_id,
                  'adapt_heads_only': FLAGS.adapt_heads_only,
                  'encoder_from_base': FLAGS.encoder_from_base,
                  'use_20_tasks': FLAGS.use_20_tasks,
                  'actor_mode': FLAGS.actor_mode,
                  'eval_episodes': FLAGS.eval_episodes,
                  'intra_eval_previous_tasks': FLAGS.intra_eval_previous_tasks,
                  'log_rl_metrics': FLAGS.log_rl_metrics,
                  'k_sample_k': FLAGS.k_sample_k,
                  'actor_auto_reset': FLAGS.actor_auto_reset,
                  'actor_reset_dormant_threshold': FLAGS.actor_reset_dormant_threshold,
                  'actor_reset_warmup': FLAGS.actor_reset_warmup,
                  'actor_reset_max': FLAGS.actor_reset_max,
                  'neg_bank_mode': FLAGS.neg_bank_mode,
                  'neg_bank_per_task_capacity': FLAGS.neg_bank_per_task_capacity,
                  'neg_bank_n_per_step': FLAGS.neg_bank_n_per_step,
                  'neg_bank_candidate_pool': FLAGS.neg_bank_candidate_pool,
                  'neg_bank_weight': FLAGS.neg_bank_weight,
                  'neg_bank_max_tasks': FLAGS.neg_bank_max_tasks},
          name=f'task{task_id}_{env_name}_s{seed}',
          reinit=True,
      )

    # Actor mode branching before each task
    if FLAGS.actor_mode == 'reset' and task_id > 0:
      # Reset: each task trains a fresh policy independently
      _theta_base = None
      _pool = KnowledgePool(k_max=continual_cfg.k_max)
    elif FLAGS.actor_mode == 'persistent' and task_id > 0:
      # Persistent: carry forward composed policy, no decomposition
      # theta_base was set to composed_policy after previous task
      _theta_base = theta_base
      _pool = KnowledgePool(k_max=continual_cfg.k_max)  # empty pool
    else:
      # CKA (default) or task_id == 0
      _theta_base = theta_base
      _pool = pool

    # Lazily create the negative bank once config.obs_dim is known.
    if FLAGS.neg_bank_mode != 'off' and neg_bank is None:
      neg_bank = NegativeBank(
          goal_dim=config.obs_dim,
          per_task_capacity=FLAGS.neg_bank_per_task_capacity,
          max_tasks=FLAGS.neg_bank_max_tasks)
      print(f'  [neg_bank] Created: goal_dim={config.obs_dim}, '
            f'per_task_capacity={FLAGS.neg_bank_per_task_capacity}, '
            f'max_tasks={FLAGS.neg_bank_max_tasks}.', flush=True)

    (theta_base, prev_q, prev_tgt_q, prev_q_opt, pool,
     composed_policy, q_base, critic_pool,
     task_goals_for_bank,
     prev_b_shared_params, prev_b_shared_opt_state,
     prev_h_phi_params, prev_h_phi_opt_state,
     prev_h_dyn_params, prev_h_dyn_opt_state,
     prev_psi_params, prev_psi_opt_state) = train_single_task(
        task_id=task_id,
        env_name=env_name,
        config=config,
        continual_cfg=continual_cfg,
        seed=seed,
        theta_base=_theta_base,
        pool=_pool,
        prev_q_params=prev_q,
        prev_target_q_params=prev_tgt_q,
        prev_q_optimizer_state=prev_q_opt,
        critic_mode=FLAGS.critic_mode,
        actor_mode=FLAGS.actor_mode,
        adapt_heads_only=FLAGS.adapt_heads_only,
        encoder_from_base=FLAGS.encoder_from_base,
        task_sequence=task_sequence,
        q_base=q_base,
        critic_pool=critic_pool,
        neg_bank=neg_bank,
        # Decomposed-critic carry (None for non-decomposed paths).
        prev_b_shared_params=prev_b_shared_params,
        prev_b_shared_opt_state=prev_b_shared_opt_state,
        prev_h_phi_params=prev_h_phi_params,
        prev_h_phi_opt_state=prev_h_phi_opt_state,
        prev_h_dyn_params=prev_h_dyn_params,
        prev_h_dyn_opt_state=prev_h_dyn_opt_state,
        prev_psi_params=prev_psi_params,
        prev_psi_opt_state=prev_psi_opt_state,
    )

    # Post-task: add this task's goals to the negative bank.
    if neg_bank is not None and task_goals_for_bank is not None:
      neg_bank.add_task(task_id=task_id, goals=task_goals_for_bank)
      print(f'  [neg_bank] Bank now: {neg_bank.size()} goals across '
            f'{neg_bank.num_tasks()} tasks.', flush=True)

    # Post-task actor state management
    if FLAGS.actor_mode == 'reset':
      # Discard actor state, keep only critic
      theta_base = None
      pool = KnowledgePool(k_max=continual_cfg.k_max)
    elif FLAGS.actor_mode == 'persistent':
      # Fold v_k into theta_base: carry forward composed policy
      theta_base = composed_policy
      pool = KnowledgePool(k_max=continual_cfg.k_max)  # empty pool

    # Save checkpoint
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
    if FLAGS.critic_mode == 'cka':
      ckpt_data['q_base'] = q_base
      ckpt_data['critic_pool_vectors'] = critic_pool.state_dict()
    if FLAGS.critic_mode == 'decomposed':
      ckpt_data['decomposed_b_shared_params'] = prev_b_shared_params
      ckpt_data['decomposed_b_shared_opt_state'] = prev_b_shared_opt_state
      ckpt_data['decomposed_h_phi_params'] = prev_h_phi_params
      ckpt_data['decomposed_h_phi_opt_state'] = prev_h_phi_opt_state
      ckpt_data['decomposed_h_dyn_params'] = prev_h_dyn_params
      ckpt_data['decomposed_h_dyn_opt_state'] = prev_h_dyn_opt_state
      ckpt_data['decomposed_psi_params'] = prev_psi_params
      ckpt_data['decomposed_psi_opt_state'] = prev_psi_opt_state
    save_ckpt(FLAGS.checkpoint_dir, task_id, seed, ckpt_data,
              critic_mode=FLAGS.critic_mode, use_task_id=FLAGS.use_task_id,
              adapt_heads_only=FLAGS.adapt_heads_only,
              actor_mode=FLAGS.actor_mode)

    # ---- cross-task evaluation (forgetting measurement) ------------------
    if FLAGS.eval_episodes > 0:
      print(f'\n  Evaluating on all tasks seen so far...', flush=True)
      eval_results = {}
      for eval_tid in range(task_id + 1):
        eval_env_name_i = task_sequence[eval_tid]
        sr = evaluate_on_task(
            eval_env_name_i, eval_tid, composed_policy, prev_q, config,
            continual_cfg, seed,
            num_episodes=FLAGS.eval_episodes,
            k_sample_k=FLAGS.k_sample_k)
        eval_results[eval_env_name_i] = sr
        print(f'    Task {eval_tid} [{eval_env_name_i}]: {sr:.1%}', flush=True)
      mean_sr = np.mean(list(eval_results.values()))
      print(f'    Mean success: {mean_sr:.1%}', flush=True)
      if FLAGS.use_wandb and wandb is not None:
        wandb_eval = {f'eval/{name}': sr for name, sr in eval_results.items()}
        wandb_eval['eval/mean_success'] = mean_sr
        wandb_eval['eval/num_tasks_seen'] = task_id + 1
        wandb.log(wandb_eval)

    # Close the W&B run for this task before starting the next one
    if FLAGS.use_wandb and wandb is not None:
      wandb.finish()

  print(f'\nAll {num_tasks} tasks complete.', flush=True)


if __name__ == '__main__':
  app.run(main)
