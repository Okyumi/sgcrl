"""Flag definitions and config builders for ``run_continual_sac.py``.

Imports only ``absl`` and this package's light modules, so ``--help`` works
and the builders are unit-testable without ``reverb`` / ``tensorflow`` /
``acme`` installed.  Everything heavy lives in :mod:`sac.training`.

Flags whose names and defaults match ``run_continual_contrastive.py`` keep the
same meaning there and here; the SAC-only additions are grouped at the bottom
of each section and noted in the help text.
"""
from typing import Any, Dict, Tuple

from absl import flags
from absl import logging as _logging  # Defines --log_dir, read below.

from contrastive.continual_config import ContinualConfig
from sac import tasks

FLAGS = flags.FLAGS

# -- Run identity -----------------------------------------------------------
flags.DEFINE_integer('seed', 42, 'Random seed.')
# SAC has no CPC / C-learning / NCE variants, so this is a single fixed tag.
# It is still a flag because it names the log/checkpoint subtree.
flags.DEFINE_string('alg', 'sac_her',
                    'Algorithm tag used in log paths and W&B config.')

# -- Task sequence ----------------------------------------------------------
flags.DEFINE_integer('num_tasks', 10, 'Number of tasks to train on.')
flags.DEFINE_bool('use_20_tasks', False,
                  'Use the 20-task sequence (two passes of the 10-task '
                  'CKA-RL sequence).')
flags.DEFINE_string('single_task', '',
                    'If set, train on this one environment only '
                    '(num_tasks becomes 1).')
# SAC-only: explicit sequence, handy for smoke tests and for two-task cells.
flags.DEFINE_string('task_sequence', '',
                    'Comma-separated environment names, in order. Overrides '
                    '--use_20_tasks; ignored when --single_task is set. '
                    'Names must exist in sac.tasks.FIXED_GOALS.')

# -- Step budget ------------------------------------------------------------
flags.DEFINE_integer('steps_per_task', 8_000_000,
                     'Env steps per continual task (tasks 1..N-1).')
flags.DEFINE_integer('base_steps', 8_000_000, 'Env steps for the base task 0.')

# -- Transfer modes ---------------------------------------------------------
flags.DEFINE_enum('actor_mode', 'cka', list(tasks.ACTOR_MODES),
                  'Actor evolution across tasks. "cka": theta_base + '
                  'sum_j alpha_j v_j + v_k. "reset": fresh actor each task. '
                  '"persistent": previous composed policy becomes the new '
                  'base, no mixture.')
flags.DEFINE_enum('critic_mode', 'persistent', list(tasks.CRITIC_MODES),
                  'Critic evolution across tasks: carry the critic over '
                  '("persistent"), reinitialise it ("reset"), or apply the '
                  'same CKA decomposition as the actor ("cka").')
flags.DEFINE_integer('k_max', 10,
                     'Max knowledge-pool size before vectors are merged.')
flags.DEFINE_bool('adapt_heads_only', True,
                  'Only actor output-head layers contribute to the pooled '
                  'v_k; body deltas are folded into theta_base (CKA-RL '
                  'default).')
flags.DEFINE_bool('encoder_from_base', False,
                  'Freeze the shared encoder from the base task.')
flags.DEFINE_bool('use_task_id', False,
                  'Append a one-hot task ID to both state and goal.')

# -- Reward / HER -----------------------------------------------------------
# SAC-only: the contrastive critic never reads the reward, so the CRL driver
# has no analogue of these two flags.  See sac/her.py for the exact rule.
flags.DEFINE_float('her_reward_threshold', 0.05,
                   'Goal-reach radius tau for the sparse HER reward: the '
                   'goal counts as reached iff '
                   '||achieved_goal(s_{t+1}) - g_relabeled|| < tau. 0.05 '
                   'matches sawyer_push / window_close / faucet_close; '
                   'looser tasks (stick_pull 0.12, hammer 0.07) may want '
                   'a larger value.')
flags.DEFINE_bool('step_penalty_reward', True,
                  'If True (default), reward is 0 on goal reach and -1 '
                  'otherwise (step penalty). If False, +1 on goal reach and '
                  '0 otherwise. Either way the discount is zeroed on '
                  'goal-reaching transitions (terminal bootstrap). This flag '
                  'is part of the checkpoint path key.')

# -- Network architecture ---------------------------------------------------
flags.DEFINE_bool('use_residual', True,
                  'Use ResidualMLP (LayerNorm+Swish+skip) instead of a plain '
                  'MLP (Wang et al., 2025).')
flags.DEFINE_integer('network_width', 256, 'Hidden dim for ResidualMLP.')
flags.DEFINE_integer('critic_depth', 4,
                     'Dense layers in the critic residual blocks (multiple '
                     'of 4).')
flags.DEFINE_integer('actor_depth', 4,
                     'Dense layers in the actor residual blocks (multiple '
                     'of 4).')

# -- Evaluation -------------------------------------------------------------
flags.DEFINE_integer('eval_every', 50_000,
                     'Run the in-task evaluator every N env steps. Also sets '
                     'the rl_metrics cadence.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Episodes per task for evaluation (0 disables both the '
                     'in-task evaluator and the cross-task sweep).')
flags.DEFINE_bool('intra_eval_previous_tasks', False,
                  'Also evaluate on all previous tasks during training, not '
                  'only at task boundaries.')
flags.DEFINE_integer('k_sample_k', 0,
                     'K for K-sample-argmax evaluation, scored by '
                     'min(Q1,Q2). 0 uses the deterministic policy mean.')
flags.DEFINE_bool('log_rl_metrics', True,
                  'Log representation-level metrics (entropy, gini, rank, '
                  'NRC1/NRC2, dormant ratio) for the actor trunk and both Q '
                  'heads.')

# -- Actor auto-reset (task 0 only) -----------------------------------------
flags.DEFINE_bool('actor_auto_reset', False,
                  'Monitor actor dormancy during task 0 and reset the actor '
                  'when it degrades. Disabled by default.')
flags.DEFINE_float('actor_reset_dormant_threshold', 0.1,
                   'Dormant ratio above which an auto reset triggers.')
flags.DEFINE_integer('actor_reset_warmup', 200_000,
                     'Minimum env steps before the first dormancy check.')
flags.DEFINE_integer('actor_reset_max', 3,
                     'Maximum automatic actor resets per task-0 run.')

# -- Checkpointing / resume -------------------------------------------------
flags.DEFINE_string('checkpoint_dir', 'logs/continual_sac_checkpoints',
                    'Root directory for cross-task checkpoints.')
flags.DEFINE_integer('start_task', 0,
                     'Task index to start at; loads the checkpoint for '
                     'start_task-1. 0 means "let --auto_resume decide".')
# SAC-only: makes the archive's implicit auto-resume switchable, which is what
# a fresh sweep in a reused checkpoint_dir needs.
flags.DEFINE_bool('auto_resume', True,
                  'When --start_task=0, probe the checkpoint directory '
                  'backwards and resume after the newest checkpoint matching '
                  'this exact config. Set --noauto_resume to always start at '
                  'task 0.')
flags.DEFINE_integer('time_delta_minutes', 5, 'Logger flush period (minutes).')
flags.DEFINE_bool('add_uid', False, 'Append a UID to log directory names.')

# -- W&B --------------------------------------------------------------------
# The collaborator's script hard-coded entity/project/group; they are flags
# here so no account details live in the repository.
flags.DEFINE_bool('use_wandb', True, 'Log to W&B.')
flags.DEFINE_string('wandb_project', 'continual_sac', 'W&B project.')
flags.DEFINE_string('wandb_entity', '',
                    'W&B entity (team or user). Empty uses the entity from '
                    'the local W&B login / WANDB_ENTITY.')
flags.DEFINE_string('wandb_group', 'sac_baseline',
                    'W&B group; use one group per sweep cell.')
flags.DEFINE_enum('wandb_mode', 'online', ['online', 'offline', 'disabled'],
                  'W&B mode. "offline" writes to wandb/ for later '
                  '`wandb sync`; "disabled" drops all logging.')

# -- Misc -------------------------------------------------------------------
flags.DEFINE_integer('num_actors', 1,
                     'Parallel actors (1 = sequential driver; kept for '
                     'parity with the CRL driver).')


def resolve_tasks(flag_values=None) -> Tuple[Tuple[str, ...], int]:
  """Resolve and validate the task sequence from flags."""
  f = flag_values or FLAGS
  sequence, num_tasks = tasks.resolve_task_sequence(
      num_tasks=f.num_tasks,
      single_task=f.single_task,
      use_20_tasks=f.use_20_tasks,
      task_sequence=f.task_sequence)
  tasks.validate_task_sequence(sequence[:num_tasks])
  return sequence, num_tasks


def build_continual_config(num_tasks: int, flag_values=None) -> ContinualConfig:
  """Build the :class:`ContinualConfig` for this run."""
  f = flag_values or FLAGS
  return ContinualConfig(
      num_tasks=num_tasks,
      steps_per_task=f.steps_per_task,
      base_steps=f.base_steps,
      k_max=f.k_max,
      checkpoint_dir=f.checkpoint_dir,
      seed=f.seed,
  )


def build_contrastive_params(flag_values=None) -> Dict[str, Any]:
  """Keyword arguments for ``ContrastiveConfig`` (``env_name`` filled per task).

  SAC always runs with adaptive alpha (``entropy_coefficient=None``) and twin
  Q.  ``target_entropy=-2.0`` is ``-0.5 * |A|`` for the 4-D Sawyer action
  space, the brax/JaxGCRL convention.  Contrastive-only fields (``energy_fn``,
  ``logsumexp_penalty``, ``repr_dim``, ...) are left at their dataclass
  defaults because the SAC learner never reads them.
  """
  f = flag_values or FLAGS
  return {
      'seed': f.seed,
      'use_random_actor': True,
      'entropy_coefficient': None,   # adaptive alpha
      'target_entropy': -2.0,        # -0.5 * action_dim for Sawyer
      'env_name': '',                # set per task by the driver
      'max_number_of_steps': 0,
      'alg_name': f.alg,
      'fix_goals': True,
      'add_uid': f.add_uid,
      'log_dir': f.log_dir,
      'time_delta_minutes': f.time_delta_minutes,
      'use_wandb': f.use_wandb,
      'use_residual': f.use_residual,
      'network_width': f.network_width,
      'critic_depth': f.critic_depth,
      'actor_depth': f.actor_depth,
      'twin_q': True,
  }


def wandb_run_config(params: Dict[str, Any], task_id: int, env_name: str,
                     num_tasks: int, k_max: int,
                     flag_values=None) -> Dict[str, Any]:
  """Config payload recorded on the W&B run for one task."""
  f = flag_values or FLAGS
  ablation_flags = (
      'critic_mode', 'use_task_id', 'adapt_heads_only', 'encoder_from_base',
      'use_20_tasks', 'actor_mode', 'eval_episodes',
      'intra_eval_previous_tasks', 'k_sample_k', 'actor_auto_reset',
      'actor_reset_dormant_threshold', 'actor_reset_warmup',
      'actor_reset_max', 'her_reward_threshold', 'step_penalty_reward',
      'task_sequence',
  )
  config = {**params, 'task_id': task_id, 'env_name': env_name,
            'num_tasks': num_tasks, 'k_max': k_max}
  config.update({name: getattr(f, name) for name in ablation_flags})
  return config
