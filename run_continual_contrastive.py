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
import json
import os
import pickle
import subprocess
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
from contrastive.continual_learning_rbc import ContinualRBCDecomposedLearner
from contrastive.continual_learning_dcc_sac import ContinualDCCSACLearner
from contrastive.decomposed_networks import make_decomposed_networks
from contrastive.rbc_networks import make_rbc_networks
from contrastive import rbc_checkpointing
from contrastive import intrajectory
from contrastive import action_ranking_diagnostics
from contrastive import counterfactual_ranking
from contrastive import goal_semantics
from contrastive import phase_gated_control
from contrastive import outcome_credit
from contrastive import task58_reevaluation
from contrastive.knowledge_pool import (
    KnowledgePool, _pytree_zeros_like, cosine_summary_from_vectors,
    cosine_matrix_from_vectors,
)
from contrastive.negative_bank import NegativeBank
from contrastive import rl_metrics
from default import make_default_logger
from sac import her as sac_her
from sac import networks as sac_networks

import env_utils

# Conditional wandb import (only needed when --use_wandb is set)
try:
  import wandb
except ImportError:
  wandb = None

# DCC-family modes with separate parameter groups and a reset actor.
_HYBRID_CRITIC_MODES = (
    'dcc_sac', 'dcc_sac_separate', 'action_dcc', 'action_dcc_sac')
_Q_HYBRID_CRITIC_MODES = (
    'dcc_sac', 'dcc_sac_separate', 'action_dcc_sac')
_PLAIN_DCC_MODES = (
    'decomposed', 'iwr_decomposed', 'advantage_decomposed',
    'bridge_decomposed')
_ACTION_EFFECT_MODES = ('advantage_decomposed', 'bridge_decomposed')
_IWR_MODES = ('iwr_decomposed', 'bridge_decomposed')
_DECOMPOSED_CRITIC_MODES = (
    _PLAIN_DCC_MODES + ('rbc_decomposed',) + _HYBRID_CRITIC_MODES)
_HER_CRITIC_MODES = ('rbc_decomposed',) + _Q_HYBRID_CRITIC_MODES

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
flags.DEFINE_string(
    'resume_checkpoint_dir', '',
    'Optional source directory for the checkpoint loaded before '
    '--start_task. New checkpoints are still written to --checkpoint_dir. '
    'Empty means read and write the same directory.')
flags.DEFINE_bool('use_wandb', True, 'Log to W&B.')
flags.DEFINE_string('wandb_project', 'continual_gcrl_paper',
                    'W&B project name.')
flags.DEFINE_string('wandb_group', 'C2: decomposed single-cell sanity',
                    'W&B group name.')
flags.DEFINE_bool('add_uid', False, 'Add UID to log dirs.')
flags.DEFINE_integer('start_task', 0, 'Resume from this task (loads ckpt from task-1).')
flags.DEFINE_integer('eval_every', 50_000, 'Evaluate every N env steps.')
flags.DEFINE_integer('time_delta_minutes', 5, 'Checkpoint frequency (minutes).')
flags.DEFINE_integer('num_actors', 1, 'Number of parallel actors (1 for sequential).')
flags.DEFINE_bool('use_task_id', False, 'Append one-hot task ID to state and goal.')
flags.DEFINE_string('critic_mode', 'persistent',
                    'Critic evolution across tasks: "persistent" (never reset, carry forward), '
                    '"reset" (reinitialize critic each task), '
                    '"cka" (CKA-RL style base+vectors for critic too), '
                    '"decomposed" (DCC), "rbc_decomposed" (RBC-DCC), '
                    '"dcc_sac" (gated actor fusion), '
                    '"dcc_sac_separate" (SAC-only actor ablation), '
                    '"action_dcc" (reward-free AC-DCC), or '
                    '"action_dcc_sac" (AC-DCC plus gated Q correction).')
flags.DEFINE_integer('eval_episodes', 10,
                     'Episodes per task for cross-task evaluation (0 to disable).')
flags.DEFINE_bool('intra_eval_previous_tasks', False,
                  'During training on the current task, periodically evaluate on '
                  'all previously learned tasks. Disabled by default because it '
                  'is expensive (creates envs for every past task at each eval interval).')
flags.DEFINE_enum(
    'post_task_eval_scope', 'all_seen', ['all_seen', 'current', 'none'],
    'Boundary evaluation scope: all tasks seen so far (legacy default), '
    'only the task just trained, or no boundary evaluation.')
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
flags.DEFINE_enum(
    'goal_conditioning_mode', 'full_state',
    goal_semantics.GOAL_CONDITIONING_MODES,
    'Goal contract seen by the policy and critic. full_state preserves the '
    'historical hand+gripper+mechanism goal; success_mechanism exposes the '
    'three mechanism coordinates; native_success_axis exposes only the exact '
    'official Task-5/Task-8 success coordinate.')
flags.DEFINE_enum(
    'sawyer_success_mode', 'corrected',
    ('corrected', 'legacy_distance', 'task_axis', 'native_info'),
    'Sparse success semantics for custom Sawyer wrappers. corrected is the '
    'normal behavior: Task 5 uses z, Task 8 uses x, Stick Pull requires an '
    'inserted stick, and other tasks retain their local predicates. '
    'legacy_distance reproduces historical paper '
    'runs; task_axis is the Task-5/8-only alias; native_info uses MetaWorld '
    'info[success].')
flags.DEFINE_bool(
    'profile_runtime', False,
    'Log coarse actor, learner, evaluation, and simulator-diagnostic wall '
    'times. Uses host clocks only and does not synchronize JAX arrays.')
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
flags.DEFINE_float(
    'shared_repr_scale', 1.0,
    'Fixed alpha in alpha * phi_shared(s,a) + phi_task(s,a) for plain '
    'DCC. 1.0 is the original DCC; 0.0 removes the shared branch from '
    'the contrastive score while leaving dynamics training on.')
flags.DEFINE_float('dyn_aux_after_task0', -1.0,
                   'If non-negative, override dyn_aux_weight to this value '
                   'starting at task 1 (the base task k=0 still uses '
                   '--dyn_aux_weight). Default -1.0 means "do nothing". '
                   'Used by the C2b ablation cell: dyn_aux_weight=1.0 at '
                   'k=0 only, 0.0 afterward, to test whether the dynamics '
                   'auxiliary contributes anything beyond the task-0 '
                   'initialiser role identified in '
                   'docs/2026-05-14_c2_ldyn_interpretation.md.')
flags.DEFINE_integer('phi_task_width', 256,
                     'Width of the per-task additive encoder phi_task '
                     '(critic_mode="decomposed" only). Smaller than the '
                     'shared body. Plan section 6.')
flags.DEFINE_integer('phi_task_depth', 4,
                     'Depth of the per-task additive encoder phi_task '
                     '(critic_mode="decomposed" only). Must be a positive '
                     'multiple of 4 when use_residual=True (one residual '
                     'block at depth=4). Plan section 6.')
flags.DEFINE_string('combine_mode', 'add',
                    'How z_shared and z_task are combined inside z_sa. '
                    '"add" (default) | "concat". When "concat", a learnable '
                    'Linear projection is automatically attached to the goal '
                    'side so the contrastive score is taken in matching '
                    '2*repr_dim space.')
flags.DEFINE_string('goal_encoder_mode', 'shared',
                    'Goal-encoder mode for the decomposed critic. '
                    '"shared" (default; single psi reused across tasks) | '
                    '"projected" (adds a Linear projection on top of psi(g) '
                    'regardless of combine_mode; useful ablation for '
                    'isolating the contribution of the projection itself).')
flags.DEFINE_integer(
    'in_trajectory_negative_repeats', 1,
    'Number of independently relabeled state--future-goal pairs sampled '
    'from each replay episode. 1 preserves the legacy sampler; values >1 '
    'enable CRTR/StableCRL in-trajectory negatives. StableCRL uses 12.')
flags.DEFINE_bool(
    'interaction_weighted_relabeling', False,
    'Reweight future-goal relabeling toward the hand/object interaction '
    'boundary (IWR); enabled by iwr_decomposed and bridge_decomposed.')
flags.DEFINE_float('interaction_threshold', 0.09,
                   'Hand/object distance defining the interaction bridge.')
flags.DEFINE_float('interaction_bandwidth', 0.03,
                   'Gaussian bandwidth around the interaction threshold.')
flags.DEFINE_float('interaction_weight_floor', 0.05,
                   'Minimum IWR weight, preserving support everywhere.')
flags.DEFINE_bool(
    'action_effect_enabled', False,
    'Enable the task-local forward action-effect/dual-advantage head.')
flags.DEFINE_float('action_effect_loss_weight', 1.0,
                   'Gradient weight for the action-effect prediction loss.')
flags.DEFINE_float('action_effect_discount', 0.99,
                   'Discount in gamma*psi(s_next)-psi(s).')
flags.DEFINE_float('action_effect_temperature', 1.0,
                   'Temperature before tanh-bounding local advantage.')
flags.DEFINE_float('action_effect_actor_weight', 1.0,
                   'Weight of local advantage in the actor objective.')
flags.DEFINE_float('action_effect_normalization_eps', 1e-3,
                   'Floor for DCC-logit actor normalization.')
flags.DEFINE_float('action_effect_q_scale_ema_decay', 0.99,
                   'EMA decay for the DCC-logit scale in the actor objective.')
flags.DEFINE_integer('action_effect_hidden_dim', 256,
                     'Hidden width of the task-local action-effect MLP.')
flags.DEFINE_enum(
    'action_effect_actor_mode', 'combined', ('combined', 'effect_only'),
    'Whether the actor maximizes normalized DCC plus the task-local head, '
    'or the task-local head alone. effect_only is the Stage-1 falsification.')
flags.DEFINE_enum(
    'action_effect_target_mode', 'psi_one_step',
    ('psi_one_step', 'raw_horizon', 'counterfactual_rank'),
    'Target for the task-local head. raw_horizon predicts H-step mechanism '
    'progress and success; counterfactual_rank learns within-state action '
    'ordering from exact simulator interventions on the original task goal.')
flags.DEFINE_integer('outcome_horizon', 25,
                     'Finite horizon H for raw outcome-credit labels.')
flags.DEFINE_float('outcome_success_threshold', 0.05,
                   'Mechanism-goal radius used for H-step success labels.')
flags.DEFINE_float('outcome_progress_loss_weight', 1.0,
                   'Huber-loss weight for standardized H-step progress.')
flags.DEFINE_float('outcome_success_loss_weight', 1.0,
                   'Sigmoid-BCE weight for H-step success prediction.')
flags.DEFINE_float('outcome_success_actor_weight', 1.0,
                   'Weight of predicted H-step success in the actor score.')
flags.DEFINE_float('outcome_progress_ema_decay', 0.99,
                   'EMA decay for raw progress target standardization.')
flags.DEFINE_float('outcome_progress_std_floor', 0.01,
                   'Minimum target standard deviation for normalization.')
flags.DEFINE_float('success_bc_weight', 0.0,
                   'Actor BC weight on retained task-goal successful actions.')
flags.DEFINE_integer('success_buffer_capacity', 4096,
                     'Task-local successful-transition ring-buffer capacity.')
flags.DEFINE_integer('success_bc_batch_size', 64,
                     'Successful actions sampled per actor update for BC.')
flags.DEFINE_integer('counterfactual_rank_interval_steps', 0,
                     'Env-step interval between exact-state rank collections; '
                     'zero disables the experiment and its extra environment.')
flags.DEFINE_integer('counterfactual_rank_num_anchors', 4,
                     'Identical-state anchors collected at each rank event.')
flags.DEFINE_integer('counterfactual_rank_candidates_per_family', 4,
                     'Candidates from each of policy/local/replay/uniform.')
flags.DEFINE_integer('counterfactual_rank_rollout_horizon', 100,
                     'Common task-goal rollout horizon for each candidate.')
flags.DEFINE_integer('counterfactual_rank_action_repeat', 5,
                     'Initial steps for which each candidate action is held.')
flags.DEFINE_float('counterfactual_rank_local_noise_std', 0.10,
                   'Stddev of local perturbations around the policy action.')
flags.DEFINE_enum('counterfactual_rank_anchor_mode', 'scripted_contact',
                  ('policy', 'scripted_contact'),
                  'How the isolated collector reaches interaction anchors.')
flags.DEFINE_integer('counterfactual_rank_anchor_search_steps', 150,
                     'Maximum prefix steps used to locate an anchor.')
flags.DEFINE_float('counterfactual_rank_interaction_threshold', 0.09,
                   'Hand-to-mechanism distance defining a near-contact anchor.')
flags.DEFINE_float('counterfactual_rank_contact_gain', 5.0,
                   'Proportional gain for the anchor-only contact controller.')
flags.DEFINE_float('counterfactual_rank_success_threshold', 0.05,
                   'Task-goal mechanism-distance success threshold.')
flags.DEFINE_float('counterfactual_rank_success_bonus', 1.0,
                   'Success bonus added to mechanism progress for ranking.')
flags.DEFINE_float('counterfactual_rank_min_outcome_gap', 0.002,
                   'Minimum outcome difference defining an informative pair.')
flags.DEFINE_integer('counterfactual_rank_buffer_capacity', 128,
                     'Task-local capacity measured in same-state anchors.')
flags.DEFINE_integer('counterfactual_rank_batch_anchors', 16,
                     'Anchor groups sampled per rank-head update.')
flags.DEFINE_integer('counterfactual_rank_updates_per_event', 25,
                     'Pairwise rank-head updates after each collection event.')
flags.DEFINE_float('counterfactual_rank_pairwise_temperature', 1.0,
                   'Temperature of the within-anchor logistic ranking loss.')
flags.DEFINE_float('counterfactual_rank_l2_weight', 1e-4,
                   'Mean-parameter L2 regularization on the task-local head.')
flags.DEFINE_integer(
    'counterfactual_rank_validation_anchors', 0,
    'Fresh disjoint anchors collected only for pre/post held-out metrics.')
flags.DEFINE_enum(
    'counterfactual_rank_success_mode', 'goal_distance',
    ('goal_distance', 'zero_reward', 'positive_reward'),
    'Success signal used in counterfactual labels. Raw Sawyer environment '
    'rewards are 0/1 and therefore require positive_reward; zero_reward is '
    'only for an explicitly step-penalized -1/0 stream.')
flags.DEFINE_bool(
    'counterfactual_rank_actor_enabled', True,
    'Whether the ordinary policy-gradient actor directly maximizes the rank '
    'head. False leaves DCC as the actor objective while still training and '
    'diagnosing the head.')
flags.DEFINE_integer(
    'counterfactual_oracle_interval_steps', 0,
    'Cadence for the four-condition exact-simulator oracle decomposition; '
    'zero disables it.')
flags.DEFINE_integer('counterfactual_oracle_num_anchors', 4,
                     'Anchors per condition in each oracle event.')
flags.DEFINE_enum(
    'counterfactual_oracle_condition_set', 'all',
    ('all', 'promotion_only'),
    'all runs policy/scripted-contact x repeat-1/repeat-5; promotion_only '
    'runs only scripted-contact/repeat-5 and costs about one quarter as many '
    'serial MuJoCo rollouts.')
flags.DEFINE_integer(
    'counterfactual_oracle_max_events', 0,
    'Maximum oracle events per task; zero preserves the unlimited legacy '
    'schedule. Use a small positive value for diagnostic-only runs.')
flags.DEFINE_bool(
    'phase_gated_control', False,
    'Use a reach/contact gate and execute rank-selected actions for the same '
    'chunk length used by counterfactual labels.')
flags.DEFINE_enum('phase_gate_reach_mode', 'policy',
                  ('policy', 'scripted_contact'),
                  'Controller used outside contact support.')
flags.DEFINE_float('phase_gate_interaction_threshold', 0.09,
                   'Hand-to-mechanism distance activating chunk control.')
flags.DEFINE_integer('phase_gate_chunk_length', 5,
                     'Environment steps used for each selected contact chunk.')
flags.DEFINE_integer('phase_gate_num_candidates', 16,
                     'Policy/local/uniform candidates scored at each replan.')
flags.DEFINE_float('phase_gate_local_noise_std', 0.10,
                   'Local candidate noise for phase-gated selection.')
flags.DEFINE_float('phase_gate_contact_gain', 5.0,
                   'Gain of the diagnostic scripted reach controller.')
flags.DEFINE_float('bellman_loss_weight', 1.0,
                   'RBC-DCC weight lambda_Q on the twin scalar TD loss.')
flags.DEFINE_float('bellman_residual_l2_weight', 1e-4,
                   'RBC-DCC L2 weight lambda_Delta on residual outputs.')
flags.DEFINE_float('bellman_discount', 0.99,
                   'RBC-DCC Bellman bootstrap discount gamma.')
flags.DEFINE_float('bellman_tau', 0.005,
                   'RBC-DCC Polyak target update rate.')
flags.DEFINE_integer('bellman_hidden_dim', 256,
                     'Hidden width of each resettable RBC Bellman residual.')
flags.DEFINE_float('her_reward_threshold', 0.05,
                   'RBC-DCC HER goal-reach radius. Included in checkpoint '
                   'identity; the same value is used by standalone SAC.')
flags.DEFINE_bool('step_penalty_reward', True,
                  'HER reward shape: -1/0 when true, 0/+1 otherwise.')
flags.DEFINE_float('dcc_sac_q_loss_weight', 1.0,
                   'Weight on the independent twin-Q TD loss.')
flags.DEFINE_float('dcc_sac_q_learning_rate', 3e-4,
                   'Learning rate for the independent raw-input twin Q.')
flags.DEFINE_float('dcc_sac_discount', 0.99,
                   'Bootstrap discount for the DCC-SAC Q critic.')
flags.DEFINE_float('dcc_sac_tau', 0.005,
                   'Polyak update rate for the DCC-SAC target Q.')
flags.DEFINE_integer('dcc_sac_q_hidden_dim', 1024,
                     'Width of each independent raw-input Q head.')
flags.DEFINE_float('dcc_sac_beta_max', 0.1,
                   'Maximum normalized Q-ranking correction weight.')
flags.DEFINE_integer('dcc_sac_q_warmup_updates', 10000,
                     'Learner updates before Q may influence the actor.')
flags.DEFINE_integer('dcc_sac_q_ramp_updates', 25000,
                     'Updates used to ramp the stable Q gate from zero.')
flags.DEFINE_float('dcc_sac_td_error_threshold', 0.5,
                   'Maximum EMA absolute TD error for opening the Q gate.')
flags.DEFINE_float('dcc_sac_twin_disagreement_threshold', 0.1,
                   'Maximum normalized twin disagreement for the Q gate.')
flags.DEFINE_float('dcc_sac_ema_decay', 0.99,
                   'EMA decay for DCC-SAC Q stability statistics.')
flags.DEFINE_integer('dcc_sac_candidate_actions', 8,
                     'Candidate actions used to normalize Q at fixed (s,g).')
flags.DEFINE_float('dcc_sac_normalization_eps', 1e-3,
                   'Minimum across-action Q scale in actor normalization.')
flags.DEFINE_float('dcc_sac_correction_clip', 5.0,
                   'Absolute clip on the normalized Q correction.')
flags.DEFINE_float('action_contrast_weight', 1.0,
                   'AC-DCC weight on fixed-(s,g_next) action InfoNCE.')
flags.DEFINE_float('action_contrast_temperature', 1.0,
                   'Temperature for AC-DCC action logits.')
flags.DEFINE_integer('action_contrast_batch_size', 32,
                     'Number of replay actions in each AC-DCC matrix.')
flags.DEFINE_integer('shortcut_diagnostic_interval', 0,
                     'Learner-call interval for shortcut/action diagnostics; '
                     '0 disables them and preserves the legacy hot path.')
flags.DEFINE_integer('shortcut_diagnostic_batch_size', 32,
                     'Replay anchors used by periodic shortcut diagnostics.')
flags.DEFINE_integer('shortcut_candidate_actions', 16,
                     'Fixed-state candidate actions used by diagnostics.')
flags.DEFINE_integer(
    'action_landscape_diagnostic_interval_steps', 0,
    'Environment-step cadence for causal same-state action ranking; 0 '
    'disables the probe.')
flags.DEFINE_integer('action_landscape_num_anchors', 1,
                     'MuJoCo anchor states tested at each causal probe event.')
flags.DEFINE_integer(
    'action_landscape_candidates_per_family', 4,
    'Actions per policy/local/replay/uniform family in the causal probe.')
flags.DEFINE_integer('action_landscape_rollout_horizon', 25,
                     'Counterfactual rollout length including the first '
                     'intervened action.')
flags.DEFINE_integer('action_landscape_anchor_prefix_steps', 20,
                     'Policy steps used to reach each diagnostic anchor.')
flags.DEFINE_float('action_landscape_local_noise_std', 0.10,
                   'Standard deviation of local policy-action perturbations.')
flags.DEFINE_bool(
    'action_landscape_interaction_aware_anchor', False,
    'Search a policy prefix for the closest hand/object interaction state.')
flags.DEFINE_integer('action_landscape_anchor_search_steps', 200,
                     'Maximum prefix length for interaction-aware anchors.')
flags.DEFINE_float('action_landscape_interaction_threshold', 0.09,
                   'Near-interaction distance used by the causal probe.')
flags.DEFINE_integer(
    'action_landscape_action_repeat', 1,
    'Initial repetitions of each intervened action. Set to the label chunk '
    'length for an aligned causal-ranking test.')
flags.DEFINE_bool(
    'action_landscape_use_best_progress', False,
    'Use best-over-horizon rather than final mechanism progress as the '
    'registered aligned causal outcome.')
flags.DEFINE_float('action_landscape_success_threshold', 0.05,
                   'Mechanism-distance proxy threshold logged by the probe.')
flags.DEFINE_enum(
    'action_landscape_success_mode', 'goal_distance',
    ('goal_distance', 'zero_reward', 'positive_reward'),
    'Success semantics for the causal probe. Use positive_reward for raw '
    'Sawyer 0/1 rewards and zero_reward only for a -1/0 stream.')
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

def _rbc_identity_config():
  """Resolved RBC settings that must participate in checkpoint identity."""
  return {
      'dyn_aux_weight': FLAGS.dyn_aux_weight,
      'dyn_aux_after_task0': FLAGS.dyn_aux_after_task0,
      'phi_task_width': FLAGS.phi_task_width,
      'phi_task_depth': FLAGS.phi_task_depth,
      'combine_mode': FLAGS.combine_mode,
      'goal_encoder_mode': FLAGS.goal_encoder_mode,
      'bellman_loss_weight': FLAGS.bellman_loss_weight,
      'bellman_residual_l2_weight': FLAGS.bellman_residual_l2_weight,
      'bellman_discount': FLAGS.bellman_discount,
      'bellman_tau': FLAGS.bellman_tau,
      'bellman_hidden_dim': FLAGS.bellman_hidden_dim,
      'her_reward_threshold': FLAGS.her_reward_threshold,
      'step_penalty_reward': FLAGS.step_penalty_reward,
  }



def _dcc_sac_identity_config():
  """Resolved settings defining DCC-SAC and AC-DCC checkpoints."""
  return {
      'critic_mode': FLAGS.critic_mode,
      'dyn_aux_weight': FLAGS.dyn_aux_weight,
      'dyn_aux_after_task0': FLAGS.dyn_aux_after_task0,
      'phi_task_width': FLAGS.phi_task_width,
      'phi_task_depth': FLAGS.phi_task_depth,
      'combine_mode': FLAGS.combine_mode,
      'goal_encoder_mode': FLAGS.goal_encoder_mode,
      'her_reward_threshold': FLAGS.her_reward_threshold,
      'step_penalty_reward': FLAGS.step_penalty_reward,
      'dcc_sac_q_loss_weight': FLAGS.dcc_sac_q_loss_weight,
      'dcc_sac_q_learning_rate': FLAGS.dcc_sac_q_learning_rate,
      'dcc_sac_discount': FLAGS.dcc_sac_discount,
      'dcc_sac_tau': FLAGS.dcc_sac_tau,
      'dcc_sac_q_hidden_dim': FLAGS.dcc_sac_q_hidden_dim,
      'dcc_sac_beta_max': FLAGS.dcc_sac_beta_max,
      'dcc_sac_q_warmup_updates': FLAGS.dcc_sac_q_warmup_updates,
      'dcc_sac_q_ramp_updates': FLAGS.dcc_sac_q_ramp_updates,
      'dcc_sac_td_error_threshold': FLAGS.dcc_sac_td_error_threshold,
      'dcc_sac_twin_disagreement_threshold':
          FLAGS.dcc_sac_twin_disagreement_threshold,
      'dcc_sac_ema_decay': FLAGS.dcc_sac_ema_decay,
      'dcc_sac_candidate_actions': FLAGS.dcc_sac_candidate_actions,
      'dcc_sac_normalization_eps': FLAGS.dcc_sac_normalization_eps,
      'dcc_sac_correction_clip': FLAGS.dcc_sac_correction_clip,
      'action_contrast_weight': FLAGS.action_contrast_weight,
      'action_contrast_temperature': FLAGS.action_contrast_temperature,
      'action_contrast_batch_size': FLAGS.action_contrast_batch_size,
      'shortcut_diagnostic_interval': FLAGS.shortcut_diagnostic_interval,
      'shortcut_diagnostic_batch_size':
          FLAGS.shortcut_diagnostic_batch_size,
      'shortcut_candidate_actions': FLAGS.shortcut_candidate_actions,
      'action_landscape_diagnostic_interval_steps':
          FLAGS.action_landscape_diagnostic_interval_steps,
      'action_landscape_num_anchors': FLAGS.action_landscape_num_anchors,
      'action_landscape_candidates_per_family':
          FLAGS.action_landscape_candidates_per_family,
      'action_landscape_rollout_horizon':
          FLAGS.action_landscape_rollout_horizon,
      'action_landscape_anchor_prefix_steps':
          FLAGS.action_landscape_anchor_prefix_steps,
      'action_landscape_local_noise_std':
          FLAGS.action_landscape_local_noise_std,
  }


def _bridge_identity_config():
  """Resolved settings defining IWR/action-effect DCC checkpoints."""
  return {
      'critic_mode': FLAGS.critic_mode,
      'dyn_aux_weight': FLAGS.dyn_aux_weight,
      'phi_task_width': FLAGS.phi_task_width,
      'phi_task_depth': FLAGS.phi_task_depth,
      'combine_mode': FLAGS.combine_mode,
      'goal_encoder_mode': FLAGS.goal_encoder_mode,
      'in_trajectory_negative_repeats':
          FLAGS.in_trajectory_negative_repeats,
      'interaction_weighted_relabeling':
          FLAGS.interaction_weighted_relabeling,
      'interaction_threshold': FLAGS.interaction_threshold,
      'interaction_bandwidth': FLAGS.interaction_bandwidth,
      'interaction_weight_floor': FLAGS.interaction_weight_floor,
      'action_effect_enabled': FLAGS.action_effect_enabled,
      'action_effect_loss_weight': FLAGS.action_effect_loss_weight,
      'action_effect_discount': FLAGS.action_effect_discount,
      'action_effect_temperature': FLAGS.action_effect_temperature,
      'action_effect_actor_weight': FLAGS.action_effect_actor_weight,
      'action_effect_normalization_eps':
          FLAGS.action_effect_normalization_eps,
      'action_effect_q_scale_ema_decay':
          FLAGS.action_effect_q_scale_ema_decay,
      'action_effect_hidden_dim': FLAGS.action_effect_hidden_dim,
      'action_effect_actor_mode': FLAGS.action_effect_actor_mode,
      'action_effect_target_mode': FLAGS.action_effect_target_mode,
      'outcome_horizon': FLAGS.outcome_horizon,
      'outcome_success_threshold': FLAGS.outcome_success_threshold,
      'outcome_progress_loss_weight': FLAGS.outcome_progress_loss_weight,
      'outcome_success_loss_weight': FLAGS.outcome_success_loss_weight,
      'outcome_success_actor_weight': FLAGS.outcome_success_actor_weight,
      'outcome_progress_ema_decay': FLAGS.outcome_progress_ema_decay,
      'outcome_progress_std_floor': FLAGS.outcome_progress_std_floor,
      'success_bc_weight': FLAGS.success_bc_weight,
      'success_buffer_capacity': FLAGS.success_buffer_capacity,
      'success_bc_batch_size': FLAGS.success_bc_batch_size,
      'counterfactual_rank_interval_steps':
          FLAGS.counterfactual_rank_interval_steps,
      'counterfactual_rank_num_anchors':
          FLAGS.counterfactual_rank_num_anchors,
      'counterfactual_rank_candidates_per_family':
          FLAGS.counterfactual_rank_candidates_per_family,
      'counterfactual_rank_rollout_horizon':
          FLAGS.counterfactual_rank_rollout_horizon,
      'counterfactual_rank_action_repeat':
          FLAGS.counterfactual_rank_action_repeat,
      'counterfactual_rank_local_noise_std':
          FLAGS.counterfactual_rank_local_noise_std,
      'counterfactual_rank_anchor_mode':
          FLAGS.counterfactual_rank_anchor_mode,
      'counterfactual_rank_anchor_search_steps':
          FLAGS.counterfactual_rank_anchor_search_steps,
      'counterfactual_rank_interaction_threshold':
          FLAGS.counterfactual_rank_interaction_threshold,
      'counterfactual_rank_contact_gain':
          FLAGS.counterfactual_rank_contact_gain,
      'counterfactual_rank_success_threshold':
          FLAGS.counterfactual_rank_success_threshold,
      'counterfactual_rank_success_bonus':
          FLAGS.counterfactual_rank_success_bonus,
      'counterfactual_rank_min_outcome_gap':
          FLAGS.counterfactual_rank_min_outcome_gap,
      'counterfactual_rank_buffer_capacity':
          FLAGS.counterfactual_rank_buffer_capacity,
      'counterfactual_rank_batch_anchors':
          FLAGS.counterfactual_rank_batch_anchors,
      'counterfactual_rank_updates_per_event':
          FLAGS.counterfactual_rank_updates_per_event,
      'counterfactual_rank_pairwise_temperature':
          FLAGS.counterfactual_rank_pairwise_temperature,
      'counterfactual_rank_l2_weight':
          FLAGS.counterfactual_rank_l2_weight,
      'counterfactual_rank_validation_anchors':
          FLAGS.counterfactual_rank_validation_anchors,
      'counterfactual_rank_success_mode':
          FLAGS.counterfactual_rank_success_mode,
      'counterfactual_rank_actor_enabled':
          FLAGS.counterfactual_rank_actor_enabled,
      'counterfactual_oracle_interval_steps':
          FLAGS.counterfactual_oracle_interval_steps,
      'counterfactual_oracle_num_anchors':
          FLAGS.counterfactual_oracle_num_anchors,
      'phase_gated_control': FLAGS.phase_gated_control,
      'phase_gate_reach_mode': FLAGS.phase_gate_reach_mode,
      'phase_gate_interaction_threshold':
          FLAGS.phase_gate_interaction_threshold,
      'phase_gate_chunk_length': FLAGS.phase_gate_chunk_length,
      'phase_gate_num_candidates': FLAGS.phase_gate_num_candidates,
      'phase_gate_local_noise_std': FLAGS.phase_gate_local_noise_std,
      'phase_gate_contact_gain': FLAGS.phase_gate_contact_gain,
      'action_landscape_action_repeat':
          FLAGS.action_landscape_action_repeat,
      'action_landscape_use_best_progress':
          FLAGS.action_landscape_use_best_progress,
      'action_landscape_success_threshold':
          FLAGS.action_landscape_success_threshold,
      'action_landscape_success_mode':
          FLAGS.action_landscape_success_mode,
  }

def _git_commit_sha():
  """Best-effort source revision for run manifests."""
  try:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True)
    return result.stdout.strip()
  except (OSError, subprocess.SubprocessError):
    return 'unknown'


def _ckpt_path(ckpt_dir, task_id, seed, critic_mode='persistent',
               use_task_id=True, adapt_heads_only=True, actor_mode='cka',
               dyn_aux_weight=None, phi_task_width=None, phi_task_depth=None,
               rbc_config=None, in_trajectory_negative_repeats=1,
               single_task='', goal_conditioning_mode='full_state',
               sawyer_success_mode='legacy_distance'):
  """Checkpoint path keyed by all ablation-relevant config.

  Base structure: {ckpt_dir}/actor_{mode}_critic_{mode}_tid_{bool}_heads_{bool}/seed_{seed}/task_{id}.pkl

  When ``critic_mode == 'decomposed'`` AND all three decomposed-specific
  arguments are provided, the directory key is extended with
  ``_dyn{w:.3f}_pt{W}x{D}`` so that different ``dyn_aux_weight`` /
  ``phi_task_width`` / ``phi_task_depth`` sweeps do NOT silently
  overwrite each other's checkpoints.

  Existing decomposed checkpoints written before 2026-05-14 sit under
  the un-extended path; callers should be prepared to print a
  migration notice if the new path is missing but the old path exists.

  Persistent / cka paths are unchanged.
  """
  config_key = (f'actor_{actor_mode}_critic_{critic_mode}'
                f'_tid_{use_task_id}_heads_{adapt_heads_only}')
  if goal_conditioning_mode != 'full_state':
    config_key += f'_goal_{goal_conditioning_mode}'
  if sawyer_success_mode != 'legacy_distance':
    config_key += f'_success_{sawyer_success_mode}'
  if critic_mode in _PLAIN_DCC_MODES and all(
      v is not None for v in (dyn_aux_weight, phi_task_width, phi_task_depth)):
    config_key += (f'_dyn{float(dyn_aux_weight):.3f}'
                   f'_pt{int(phi_task_width)}x{int(phi_task_depth)}')
  if (critic_mode in _PLAIN_DCC_MODES
      and int(in_trajectory_negative_repeats) > 1):
    config_key += f'_itn{int(in_trajectory_negative_repeats)}'
  # Single-task policies for different environments must never share task_0.
  # Historically this suffix was added only when repeats > 1, which made
  # ordinary repeat-1 Task-5 and Task-8 jobs overwrite one another.
  if critic_mode in _PLAIN_DCC_MODES and single_task:
    config_key += f'_env_{single_task}'
  if critic_mode == 'rbc_decomposed':
    if rbc_config is None:
      raise ValueError('rbc_config is required for RBC-DCC checkpoints.')
    config_key += (
        f'_rbc_{rbc_checkpointing.config_fingerprint(rbc_config)}')
  if critic_mode in _HYBRID_CRITIC_MODES:
    config_key += (
        f"_hybrid_{rbc_checkpointing.fingerprint_payload(_dcc_sac_identity_config())}")
  if critic_mode in (_IWR_MODES + _ACTION_EFFECT_MODES):
    config_key += (
        f"_bridge_{rbc_checkpointing.fingerprint_payload(_bridge_identity_config())}")
  return os.path.join(ckpt_dir, config_key, f'seed_{seed}',
                      f'task_{task_id}.pkl')


def save_ckpt(ckpt_dir, task_id, seed, data, critic_mode='persistent',
              use_task_id=True, adapt_heads_only=True, actor_mode='cka',
              dyn_aux_weight=None, phi_task_width=None, phi_task_depth=None,
              rbc_config=None, in_trajectory_negative_repeats=1,
              single_task='', goal_conditioning_mode='full_state',
              sawyer_success_mode='legacy_distance'):
  path = _ckpt_path(ckpt_dir, task_id, seed, critic_mode, use_task_id,
                     adapt_heads_only, actor_mode,
                     dyn_aux_weight=dyn_aux_weight,
                     phi_task_width=phi_task_width,
                     phi_task_depth=phi_task_depth,
                     rbc_config=rbc_config,
                     in_trajectory_negative_repeats=
                         in_trajectory_negative_repeats,
                     single_task=single_task,
                     goal_conditioning_mode=goal_conditioning_mode,
                     sawyer_success_mode=sawyer_success_mode)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  # Convert JAX arrays to numpy for pickling
  data_np = jax.tree_util.tree_map(
      lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x,
      data)
  with open(path, 'wb') as f:
    pickle.dump(data_np, f)
  print(f'  [ckpt] Saved → {path}', flush=True)


def load_ckpt(ckpt_dir, task_id, seed, critic_mode='persistent',
              use_task_id=True, adapt_heads_only=True, actor_mode='cka',
              dyn_aux_weight=None, phi_task_width=None, phi_task_depth=None,
              rbc_config=None, in_trajectory_negative_repeats=1,
              single_task='', goal_conditioning_mode='full_state',
              sawyer_success_mode='legacy_distance'):
  path = _ckpt_path(ckpt_dir, task_id, seed, critic_mode, use_task_id,
                     adapt_heads_only, actor_mode,
                     dyn_aux_weight=dyn_aux_weight,
                     phi_task_width=phi_task_width,
                     phi_task_depth=phi_task_depth,
                     rbc_config=rbc_config,
                     in_trajectory_negative_repeats=
                         in_trajectory_negative_repeats,
                     single_task=single_task,
                     goal_conditioning_mode=goal_conditioning_mode,
                     sawyer_success_mode=sawyer_success_mode)
  if not os.path.exists(path):
    # Migration notice: check whether a legacy (pre-2026-05-14)
    # un-disambiguated decomposed checkpoint exists at the OLD path.
    if (critic_mode == 'decomposed'
        and sawyer_success_mode == 'legacy_distance'
        and int(in_trajectory_negative_repeats) == 1):
      legacy_path = _ckpt_path(
          ckpt_dir, task_id, seed, critic_mode, use_task_id,
          adapt_heads_only, actor_mode,
          dyn_aux_weight=None, phi_task_width=None, phi_task_depth=None)
      if os.path.exists(legacy_path):
        raise FileNotFoundError(
            f'No checkpoint at the disambiguated path {path}, but a '
            f'legacy un-disambiguated checkpoint exists at {legacy_path}. '
            f'This file was written before the 2026-05-14 ckpt-path fix '
            f'and is ambiguous: it could correspond to any '
            f'(dyn_aux_weight, phi_task_width, phi_task_depth) configuration. '
            f'To recover, move it to the new path manually if you remember '
            f'the original config:\n'
            f'  mv {legacy_path} {path}\n'
            f'Otherwise re-run from task 0.')
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
      task_id=_tid, num_tasks=_ntasks,
      sawyer_success_mode=FLAGS.sawyer_success_mode)

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
    # ---- DCC actor carry (used only by decomposed + persistent actor) -----
    prev_dcc_policy_params: Optional[networks_lib.Params] = None,
    prev_dcc_policy_opt_state=None,
    prev_dcc_alpha_params=None,
    prev_dcc_alpha_opt_state=None,
    # ---- shared DCC carry (used by decomposed and rbc_decomposed) ----------
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

  When ``critic_mode`` is ``decomposed`` or ``rbc_decomposed``, the CKA actor
  pool plumbing is bypassed and the decomposed critic state is carried through
  the ``prev_b_shared_*`` / ``prev_h_phi_*`` / ``prev_h_dyn_*`` /
  ``prev_psi_*`` arguments instead. Plain DCC additionally uses the
  ``prev_dcc_*`` slots when ``actor_mode='persistent'``; the legacy CKA
  actor/pool return slots remain placeholders for decomposed modes.
  """

  np.random.seed(seed + task_id)

  # ---- early guards: incompatible flags for decomposed critic modes ------
  # Fail fast before booting the replay server. The decomposed learner
  # also raises on use_td/twin_q internally, but failing here keeps the
  # error attributable to a single source.
  if critic_mode in _DECOMPOSED_CRITIC_MODES:
    if config.use_td:
      raise ValueError(
          f"critic_mode={critic_mode!r} requires use_td=False; RBC uses its "
          "own scalar Bellman path.")
    if config.twin_q:
      raise ValueError(
          f"critic_mode={critic_mode!r} requires legacy twin_q=False.")
    if config.use_image_obs:
      raise ValueError(
          f"critic_mode={critic_mode!r} does not support use_image_obs=True.")
    if config.entropy_coefficient is not None:
      raise ValueError(
          f"critic_mode={critic_mode!r} requires adaptive entropy "
          "(config.entropy_coefficient=None).")
    if FLAGS.neg_bank_mode != 'off':
      raise ValueError(
          f"critic_mode={critic_mode!r} does not support "
          "neg_bank_mode != 'off'.")
    if actor_mode == 'cka':
      raise ValueError(
          f"critic_mode={critic_mode!r} is incompatible with actor_mode='cka' "
          "(use actor_mode='reset' or, for plain DCC, 'persistent').")
    if FLAGS.k_sample_k > 0:
      raise ValueError(
          f"critic_mode={critic_mode!r} does not currently support "
          "k_sample_k>0 "
          "in cross-task evaluation; the decomposed critic exposes a "
          "5-tuple bundle, not a single q_params pytree.")
  if critic_mode in _HYBRID_CRITIC_MODES and actor_mode != 'reset':
    raise ValueError(
        f'critic_mode={critic_mode!r} requires actor_mode=reset.')
  if critic_mode in (_IWR_MODES + _ACTION_EFFECT_MODES):
    if actor_mode != 'reset':
      raise ValueError(
          f'critic_mode={critic_mode!r} requires actor_mode=reset.')
    expected_iwr = critic_mode in _IWR_MODES
    expected_effect = critic_mode in _ACTION_EFFECT_MODES
    if bool(getattr(
        continual_cfg, 'interaction_weighted_relabeling', False)) != expected_iwr:
      raise ValueError(
          f'critic_mode={critic_mode!r} requires '
          f'interaction_weighted_relabeling={expected_iwr}.')
    if bool(getattr(
        continual_cfg, 'action_effect_enabled', False)) != expected_effect:
      raise ValueError(
          f'critic_mode={critic_mode!r} requires '
          f'action_effect_enabled={expected_effect}.')
  if critic_mode == 'rbc_decomposed':
    if actor_mode != 'reset':
      raise ValueError(
          "critic_mode='rbc_decomposed' v1 requires actor_mode='reset'.")
    if getattr(continual_cfg, 'combine_mode', 'add') != 'add':
      raise ValueError(
          "critic_mode='rbc_decomposed' v1 requires combine_mode='add'.")
    if getattr(continual_cfg, 'goal_encoder_mode', 'shared') != 'shared':
      raise ValueError(
          "critic_mode='rbc_decomposed' v1 requires "
          "goal_encoder_mode='shared'.")

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
      task_id=_tid, num_tasks=_ntasks,
      sawyer_success_mode=FLAGS.sawyer_success_mode)

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
  rbc_nets = None
  hybrid_sac_nets = None
  if critic_mode in _PLAIN_DCC_MODES:
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
        combine_mode=getattr(continual_cfg, 'combine_mode', 'add'),
        goal_encoder_mode=getattr(
            continual_cfg, 'goal_encoder_mode', 'shared'),
        shared_repr_scale=getattr(
            continual_cfg, 'shared_repr_scale', 1.0),
        action_effect_hidden_dim=getattr(
            continual_cfg, 'action_effect_hidden_dim', 256),
        action_effect_output_dim=(
            2 if getattr(continual_cfg, 'action_effect_target_mode',
                         'psi_one_step') == 'raw_horizon'
            else (1 if getattr(
                continual_cfg, 'action_effect_target_mode',
                'psi_one_step') == 'counterfactual_rank' else None)),
        action_effect_include_goal=(
            getattr(continual_cfg, 'action_effect_target_mode',
                    'psi_one_step') in (
                        'raw_horizon', 'counterfactual_rank')),
    )
  elif critic_mode in _HYBRID_CRITIC_MODES:
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
        combine_mode=getattr(continual_cfg, 'combine_mode', 'add'),
        goal_encoder_mode=getattr(
            continual_cfg, 'goal_encoder_mode', 'shared'),
    )
    if critic_mode != 'action_dcc':
      hybrid_sac_nets = sac_networks.make_sac_networks(
          env_spec,
          obs_dim=obs_dim,
          twin_q=True,
          use_residual=config.use_residual,
          network_width=int(getattr(
              continual_cfg, 'dcc_sac_q_hidden_dim', 1024)),
          critic_depth=config.critic_depth,
          actor_depth=config.actor_depth,
      )
  elif critic_mode == 'rbc_decomposed':
    rbc_nets = make_rbc_networks(
        env_spec, obs_dim=obs_dim,
        repr_dim=config.repr_dim,
        use_residual=config.use_residual,
        network_width=config.network_width,
        critic_depth=config.critic_depth,
        phi_task_width=getattr(continual_cfg, 'phi_task_width', 256),
        phi_task_depth=getattr(continual_cfg, 'phi_task_depth', 4),
        energy_fn=config.energy_fn,
        repr_norm=config.repr_norm,
        combine_mode=getattr(continual_cfg, 'combine_mode', 'add'),
        goal_encoder_mode=getattr(
            continual_cfg, 'goal_encoder_mode', 'shared'),
        bellman_hidden_dim=getattr(
            continual_cfg, 'bellman_hidden_dim', 256),
    )
    decomp_nets = rbc_nets.decomposed

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
  her_ops = (
      sac_her.tensorflow_ops()
      if critic_mode in _HER_CRITIC_MODES else None)

  in_trajectory_repeats = int(getattr(
      continual_cfg, 'in_trajectory_negative_repeats', 1))
  intrajectory.validate_repetition_factor(
      in_trajectory_repeats,
      batch_size=config.batch_size,
      episode_transitions=config.max_episode_steps)
  if in_trajectory_repeats > 1 and critic_mode not in _PLAIN_DCC_MODES:
    raise ValueError(
        'In-trajectory negatives are currently implemented only for plain '
        'DCC-family plain modes; got '
        f'critic_mode={critic_mode!r}.')
  trajectories_per_critic_batch = intrajectory.trajectories_per_batch(
      config.batch_size, in_trajectory_repeats)
  if in_trajectory_repeats > 1:
    counts = intrajectory.in_batch_repetition_counts(
        config.batch_size, in_trajectory_repeats)
    print(
        '  [in-trajectory negatives] '
        f'r={in_trajectory_repeats}; '
        f'{trajectories_per_critic_batch} replay episodes per '
        f'{config.batch_size}-row critic batch; group sizes={counts}.',
        flush=True)

  iwr_enabled = bool(getattr(
      continual_cfg, 'interaction_weighted_relabeling', False))
  interaction_threshold = float(getattr(
      continual_cfg, 'interaction_threshold', 0.09))
  interaction_bandwidth = float(getattr(
      continual_cfg, 'interaction_bandwidth', 0.03))
  interaction_weight_floor = float(getattr(
      continual_cfg, 'interaction_weight_floor', 0.05))
  if iwr_enabled:
    if config.obs_dim < 7:
      raise ValueError(
          'Interaction-weighted relabeling requires state coordinates '
          '[hand_xyz, gripper, mechanism_xyz] (obs_dim >= 7).')
    if interaction_bandwidth <= 0 or interaction_weight_floor <= 0:
      raise ValueError(
          'IWR bandwidth and weight floor must both be positive.')
    print(
        '  [IWR] future relabeling weighted near '
        f'd(hand, mechanism)={interaction_threshold:.3f} '
        f'(bandwidth={interaction_bandwidth:.3f}, '
        f'floor={interaction_weight_floor:.3f}).',
        flush=True)

  outcome_credit_enabled = (
      bool(getattr(continual_cfg, 'action_effect_enabled', False))
      and getattr(continual_cfg, 'action_effect_target_mode',
                  'psi_one_step') == 'raw_horizon')
  counterfactual_rank_enabled = (
      bool(getattr(continual_cfg, 'action_effect_enabled', False))
      and getattr(continual_cfg, 'action_effect_target_mode',
                  'psi_one_step') == 'counterfactual_rank')
  outcome_horizon = int(getattr(continual_cfg, 'outcome_horizon', 25))
  outcome_success_threshold = float(getattr(
      continual_cfg, 'outcome_success_threshold', 0.05))
  if outcome_credit_enabled:
    if config.obs_dim < 7:
      raise ValueError(
          'Raw outcome credit requires Sawyer coordinates '
          '[hand_xyz, gripper, mechanism_xyz] (obs_dim >= 7).')
    if outcome_horizon <= 0:
      raise ValueError('outcome_horizon must be positive.')
    print(
        '  [outcome credit] raw mechanism progress/success labels; '
        f'H={outcome_horizon}, threshold={outcome_success_threshold:.3f}.',
        flush=True)

  def _finite_horizon_labels(all_state, anchor_index, goal):
    """Vectorized raw mechanism progress and reachability labels."""
    return outcome_credit.tensorflow_finite_horizon_labels(
        tf, all_state, anchor_index, goal, horizon=outcome_horizon,
        threshold=outcome_success_threshold)

  def _interaction_candidate_weights(all_state):
    """Per-future-state IWR weights; returns ones when disabled."""
    if not iwr_enabled:
      return tf.ones((tf.shape(all_state)[0],), dtype=tf.float32)
    distance = tf.linalg.norm(
        all_state[:, :3] - all_state[:, 4:7], axis=1)
    standardized = (
        (distance - interaction_threshold) / interaction_bandwidth)
    return interaction_weight_floor + tf.exp(-0.5 * standardized ** 2)

  @tf.function
  def flatten_fn(sample):
    seq_len = tf.shape(sample.data.observation)[0]
    arange = tf.range(seq_len)
    is_future = tf.cast(arange[:, None] < arange[None], tf.float32)
    discount = config.discount ** tf.cast(arange[None] - arange[:, None], tf.float32)
    all_state = sample.data.observation[:, :config.obs_dim]
    probs = is_future * discount * _interaction_candidate_weights(
        all_state)[None, :]
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

    replay_reward = sample.data.reward[:-1]
    replay_discount = sample.data.discount[:-1]
    if critic_mode in _HER_CRITIC_MODES:
      achieved_next = contrastive_utils.obs_to_goal_2d(
          next_state,
          start_index=config.start_index,
          end_index=config.end_index)
      replay_reward, replay_discount = sac_her.her_reward_and_discount(
          achieved_next,
          goal,
          sample.data.discount[:-1],
          threshold=float(
              getattr(continual_cfg, 'her_reward_threshold', 0.05)),
          step_penalty_reward=bool(
              getattr(continual_cfg, 'step_penalty_reward', True)),
          ops=her_ops)

    extras = {'next_action': sample.data.action[1:]}
    if counterfactual_rank_enabled:
      original_goal = sample.data.observation[:-1, config.obs_dim:]
      extras['counterfactual_task_observation'] = tf.concat(
          [state, original_goal], axis=1)
    if outcome_credit_enabled:
      anchor_index = tf.range(seq_len - 1, dtype=tf.int32)
      progress, success = _finite_horizon_labels(
          all_state, anchor_index, goal)
      original_goal = sample.data.observation[:-1, config.obs_dim:]
      _, task_success = _finite_horizon_labels(
          all_state, anchor_index, original_goal)
      extras.update({
          'outcome_progress': progress,
          'outcome_success': success,
          'outcome_task_success': task_success,
          'outcome_retention_observation':
              tf.concat([state, original_goal], axis=1),
      })
    if iwr_enabled:
      selected_future_state = tf.gather(all_state, goal_index[:-1])
      selected_distance = tf.linalg.norm(
          selected_future_state[:, :3] - selected_future_state[:, 4:7],
          axis=1)
      extras['iwr_interaction_distance'] = selected_distance
      extras['iwr_sampling_weight'] = (
          interaction_weight_floor + tf.exp(
              -0.5 * ((selected_distance - interaction_threshold)
                      / interaction_bandwidth) ** 2))
    transition = types.Transition(
        observation=new_obs, action=sample.data.action[:-1],
        reward=replay_reward, discount=replay_discount,
        next_observation=new_next_obs,
        extras=extras)
    shift = tf.random.uniform((), 0, seq_len, tf.int32)
    transition = tree.map_structure(lambda t: tf.roll(t, shift, axis=0), transition)
    return transition

  @tf.function
  def flatten_intrajectory_fn(sample):
    """Draw repeated pairs from one episode for CRTR/StableCRL negatives."""
    seq_len = tf.shape(sample.data.observation)[0]
    num_transitions = seq_len - 1
    anchor_index = tf.random.uniform(
        shape=(in_trajectory_repeats,), minval=0,
        maxval=num_transitions, dtype=tf.int32)
    candidate_index = tf.range(seq_len, dtype=tf.int32)
    is_future = tf.cast(
        anchor_index[:, None] < candidate_index[None, :], tf.float32)
    delta = candidate_index[None, :] - anchor_index[:, None]
    all_state = sample.data.observation[:, :config.obs_dim]
    probs = is_future * (
        config.discount ** tf.cast(delta, tf.float32))
    probs *= _interaction_candidate_weights(all_state)[None, :]
    goal_index = tf.random.categorical(
        logits=tf.math.log(probs), num_samples=1)[:, 0]
    all_goal = contrastive_utils.obs_to_goal_2d(
        all_state, start_index=config.start_index,
        end_index=config.end_index)
    state = tf.gather(all_state, anchor_index)
    next_state = tf.gather(all_state, anchor_index + 1)
    goal = tf.gather(all_goal, goal_index)
    extras = {
        'next_action': tf.gather(
            sample.data.action, anchor_index + 1),
    }
    if counterfactual_rank_enabled:
      original_goal = tf.gather(
          sample.data.observation[:, config.obs_dim:], anchor_index)
      extras['counterfactual_task_observation'] = tf.concat(
          [state, original_goal], axis=1)
    if outcome_credit_enabled:
      progress, success = _finite_horizon_labels(
          all_state, anchor_index, goal)
      original_goal = tf.gather(
          sample.data.observation[:, config.obs_dim:], anchor_index)
      _, task_success = _finite_horizon_labels(
          all_state, anchor_index, original_goal)
      extras.update({
          'outcome_progress': progress,
          'outcome_success': success,
          'outcome_task_success': task_success,
          'outcome_retention_observation':
              tf.concat([state, original_goal], axis=1),
      })
    if iwr_enabled:
      selected_future_state = tf.gather(all_state, goal_index)
      selected_distance = tf.linalg.norm(
          selected_future_state[:, :3] - selected_future_state[:, 4:7],
          axis=1)
      extras['iwr_interaction_distance'] = selected_distance
      extras['iwr_sampling_weight'] = (
          interaction_weight_floor + tf.exp(
              -0.5 * ((selected_distance - interaction_threshold)
                      / interaction_bandwidth) ** 2))
    return types.Transition(
        observation=tf.concat([state, goal], axis=1),
        action=tf.gather(sample.data.action, anchor_index),
        reward=tf.gather(sample.data.reward, anchor_index),
        discount=tf.gather(sample.data.discount, anchor_index),
        next_observation=tf.concat([next_state, goal], axis=1),
        extras=extras)

  # Use a single interleave worker to avoid deadlocks with drop_remainder
  # batching during early sampling when the replay buffer is small.
  num_parallel_calls = 1

  def _make_dataset(unused):
    ds = reverb.TrajectoryDataset.from_table_signature(
        server_address=replay_client.server_address,
        table=config.replay_table_name,
        max_in_flight_samples_per_worker=100)
    ds = ds.map(
        flatten_intrajectory_fn
        if in_trajectory_repeats > 1 else flatten_fn)
    def _transpose_fn(t):
      dims = tf.range(tf.shape(tf.shape(t))[0])
      perm = tf.concat([[1, 0], dims[2:]], axis=0)
      return tf.transpose(t, perm)
    if in_trajectory_repeats > 1:
      ds = ds.batch(trajectories_per_critic_batch, drop_remainder=True)

      def _pack_intrajectory_batch(tr):
        def _pack(t):
          # ``Dataset.batch`` produces [episode, repetition, ...].
          shape = tf.shape(t)
          packed = tf.reshape(
              t,
              tf.concat([[shape[0] * shape[1]], shape[2:]], axis=0))
          return packed[:config.batch_size]
        return tree.map_structure(_pack, tr)

      ds = ds.map(_pack_intrajectory_batch)
      ds = ds.unbatch()
    else:
      # Legacy path is intentionally unchanged.
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
  if (critic_mode in _PLAIN_DCC_MODES
      and FLAGS.in_trajectory_negative_repeats > 1):
    config_tag += f'_itn{FLAGS.in_trajectory_negative_repeats}'
  if critic_mode == 'rbc_decomposed':
    config_tag += (
        f'_rbc_{rbc_checkpointing.config_fingerprint(_rbc_identity_config())}')
  elif critic_mode in _HYBRID_CRITIC_MODES:
    config_tag += (
        f"_hybrid_{rbc_checkpointing.fingerprint_payload(_dcc_sac_identity_config())}")
  if critic_mode in (_IWR_MODES + _ACTION_EFFECT_MODES):
    config_tag += (
        f"_bridge_{rbc_checkpointing.fingerprint_payload(_bridge_identity_config())}")
  log_dir = os.path.join(
      FLAGS.log_dir, f'continual_{config.alg_name}', config_tag,
      f'task{task_id}_{env_name}_s{seed}')
  os.makedirs(log_dir, exist_ok=True)
  if (
      critic_mode == 'rbc_decomposed'
      or critic_mode in _HYBRID_CRITIC_MODES
      or critic_mode in (_IWR_MODES + _ACTION_EFFECT_MODES)):
    identity = (
        _rbc_identity_config()
        if critic_mode == 'rbc_decomposed'
        else (_dcc_sac_identity_config()
              if critic_mode in _HYBRID_CRITIC_MODES
              else _bridge_identity_config()))
    manifest = {
        'git_commit': _git_commit_sha(),
        'critic_mode': critic_mode,
        'actor_mode': actor_mode,
        'task_id': task_id,
        'env_name': env_name,
        'seed': seed,
        **identity,
    }
    with open(os.path.join(log_dir, 'resolved_config.json'), 'w') as handle:
      json.dump(manifest, handle, indent=2, sort_keys=True, default=str)

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

  if critic_mode in _PLAIN_DCC_MODES:
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
        actor_mode=actor_mode,
        prev_policy_params=prev_dcc_policy_params,
        prev_policy_opt_state=prev_dcc_policy_opt_state,
        prev_alpha_params=prev_dcc_alpha_params,
        prev_alpha_opt_state=prev_dcc_alpha_opt_state,
        prev_b_shared_params=prev_b_shared_params,
        prev_b_shared_opt_state=prev_b_shared_opt_state,
        prev_h_phi_params=prev_h_phi_params,
        prev_h_phi_opt_state=prev_h_phi_opt_state,
        prev_h_dyn_params=prev_h_dyn_params,
        prev_h_dyn_opt_state=prev_h_dyn_opt_state,
        prev_psi_params=prev_psi_params,
        prev_psi_opt_state=prev_psi_opt_state,
    )
  elif critic_mode in _HYBRID_CRITIC_MODES:
    learner = ContinualDCCSACLearner(
        hybrid_mode=critic_mode,
        decomp_nets=decomp_nets,
        q_network=(
          hybrid_sac_nets.q_network
          if hybrid_sac_nets is not None else None),
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
  elif critic_mode == 'rbc_decomposed':
    learner = ContinualRBCDecomposedLearner(
        rbc_nets=rbc_nets,
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
      task_id=_tid, num_tasks=_ntasks,
      sawyer_success_mode=FLAGS.sawyer_success_mode)
  phase_control_enabled = bool(getattr(
      continual_cfg, 'phase_gated_control', False))
  if phase_control_enabled:
    if not counterfactual_rank_enabled:
      raise ValueError(
          'phase_gated_control requires counterfactual_rank target mode.')
    if not isinstance(learner, ContinualDecomposedLearner):
      raise ValueError(
          'phase_gated_control requires ContinualDecomposedLearner.')

    def _phase_score_actions(observation, actions):
      return np.asarray(
          learner.score_counterfactual_actions(observation, actions))

    phase_kwargs = {
        'obs_dim': config.obs_dim,
        'score_actions_fn': _phase_score_actions,
        'reach_mode': getattr(
            continual_cfg, 'phase_gate_reach_mode', 'policy'),
        'interaction_threshold': float(getattr(
            continual_cfg, 'phase_gate_interaction_threshold', 0.09)),
        'chunk_length': int(getattr(
            continual_cfg, 'phase_gate_chunk_length', 5)),
        'num_candidates': int(getattr(
            continual_cfg, 'phase_gate_num_candidates', 16)),
        'local_noise_std': float(getattr(
            continual_cfg, 'phase_gate_local_noise_std', 0.10)),
        'contact_gain': float(getattr(
            continual_cfg, 'phase_gate_contact_gain', 5.0)),
    }
    actor = phase_gated_control.PhaseGatedChunkActor(
        actor, action_spec=env.action_spec(),
        rng=np.random.default_rng(seed + task_id + 610), **phase_kwargs)
    eval_actor = phase_gated_control.PhaseGatedChunkActor(
        eval_actor, action_spec=eval_env.action_spec(),
        rng=np.random.default_rng(seed + task_id + 620), **phase_kwargs)
  eval_observers = [
      contrastive_utils.SuccessObserver(),
      contrastive_utils.DistanceObserver(
          obs_dim=config.obs_dim,
          start_index=config.start_index,
          end_index=config.end_index),
  ]
  task58_stage_metrics_enabled = (
      env_name in task58_reevaluation.TASK58_SUCCESS_SPECS
      and FLAGS.goal_conditioning_mode == 'full_state'
      and FLAGS.sawyer_success_mode in (
          'corrected', 'task_axis', 'legacy_distance'))
  if task58_stage_metrics_enabled:
    emitted_success_mode = (
        'legacy_distance'
        if FLAGS.sawyer_success_mode == 'legacy_distance'
        else 'corrected')
    eval_observers.append(
        task58_reevaluation.PairedTask58SuccessObserver(
            obs_dim=config.obs_dim,
            env_name=env_name,
            emitted_success_mode=emitted_success_mode,
            mechanism_target=fixed_goal))
  evaluator_logger = make_default_logger(
      'evaluator', save_data=True, save_dir=log_dir,
      add_uid=config.add_uid, use_wandb=config.use_wandb,
      time_delta=10.0, steps_key='actor_steps')
  eval_loop = environment_loop.EnvironmentLoop(
      eval_env, eval_actor, counter=counting.Counter(),
      logger=evaluator_logger, observers=eval_observers)

  # Separate environment for causal counterfactuals.  It is never attached to
  # the replay adder and therefore cannot perturb training data or evaluator
  # episode state.  The default interval is zero, preserving prior runtime.
  action_landscape_interval = int(getattr(
      continual_cfg, 'action_landscape_diagnostic_interval_steps', 0))
  action_landscape_env = None
  if action_landscape_interval > 0:
    if decomp_nets is None:
      raise ValueError(
          'The causal action-landscape probe requires a DCC-family critic.')
    action_landscape_env, _ = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index,
        seed + task_id + 400, fixed_start_end=fixed_goal,
        task_id=_tid, num_tasks=_ntasks,
        sawyer_success_mode=FLAGS.sawyer_success_mode)

  # The training intervention uses its own simulator. It never writes replay
  # and cannot alter the actor/evaluator environments. The interval defaults
  # to zero, so every previous algorithm keeps its original runtime/path.
  counterfactual_rank_interval = int(getattr(
      continual_cfg, 'counterfactual_rank_interval_steps', 0))
  counterfactual_oracle_interval = int(getattr(
      continual_cfg, 'counterfactual_oracle_interval_steps', 0))
  counterfactual_validation_anchors = int(getattr(
      continual_cfg, 'counterfactual_rank_validation_anchors', 0))
  counterfactual_rank_env = None
  counterfactual_validation_env = None
  counterfactual_oracle_envs = {}
  if FLAGS.counterfactual_oracle_condition_set == 'promotion_only':
    counterfactual_oracle_conditions = (('scripted_contact', 5),)
  else:
    counterfactual_oracle_conditions = (
        ('policy', 1), ('policy', 5),
        ('scripted_contact', 1), ('scripted_contact', 5))
  if counterfactual_rank_interval > 0 or counterfactual_oracle_interval > 0:
    if config.obs_dim < 7:
      raise ValueError(
          'Counterfactual stages require Sawyer hand/mechanism coordinates.')
  if counterfactual_rank_interval > 0:
    counterfactual_rank_env, _ = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index,
        seed + task_id + 500, fixed_start_end=fixed_goal,
        task_id=_tid, num_tasks=_ntasks,
        sawyer_success_mode=FLAGS.sawyer_success_mode)
    if not counterfactual_rank_enabled:
      raise ValueError(
          'counterfactual_rank_interval_steps requires '
          'action_effect_target_mode=counterfactual_rank.')
    if not isinstance(learner, ContinualDecomposedLearner):
      raise ValueError(
          'Counterfactual ranking requires ContinualDecomposedLearner.')
    if counterfactual_validation_anchors > 0:
      counterfactual_validation_env, _ = (
          contrastive_utils.make_environment(
              env_name, config.start_index, config.end_index,
              seed + task_id + 550, fixed_start_end=fixed_goal,
              task_id=_tid, num_tasks=_ntasks,
              sawyer_success_mode=FLAGS.sawyer_success_mode))
  if counterfactual_oracle_interval > 0:
    # Repeat-1 and repeat-5 environments for a given anchor mode use the same
    # seed and advance independently. This makes their reset/anchor sequences
    # paired without allowing one condition's rollout to perturb another.
    for oracle_anchor_mode, oracle_repeat in counterfactual_oracle_conditions:
      oracle_seed_offset = 570 if oracle_anchor_mode == 'policy' else 580
      oracle_env, _ = contrastive_utils.make_environment(
          env_name, config.start_index, config.end_index,
          seed + task_id + oracle_seed_offset,
          fixed_start_end=fixed_goal,
          task_id=_tid, num_tasks=_ntasks,
          sawyer_success_mode=FLAGS.sawyer_success_mode)
      counterfactual_oracle_envs[
          (oracle_anchor_mode, oracle_repeat)] = oracle_env

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
  next_action_landscape_at = (
      action_landscape_interval if action_landscape_interval > 0
      else float('inf'))
  action_landscape_rng = np.random.default_rng(
      seed + 88001 + task_id * 101)
  # Fire after the first learner step so the head receives valid supervision
  # before it is allowed to influence the actor on subsequent updates.
  next_counterfactual_rank_at = (
      0 if counterfactual_rank_interval > 0 else float('inf'))
  next_counterfactual_oracle_at = (
      0 if counterfactual_oracle_interval > 0 else float('inf'))
  counterfactual_rank_rng = np.random.default_rng(
      seed + 99001 + task_id * 103)
  counterfactual_validation_rng = np.random.default_rng(
      seed + 99501 + task_id * 107)
  counterfactual_oracle_event = 0
  counterfactual_rank_buffer = (
      counterfactual_ranking.CounterfactualRankingBuffer(int(getattr(
          continual_cfg, 'counterfactual_rank_buffer_capacity', 128)))
      if counterfactual_rank_interval > 0 else None)
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

  runtime_totals = {
      'actor_seconds': 0.0,
      'learner_seconds': 0.0,
      'counterfactual_rank_seconds': 0.0,
      'counterfactual_oracle_seconds': 0.0,
      'action_landscape_seconds': 0.0,
      'evaluation_seconds': 0.0,
      'rl_metrics_seconds': 0.0,
  }
  runtime_started_at = time.perf_counter()

  while env_steps_done < train_steps:
    # Actor step: run one full episode and count actual env steps.
    # NOTE: Acme's `EnvironmentLoop.run()` returns None (it only writes logs),
    # so we call `run_episode()` to get the per-episode metrics dict.
    actor_started_at = time.perf_counter()
    result = env_loop.run_episode()
    if FLAGS.profile_runtime:
      runtime_totals['actor_seconds'] += (
          time.perf_counter() - actor_started_at)
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
    learner_started_at = time.perf_counter()
    learner.step()
    if FLAGS.profile_runtime:
      runtime_totals['learner_seconds'] += (
          time.perf_counter() - learner_started_at)
    if episodes_done == 1:
      print(f'  JIT compilation done.', flush=True)

    # Diagnostic events use a learner-call cadence, whereas ordinary learner
    # logging uses an environment-step cadence.  Log these values immediately
    # so a valid event cannot be silently dropped between W&B rows.
    diagnostic_metrics = getattr(learner, 'last_diagnostic_metrics', {})
    if diagnostic_metrics and FLAGS.use_wandb and wandb is not None:
      wandb_diagnostic = {
          f'learner/{name}': float(value)
          for name, value in diagnostic_metrics.items()}
      wandb_diagnostic['learner/env_steps'] = env_steps_done
      wandb.log(wandb_diagnostic)

    if env_steps_done >= next_counterfactual_rank_at:
      rank_started_at = time.perf_counter()
      transitions = learner.last_transitions
      if transitions is None:
        raise RuntimeError(
            'Counterfactual rank event fired before replay transitions existed.')
      policy_params = learner.get_variables(['policy'])[0]

      def _rank_policy_action(observation, numpy_rng, stochastic):
        key = jax.random.PRNGKey(
            int(numpy_rng.integers(0, np.iinfo(np.int32).max)))
        distribution = networks.policy_network.apply(
            policy_params, jnp.asarray(observation)[None, :])
        sampler = networks.sample if stochastic else networks.sample_eval
        return np.asarray(sampler(distribution, key))[0]

      print(
          f'  [counterfactual-rank @ {env_steps_done}] collecting original-'
          'task-goal outcomes from identical simulator states...', flush=True)
      collection_kwargs = {
          'obs_dim': config.obs_dim,
          'replay_observations': np.asarray(transitions.observation),
          'replay_actions': np.asarray(transitions.action),
          'policy_action_fn': _rank_policy_action,
          'candidates_per_family': int(getattr(
              continual_cfg, 'counterfactual_rank_candidates_per_family', 4)),
          'rollout_horizon': int(getattr(
              continual_cfg, 'counterfactual_rank_rollout_horizon', 100)),
          'action_repeat': int(getattr(
              continual_cfg, 'counterfactual_rank_action_repeat', 5)),
          'local_noise_std': float(getattr(
              continual_cfg, 'counterfactual_rank_local_noise_std', 0.10)),
          'anchor_mode': getattr(
              continual_cfg, 'counterfactual_rank_anchor_mode',
              'scripted_contact'),
          'anchor_search_steps': int(getattr(
              continual_cfg, 'counterfactual_rank_anchor_search_steps', 150)),
          'interaction_threshold': float(getattr(
              continual_cfg, 'counterfactual_rank_interaction_threshold',
              0.09)),
          'contact_gain': float(getattr(
              continual_cfg, 'counterfactual_rank_contact_gain', 5.0)),
          'success_threshold': float(getattr(
              continual_cfg, 'counterfactual_rank_success_threshold', 0.05)),
          'success_mode': getattr(
              continual_cfg, 'counterfactual_rank_success_mode',
              'goal_distance'),
          'success_bonus': float(getattr(
              continual_cfg, 'counterfactual_rank_success_bonus', 1.0)),
          'min_outcome_gap': float(getattr(
              continual_cfg, 'counterfactual_rank_min_outcome_gap', 0.002)),
      }
      rank_batch, rank_metrics = (
          counterfactual_ranking.collect_counterfactual_ranking_batch(
              environment=counterfactual_rank_env,
              rng=counterfactual_rank_rng,
              num_anchors=int(getattr(
                  continual_cfg, 'counterfactual_rank_num_anchors', 4)),
              **collection_kwargs))
      validation_batch = None
      validation_collection_metrics = {}
      if counterfactual_validation_anchors > 0:
        validation_batch, validation_collection_metrics = (
            counterfactual_ranking.collect_counterfactual_ranking_batch(
                environment=counterfactual_validation_env,
                rng=counterfactual_validation_rng,
                num_anchors=counterfactual_validation_anchors,
                **collection_kwargs))

      def _prefixed_score_metrics(metrics, split):
        result = {}
        for name, value in metrics.items():
          suffix = name.split('/', 1)[-1]
          result[f'counterfactual_rank/{split}/{suffix}'] = float(value)
        return result

      min_gap = float(getattr(
          continual_cfg, 'counterfactual_rank_min_outcome_gap', 0.002))
      train_pre_scores = learner.score_counterfactual_batch(rank_batch)
      train_pre_metrics = counterfactual_ranking.summarize_counterfactual_scores(
          train_pre_scores, rank_batch, min_outcome_gap=min_gap)
      heldout_pre_metrics = {}
      if validation_batch is not None:
        heldout_pre_scores = learner.score_counterfactual_batch(
            validation_batch)
        heldout_pre_metrics = (
            counterfactual_ranking.summarize_counterfactual_scores(
                heldout_pre_scores, validation_batch,
                min_outcome_gap=min_gap))

      counterfactual_rank_buffer.add(rank_batch)
      train_metrics = {}
      for _ in range(int(getattr(
          continual_cfg, 'counterfactual_rank_updates_per_event', 25))):
        sampled_rank_batch = counterfactual_rank_buffer.sample(
            int(getattr(
                continual_cfg, 'counterfactual_rank_batch_anchors', 16)),
            counterfactual_rank_rng)
        train_metrics = learner.train_counterfactual_ranker(
            sampled_rank_batch, updates=1)
      rank_scores = learner.score_counterfactual_batch(rank_batch)
      score_metrics = counterfactual_ranking.summarize_counterfactual_scores(
          rank_scores, rank_batch, min_outcome_gap=min_gap)
      heldout_post_metrics = {}
      if validation_batch is not None:
        heldout_post_scores = learner.score_counterfactual_batch(
            validation_batch)
        heldout_post_metrics = (
            counterfactual_ranking.summarize_counterfactual_scores(
                heldout_post_scores, validation_batch,
                min_outcome_gap=min_gap))
      rank_metrics = {
          **rank_metrics,
          **_prefixed_score_metrics(
              validation_collection_metrics, 'heldout_collection'),
          **_prefixed_score_metrics(train_pre_metrics, 'train_pre'),
          **_prefixed_score_metrics(heldout_pre_metrics, 'heldout_pre'),
          **_prefixed_score_metrics(train_metrics, 'optimization'),
          **score_metrics,
          **_prefixed_score_metrics(score_metrics, 'train_post'),
          **_prefixed_score_metrics(heldout_post_metrics, 'heldout_post'),
          'counterfactual_rank/buffer_anchors': float(
              len(counterfactual_rank_buffer)),
          'counterfactual_rank/updates_total': float(
              train_metrics.get('counterfactual_rank/updates_total', 0.0)),
      }
      if FLAGS.use_wandb and wandb is not None:
        wandb_rank = {
            f'learner/{name}': float(value)
            for name, value in rank_metrics.items()}
        wandb_rank['learner/env_steps'] = env_steps_done
        wandb.log(wandb_rank)
      print(
          '  [counterfactual-rank] '
          f"informative={rank_metrics.get('counterfactual_rank/informative_anchor_fraction', 0.0):.2f} "
          f"train_rho={rank_metrics.get('counterfactual_rank/score_vs_outcome_spearman', 0.0):.3f} "
          f"heldout_rho={rank_metrics.get('counterfactual_rank/heldout_post/score_vs_outcome_spearman', float('nan')):.3f} "
          f"pair_acc={rank_metrics.get('counterfactual_rank/pairwise_accuracy', 0.0):.3f} "
          f"updates={rank_metrics.get('counterfactual_rank/updates_total', 0.0):.0f}",
          flush=True)
      next_counterfactual_rank_at = (
          env_steps_done + counterfactual_rank_interval)
      if FLAGS.profile_runtime:
        runtime_totals['counterfactual_rank_seconds'] += (
            time.perf_counter() - rank_started_at)

    if env_steps_done >= next_counterfactual_oracle_at:
      oracle_started_at = time.perf_counter()
      transitions = learner.last_transitions
      if transitions is None:
        raise RuntimeError(
            'Counterfactual oracle event fired before replay existed.')
      policy_params = learner.get_variables(['policy'])[0]

      def _oracle_policy_action(observation, numpy_rng, stochastic):
        key = jax.random.PRNGKey(
            int(numpy_rng.integers(0, np.iinfo(np.int32).max)))
        distribution = networks.policy_network.apply(
            policy_params, jnp.asarray(observation)[None, :])
        sampler = networks.sample if stochastic else networks.sample_eval
        return np.asarray(sampler(distribution, key))[0]

      oracle_metrics = {}
      for anchor_mode, action_repeat in counterfactual_oracle_conditions:
        oracle_condition_rng = np.random.default_rng(
            seed + 99701 + task_id * 109
            + counterfactual_oracle_event * 1009
            + (0 if anchor_mode == 'policy' else 1))
        oracle_batch, collection_metrics = (
            counterfactual_ranking.collect_counterfactual_ranking_batch(
                environment=counterfactual_oracle_envs[
                    (anchor_mode, action_repeat)],
                obs_dim=config.obs_dim,
                replay_observations=np.asarray(transitions.observation),
                replay_actions=np.asarray(transitions.action),
                policy_action_fn=_oracle_policy_action,
                rng=oracle_condition_rng,
                num_anchors=int(getattr(
                    continual_cfg, 'counterfactual_oracle_num_anchors', 4)),
                candidates_per_family=int(getattr(
                    continual_cfg,
                    'counterfactual_rank_candidates_per_family', 4)),
                rollout_horizon=int(getattr(
                    continual_cfg,
                    'counterfactual_rank_rollout_horizon', 100)),
                action_repeat=action_repeat,
                local_noise_std=float(getattr(
                    continual_cfg,
                    'counterfactual_rank_local_noise_std', 0.10)),
                anchor_mode=anchor_mode,
                anchor_search_steps=int(getattr(
                    continual_cfg,
                    'counterfactual_rank_anchor_search_steps', 150)),
                interaction_threshold=float(getattr(
                    continual_cfg,
                    'counterfactual_rank_interaction_threshold', 0.09)),
                contact_gain=float(getattr(
                    continual_cfg,
                    'counterfactual_rank_contact_gain', 5.0)),
                success_threshold=float(getattr(
                    continual_cfg,
                    'counterfactual_rank_success_threshold', 0.05)),
                success_mode=getattr(
                    continual_cfg, 'counterfactual_rank_success_mode',
                    'goal_distance'),
                success_bonus=float(getattr(
                    continual_cfg,
                    'counterfactual_rank_success_bonus', 1.0)),
                min_outcome_gap=float(getattr(
                    continual_cfg,
                    'counterfactual_rank_min_outcome_gap', 0.002))))
        condition = f'{anchor_mode}/repeat{action_repeat}'
        condition_metrics = {
            **counterfactual_ranking.summarize_oracle(oracle_batch),
            'oracle/proxy_success_fraction': collection_metrics[
                'counterfactual_rank/proxy_success_fraction'],
            'oracle/benchmark_success_fraction': collection_metrics[
                'counterfactual_rank/benchmark_success_fraction'],
            'oracle/success_predicate_agreement': collection_metrics[
                'counterfactual_rank/success_predicate_agreement'],
        }
        for name, value in condition_metrics.items():
          suffix = name.split('/', 1)[-1]
          oracle_metrics[f'oracle/{condition}/{suffix}'] = float(value)
      if FLAGS.use_wandb and wandb is not None:
        wandb_oracle = {
            f'learner/{name}': value
            for name, value in oracle_metrics.items()}
        wandb_oracle['learner/env_steps'] = env_steps_done
        wandb.log(wandb_oracle)
      print(
          '  [counterfactual-oracle] '
          f"policy-r5={oracle_metrics.get('oracle/policy/repeat5/best_success_fraction', 0.0):.2f} "
          f"scripted-r5={oracle_metrics.get('oracle/scripted_contact/repeat5/best_success_fraction', 0.0):.2f}",
          flush=True)
      counterfactual_oracle_event += 1
      oracle_max_events = int(FLAGS.counterfactual_oracle_max_events)
      if oracle_max_events > 0 and counterfactual_oracle_event >= oracle_max_events:
        next_counterfactual_oracle_at = float('inf')
      else:
        next_counterfactual_oracle_at = (
            env_steps_done + counterfactual_oracle_interval)
      if FLAGS.profile_runtime:
        runtime_totals['counterfactual_oracle_seconds'] += (
            time.perf_counter() - oracle_started_at)

    if env_steps_done >= next_action_landscape_at:
      landscape_started_at = time.perf_counter()
      transitions = learner.last_transitions
      if transitions is None:
        raise RuntimeError(
            'Action-landscape event fired before replay transitions existed.')
      policy_params = learner.get_variables(['policy'])[0]
      critic_params = learner.get_variables(['critic'])[0]

      def _diagnostic_policy_action(observation, numpy_rng, stochastic):
        key = jax.random.PRNGKey(
            int(numpy_rng.integers(0, np.iinfo(np.int32).max)))
        observation_batch = jnp.asarray(observation)[None, :]
        distribution = networks.policy_network.apply(
            policy_params, observation_batch)
        sampler = networks.sample if stochastic else networks.sample_eval
        return np.asarray(sampler(distribution, key))[0]

      def _diagnostic_score_actions(observation, actions):
        if (counterfactual_rank_enabled
            and hasattr(learner, 'score_counterfactual_actions')):
          return np.asarray(
              learner.score_counterfactual_actions(observation, actions))
        if hasattr(learner, 'score_actions'):
          return np.asarray(learner.score_actions(observation, actions))
        observation_batch = jnp.repeat(
            jnp.asarray(observation)[None, :], actions.shape[0], axis=0)
        values = decomp_nets.apply_paired_score(
            critic_params['b_shared'], critic_params['h_phi'],
            critic_params['phi_task'], critic_params['psi'],
            observation_batch, jnp.asarray(actions))
        return np.asarray(values)

      print(
          f'  [action-landscape @ {env_steps_done}] running same-state '
          'counterfactual rollouts...', flush=True)
      landscape_metrics = (
          action_ranking_diagnostics.run_causal_action_ranking_probe(
              environment=action_landscape_env,
              obs_dim=config.obs_dim,
              replay_observations=np.asarray(transitions.observation),
              replay_actions=np.asarray(transitions.action),
              policy_action_fn=_diagnostic_policy_action,
              score_actions_fn=_diagnostic_score_actions,
              rng=action_landscape_rng,
              num_anchors=int(getattr(
                  continual_cfg, 'action_landscape_num_anchors', 1)),
              candidates_per_family=int(getattr(
                  continual_cfg,
                  'action_landscape_candidates_per_family', 4)),
              rollout_horizon=int(getattr(
                  continual_cfg, 'action_landscape_rollout_horizon', 25)),
              anchor_prefix_steps=int(getattr(
                  continual_cfg,
                  'action_landscape_anchor_prefix_steps', 20)),
              local_noise_std=float(getattr(
                  continual_cfg, 'action_landscape_local_noise_std', 0.10)),
              interaction_aware_anchor=bool(getattr(
                  continual_cfg,
                  'action_landscape_interaction_aware_anchor', False)),
              interaction_threshold=float(getattr(
                  continual_cfg,
                  'action_landscape_interaction_threshold', 0.09)),
              anchor_search_steps=int(getattr(
                  continual_cfg,
                  'action_landscape_anchor_search_steps', 200)),
              action_repeat=int(getattr(
                  continual_cfg, 'action_landscape_action_repeat', 1)),
              use_best_progress=bool(getattr(
                  continual_cfg, 'action_landscape_use_best_progress',
                  False)),
              success_threshold=float(getattr(
                  continual_cfg, 'action_landscape_success_threshold',
                  0.05)),
              success_mode=getattr(
                  continual_cfg, 'action_landscape_success_mode',
                  'goal_distance'),
          ))
      if FLAGS.use_wandb and wandb is not None:
        wandb_landscape = {
            f'learner/{name}': float(value)
            for name, value in landscape_metrics.items()}
        wandb_landscape['learner/env_steps'] = env_steps_done
        wandb.log(wandb_landscape)
      print(
          '  [action-landscape] '
          f"rho_aligned={landscape_metrics.get('action_landscape/aligned_score_vs_progress_spearman', float('nan')):.3f} "
          f"policy_score_pct={landscape_metrics.get('action_landscape/policy_score_percentile', float('nan')):.3f} "
          f"policy_outcome_pct={landscape_metrics.get('action_landscape/policy_outcome_percentile', float('nan')):.3f}",
          flush=True)
      next_action_landscape_at = (
          env_steps_done + action_landscape_interval)
      if FLAGS.profile_runtime:
        runtime_totals['action_landscape_seconds'] += (
            time.perf_counter() - landscape_started_at)

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
      if phase_control_enabled:
        phase_metrics = actor.get_and_reset_metrics()
        if FLAGS.use_wandb and wandb is not None:
          wandb.log({
              f'learner/{name}': float(value)
              for name, value in {
                  **phase_metrics, 'env_steps': env_steps_done}.items()})
      print(f'  Task {task_id} [{env_name}]: '
            f'{env_steps_done}/{train_steps} env steps '
            f'({episodes_done} episodes)', flush=True)
      if FLAGS.profile_runtime:
        elapsed = max(time.perf_counter() - runtime_started_at, 1e-9)
        diagnostic_seconds = sum(
            runtime_totals[name] for name in (
                'counterfactual_rank_seconds',
                'counterfactual_oracle_seconds',
                'action_landscape_seconds'))
        accounted = sum(runtime_totals.values())
        runtime_metrics = {
            **{f'runtime/{name}': value
               for name, value in runtime_totals.items()},
            'runtime/total_seconds': elapsed,
            'runtime/unaccounted_seconds': max(0.0, elapsed - accounted),
            'runtime/diagnostic_fraction': diagnostic_seconds / elapsed,
            'runtime/learner_fraction': (
                runtime_totals['learner_seconds'] / elapsed),
            'runtime/env_steps_per_second': env_steps_done / elapsed,
        }
        if FLAGS.use_wandb and wandb is not None:
          runtime_metrics['runtime/env_steps'] = env_steps_done
          wandb.log(runtime_metrics)
        print(
            '  [runtime] '
            f"actor={runtime_totals['actor_seconds']:.1f}s "
            f"learner={runtime_totals['learner_seconds']:.1f}s "
            f'diagnostics={diagnostic_seconds:.1f}s '
            f'total={elapsed:.1f}s',
            flush=True)
      next_log_at = env_steps_done + log_every_steps

    # Periodic evaluation (deterministic policy on current task)
    if env_steps_done >= next_evaluator_at:
      evaluation_started_at = time.perf_counter()
      eval_variable_client.update_and_wait()
      eval_successes = []
      eval_returns = []
      task58_episode_metrics = {}
      for _ in range(FLAGS.eval_episodes):
        ep_result = eval_loop.run_episode()
        eval_successes.append(ep_result.get('success', 0))
        eval_returns.append(float(ep_result.get('episode_return', 0)))
        if task58_stage_metrics_enabled:
          for name in (
              'legacy_success',
              'task_axis_success',
              'axis_rescued_success',
              'approach_success',
              'interaction_step_fraction',
              'minimum_hand_mechanism_distance',
              'mechanism_moved',
              'max_mechanism_axis_displacement',
              'max_task_axis_progress',
              'initial_task_axis_distance',
              'legacy_min_distance',
              'task_axis_min_distance',
              'success_reward_mismatch_steps'):
            task58_episode_metrics.setdefault(name, []).append(
                float(ep_result[name]))
      eval_success_rate = np.mean(eval_successes)
      eval_mean_return = np.mean(eval_returns)
      task58_eval_metrics = {
          name: float(np.mean(values))
          for name, values in task58_episode_metrics.items()
      }
      if task58_eval_metrics.get('success_reward_mismatch_steps', 0.0) > 0:
        raise RuntimeError(
            'Task-5/Task-8 trajectory scorer disagreed with the corrected '
            'wrapper reward during evaluation.')
      eval_phase_metrics = (
          eval_actor.get_and_reset_metrics()
          if phase_control_enabled else {})
      if FLAGS.use_wandb and wandb is not None:
        eval_log = {
            'evaluator/success_rate': eval_success_rate,
            'evaluator/mean_return': eval_mean_return,
            'evaluator/env_steps': env_steps_done,
        }
        eval_log.update({
            f'evaluator/{name}': float(value)
            for name, value in eval_phase_metrics.items()})
        eval_log.update({
            f'evaluator/task58/{name}': value
            for name, value in task58_eval_metrics.items()})
        wandb.log(eval_log)
      print(f'  [eval @ {env_steps_done}] success={eval_success_rate:.1%} '
            f'return={eval_mean_return:.1f}', flush=True)
      if task58_stage_metrics_enabled:
        print(
            '  [task58 stages] '
            f"legacy={task58_eval_metrics['legacy_success']:.1%} "
            f"axis={task58_eval_metrics['task_axis_success']:.1%} "
            f"approach={task58_eval_metrics['approach_success']:.1%} "
            f"moved={task58_eval_metrics['mechanism_moved']:.1%} "
            f"progress={task58_eval_metrics['max_task_axis_progress']:.4f}",
            flush=True)
      if FLAGS.profile_runtime:
        runtime_totals['evaluation_seconds'] += (
            time.perf_counter() - evaluation_started_at)
      next_evaluator_at = env_steps_done + eval_every

    # ---- RL representation metrics ----------------------------------------
    if env_steps_done >= next_metrics_frequent:
      rl_metrics_started_at = time.perf_counter()
      if env_steps_done >= next_metrics_occasional:
        level = 'occasional'
        next_metrics_occasional = env_steps_done + 5 * metrics_every
        next_metrics_frequent = env_steps_done + metrics_every
      else:
        level = 'frequent'
        next_metrics_frequent = env_steps_done + metrics_every

      try:
        transitions = learner.last_transitions
        # Two layouts:
        #   (a) persistent / CKA / reset critic: learner.q_params is the
        #       monolithic q-network pytree with sa_encoder / g_encoder
        #       modules. rl_metrics.compute_all_metrics consumes it via
        #       ``networks.repr_fn`` and ``networks.critic_hidden_repr_fn``.
        #   (b) decomposed critic: learner.q_params is None; the critic
        #       lives in five (or four when h_dyn is disabled) separate
        #       param groups. We build a tiny networks shim whose
        #       ``repr_fn(params, obs, action)`` calls
        #       decomp_nets.apply_score's component pieces, and we hand
        #       compute_all_metrics the bundle dict from
        #       ``learner.get_variables(['critic'])[0]``. The hidden-
        #       feature path (critic NRC2 / dormancy on hidden) is set to
        #       None so the function silently skips those entries on the
        #       decomposed critic.
        if transitions is not None:
          current_actor = learner.get_variables(['policy'])[0]
          bs = config.batch_size
          obs_sample = jnp.array(transitions.observation[:bs])
          act_sample = jnp.array(transitions.action[:bs])

          if learner.q_params is not None:
            metrics_networks = networks
            current_critic = learner.q_params
          else:
            # Decomposed layout: build / cache a shim.
            if not hasattr(learner, '_dcc_metrics_networks'):
              from types import SimpleNamespace
              _decomp = decomp_nets  # captured from enclosing scope
              def _repr_fn(params, obs, action, hidden=None):
                z_sa = _decomp.apply_sa_repr(
                    params['b_shared'], params['h_phi'],
                    params['phi_task'], obs, action)
                # ``params['psi']`` is either the bare psi params or the
                # bundle dict {psi, psi_proj}; decomp.apply_psi handles
                # both.
                z_g = _decomp.apply_psi(params['psi'], obs)
                return z_sa, z_g, None
              shim = SimpleNamespace(
                  repr_fn=_repr_fn,
                  critic_hidden_repr_fn=None,
                  actor_repr_fn=networks.actor_repr_fn,
              )
              try:
                setattr(learner, '_dcc_metrics_networks', shim)
              except Exception:
                pass
            metrics_networks = getattr(
                learner, '_dcc_metrics_networks', networks)
            current_critic = learner.get_variables(['critic'])[0]
          m = rl_metrics.compute_all_metrics(
              metrics_networks, current_actor, current_critic,
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
      if FLAGS.profile_runtime:
        runtime_totals['rl_metrics_seconds'] += (
            time.perf_counter() - rl_metrics_started_at)

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
              FLAGS.use_task_id, adapt_heads_only, actor_mode,
              dyn_aux_weight=FLAGS.dyn_aux_weight,
              phi_task_width=FLAGS.phi_task_width,
              phi_task_depth=FLAGS.phi_task_depth,
              rbc_config=(
                  _rbc_identity_config()
                  if critic_mode == 'rbc_decomposed' else None),
              in_trajectory_negative_repeats=
                  FLAGS.in_trajectory_negative_repeats,
              single_task=FLAGS.single_task,
              goal_conditioning_mode=FLAGS.goal_conditioning_mode,
              sawyer_success_mode=FLAGS.sawyer_success_mode)),
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
  if critic_mode in _DECOMPOSED_CRITIC_MODES:
    # There is no v_k actor pool or critic CKA pool. Plain DCC optionally
    # carries the actor, actor optimiser, and entropy state when
    # actor_mode='persistent'; all other decomposed-family modes reset the
    # actor. The shared critic groups and their optimiser states always carry.
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
    if critic_mode == 'decomposed' and actor_mode == 'persistent':
      out_dcc_policy_params = learner.policy_params
      out_dcc_policy_opt_state = learner.policy_opt_state
      out_dcc_alpha_params = learner.alpha_params
      out_dcc_alpha_opt_state = learner.alpha_opt_state
      actor_handoff = 'actor / actor optimizer / entropy state carried'
    else:
      out_dcc_policy_params = None
      out_dcc_policy_opt_state = None
      out_dcc_alpha_params = None
      out_dcc_alpha_opt_state = None
      actor_handoff = 'actor reinitialised next task'
    print(
        f'  [{critic_mode}] Carrying b_shared / h_phi / h_dyn / psi to task '
        f'{task_id + 1}; phi_task reinitialised; {actor_handoff}.',
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
    if action_landscape_env is not None:
      try:
        action_landscape_env.close()
      except Exception:
        pass
    if counterfactual_rank_env is not None:
      try:
        counterfactual_rank_env.close()
      except Exception:
        pass
    if counterfactual_validation_env is not None:
      try:
        counterfactual_validation_env.close()
      except Exception:
        pass
    for oracle_env in counterfactual_oracle_envs.values():
      try:
        oracle_env.close()
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
        out_dcc_policy_params, out_dcc_policy_opt_state,
        out_dcc_alpha_params, out_dcc_alpha_opt_state,
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
              FLAGS.use_task_id, adapt_heads_only, actor_mode,
              dyn_aux_weight=FLAGS.dyn_aux_weight,
              phi_task_width=FLAGS.phi_task_width,
              phi_task_depth=FLAGS.phi_task_depth,
              in_trajectory_negative_repeats=
                  FLAGS.in_trajectory_negative_repeats,
              single_task=FLAGS.single_task,
              goal_conditioning_mode=FLAGS.goal_conditioning_mode,
              sawyer_success_mode=FLAGS.sawyer_success_mode)),
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
                FLAGS.use_task_id, adapt_heads_only, actor_mode,
                dyn_aux_weight=FLAGS.dyn_aux_weight,
                phi_task_width=FLAGS.phi_task_width,
                phi_task_depth=FLAGS.phi_task_depth,
                in_trajectory_negative_repeats=
                    FLAGS.in_trajectory_negative_repeats,
                single_task=FLAGS.single_task,
                goal_conditioning_mode=FLAGS.goal_conditioning_mode,
                sawyer_success_mode=FLAGS.sawyer_success_mode)),
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
  if action_landscape_env is not None:
    try:
      action_landscape_env.close()
    except Exception:
      pass
  if counterfactual_rank_env is not None:
    try:
      counterfactual_rank_env.close()
    except Exception:
      pass
  if counterfactual_validation_env is not None:
    try:
      counterfactual_validation_env.close()
    except Exception:
      pass
  for oracle_env in counterfactual_oracle_envs.values():
    try:
      oracle_env.close()
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
      None, None,  # DCC policy params / opt_state
      None, None,  # DCC entropy params / opt_state
  )


# ---- main ----------------------------------------------------------------

def main(_):
  seed = FLAGS.seed

  if FLAGS.goal_conditioning_mode != 'full_state' and FLAGS.use_task_id:
    raise ValueError(
        f'{FLAGS.goal_conditioning_mode} currently requires --nouse_task_id '
        'because the '
        'historical TaskIDGymWrapper appends a non-contiguous task code to '
        'the goal. The single-task validity cells already disable task IDs.')

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
      shared_repr_scale=FLAGS.shared_repr_scale,
      phi_task_width=FLAGS.phi_task_width,
      phi_task_depth=FLAGS.phi_task_depth,
      combine_mode=FLAGS.combine_mode,
      goal_encoder_mode=FLAGS.goal_encoder_mode,
      in_trajectory_negative_repeats=
          FLAGS.in_trajectory_negative_repeats,
      interaction_weighted_relabeling=
          FLAGS.interaction_weighted_relabeling,
      interaction_threshold=FLAGS.interaction_threshold,
      interaction_bandwidth=FLAGS.interaction_bandwidth,
      interaction_weight_floor=FLAGS.interaction_weight_floor,
      action_effect_enabled=FLAGS.action_effect_enabled,
      action_effect_loss_weight=FLAGS.action_effect_loss_weight,
      action_effect_discount=FLAGS.action_effect_discount,
      action_effect_temperature=FLAGS.action_effect_temperature,
      action_effect_actor_weight=FLAGS.action_effect_actor_weight,
      action_effect_normalization_eps=
          FLAGS.action_effect_normalization_eps,
      action_effect_q_scale_ema_decay=
          FLAGS.action_effect_q_scale_ema_decay,
      action_effect_hidden_dim=FLAGS.action_effect_hidden_dim,
      action_effect_actor_mode=FLAGS.action_effect_actor_mode,
      action_effect_target_mode=FLAGS.action_effect_target_mode,
      outcome_horizon=FLAGS.outcome_horizon,
      outcome_success_threshold=FLAGS.outcome_success_threshold,
      outcome_progress_loss_weight=FLAGS.outcome_progress_loss_weight,
      outcome_success_loss_weight=FLAGS.outcome_success_loss_weight,
      outcome_success_actor_weight=FLAGS.outcome_success_actor_weight,
      outcome_progress_ema_decay=FLAGS.outcome_progress_ema_decay,
      outcome_progress_std_floor=FLAGS.outcome_progress_std_floor,
      success_bc_weight=FLAGS.success_bc_weight,
      success_buffer_capacity=FLAGS.success_buffer_capacity,
      success_bc_batch_size=FLAGS.success_bc_batch_size,
      counterfactual_rank_interval_steps=
          FLAGS.counterfactual_rank_interval_steps,
      counterfactual_rank_num_anchors=
          FLAGS.counterfactual_rank_num_anchors,
      counterfactual_rank_candidates_per_family=
          FLAGS.counterfactual_rank_candidates_per_family,
      counterfactual_rank_rollout_horizon=
          FLAGS.counterfactual_rank_rollout_horizon,
      counterfactual_rank_action_repeat=
          FLAGS.counterfactual_rank_action_repeat,
      counterfactual_rank_local_noise_std=
          FLAGS.counterfactual_rank_local_noise_std,
      counterfactual_rank_anchor_mode=
          FLAGS.counterfactual_rank_anchor_mode,
      counterfactual_rank_anchor_search_steps=
          FLAGS.counterfactual_rank_anchor_search_steps,
      counterfactual_rank_interaction_threshold=
          FLAGS.counterfactual_rank_interaction_threshold,
      counterfactual_rank_contact_gain=
          FLAGS.counterfactual_rank_contact_gain,
      counterfactual_rank_success_threshold=
          FLAGS.counterfactual_rank_success_threshold,
      counterfactual_rank_success_bonus=
          FLAGS.counterfactual_rank_success_bonus,
      counterfactual_rank_min_outcome_gap=
          FLAGS.counterfactual_rank_min_outcome_gap,
      counterfactual_rank_buffer_capacity=
          FLAGS.counterfactual_rank_buffer_capacity,
      counterfactual_rank_batch_anchors=
          FLAGS.counterfactual_rank_batch_anchors,
      counterfactual_rank_updates_per_event=
          FLAGS.counterfactual_rank_updates_per_event,
      counterfactual_rank_pairwise_temperature=
          FLAGS.counterfactual_rank_pairwise_temperature,
      counterfactual_rank_l2_weight=
          FLAGS.counterfactual_rank_l2_weight,
      counterfactual_rank_validation_anchors=
          FLAGS.counterfactual_rank_validation_anchors,
      counterfactual_rank_success_mode=
          FLAGS.counterfactual_rank_success_mode,
      counterfactual_rank_actor_enabled=
          FLAGS.counterfactual_rank_actor_enabled,
      counterfactual_oracle_interval_steps=
          FLAGS.counterfactual_oracle_interval_steps,
      counterfactual_oracle_num_anchors=
          FLAGS.counterfactual_oracle_num_anchors,
      phase_gated_control=FLAGS.phase_gated_control,
      phase_gate_reach_mode=FLAGS.phase_gate_reach_mode,
      phase_gate_interaction_threshold=
          FLAGS.phase_gate_interaction_threshold,
      phase_gate_chunk_length=FLAGS.phase_gate_chunk_length,
      phase_gate_num_candidates=FLAGS.phase_gate_num_candidates,
      phase_gate_local_noise_std=FLAGS.phase_gate_local_noise_std,
      phase_gate_contact_gain=FLAGS.phase_gate_contact_gain,
      bellman_loss_weight=FLAGS.bellman_loss_weight,
      bellman_residual_l2_weight=FLAGS.bellman_residual_l2_weight,
      bellman_discount=FLAGS.bellman_discount,
      bellman_tau=FLAGS.bellman_tau,
      bellman_hidden_dim=FLAGS.bellman_hidden_dim,
      her_reward_threshold=FLAGS.her_reward_threshold,
      step_penalty_reward=FLAGS.step_penalty_reward,
      dcc_sac_q_loss_weight=FLAGS.dcc_sac_q_loss_weight,
      dcc_sac_q_learning_rate=FLAGS.dcc_sac_q_learning_rate,
      dcc_sac_discount=FLAGS.dcc_sac_discount,
      dcc_sac_tau=FLAGS.dcc_sac_tau,
      dcc_sac_q_hidden_dim=FLAGS.dcc_sac_q_hidden_dim,
      dcc_sac_beta_max=FLAGS.dcc_sac_beta_max,
      dcc_sac_q_warmup_updates=FLAGS.dcc_sac_q_warmup_updates,
      dcc_sac_q_ramp_updates=FLAGS.dcc_sac_q_ramp_updates,
      dcc_sac_td_error_threshold=FLAGS.dcc_sac_td_error_threshold,
      dcc_sac_twin_disagreement_threshold=
          FLAGS.dcc_sac_twin_disagreement_threshold,
      dcc_sac_ema_decay=FLAGS.dcc_sac_ema_decay,
      dcc_sac_candidate_actions=FLAGS.dcc_sac_candidate_actions,
      dcc_sac_normalization_eps=FLAGS.dcc_sac_normalization_eps,
      dcc_sac_correction_clip=FLAGS.dcc_sac_correction_clip,
      action_contrast_weight=FLAGS.action_contrast_weight,
      action_contrast_temperature=FLAGS.action_contrast_temperature,
      action_contrast_batch_size=FLAGS.action_contrast_batch_size,
      shortcut_diagnostic_interval=FLAGS.shortcut_diagnostic_interval,
      shortcut_diagnostic_batch_size=FLAGS.shortcut_diagnostic_batch_size,
      shortcut_candidate_actions=FLAGS.shortcut_candidate_actions,
      action_landscape_diagnostic_interval_steps=
          FLAGS.action_landscape_diagnostic_interval_steps,
      action_landscape_num_anchors=FLAGS.action_landscape_num_anchors,
      action_landscape_candidates_per_family=
          FLAGS.action_landscape_candidates_per_family,
      action_landscape_rollout_horizon=
          FLAGS.action_landscape_rollout_horizon,
      action_landscape_anchor_prefix_steps=
          FLAGS.action_landscape_anchor_prefix_steps,
      action_landscape_local_noise_std=
          FLAGS.action_landscape_local_noise_std,
      action_landscape_interaction_aware_anchor=
          FLAGS.action_landscape_interaction_aware_anchor,
      action_landscape_anchor_search_steps=
          FLAGS.action_landscape_anchor_search_steps,
      action_landscape_interaction_threshold=
          FLAGS.action_landscape_interaction_threshold,
      action_landscape_action_repeat=FLAGS.action_landscape_action_repeat,
      action_landscape_use_best_progress=
          FLAGS.action_landscape_use_best_progress,
      action_landscape_success_threshold=
          FLAGS.action_landscape_success_threshold,
      action_landscape_success_mode=FLAGS.action_landscape_success_mode,
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
  prev_dcc_policy_params = None
  prev_dcc_policy_opt_state = None
  prev_dcc_alpha_params = None
  prev_dcc_alpha_opt_state = None

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
          actor_mode=FLAGS.actor_mode,
          dyn_aux_weight=FLAGS.dyn_aux_weight,
          phi_task_width=FLAGS.phi_task_width,
          phi_task_depth=FLAGS.phi_task_depth,
          rbc_config=(
              _rbc_identity_config()
              if FLAGS.critic_mode == 'rbc_decomposed' else None),
          in_trajectory_negative_repeats=
              FLAGS.in_trajectory_negative_repeats,
          single_task=FLAGS.single_task,
          goal_conditioning_mode=FLAGS.goal_conditioning_mode,
          sawyer_success_mode=FLAGS.sawyer_success_mode)
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
    resume_checkpoint_dir = (
        FLAGS.resume_checkpoint_dir or FLAGS.checkpoint_dir)
    ckpt = load_ckpt(resume_checkpoint_dir, start_task - 1, seed,
                      critic_mode=FLAGS.critic_mode,
                      use_task_id=FLAGS.use_task_id,
                      adapt_heads_only=FLAGS.adapt_heads_only,
                      actor_mode=FLAGS.actor_mode,
                      dyn_aux_weight=FLAGS.dyn_aux_weight,
                      phi_task_width=FLAGS.phi_task_width,
                      phi_task_depth=FLAGS.phi_task_depth,
                      rbc_config=(
                          _rbc_identity_config()
                          if FLAGS.critic_mode == 'rbc_decomposed' else None),
        in_trajectory_negative_repeats=
            FLAGS.in_trajectory_negative_repeats,
        single_task=FLAGS.single_task,
        goal_conditioning_mode=FLAGS.goal_conditioning_mode,
        sawyer_success_mode=FLAGS.sawyer_success_mode)
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
    if FLAGS.critic_mode in _DECOMPOSED_CRITIC_MODES:
      prev_b_shared_params = ckpt.get('decomposed_b_shared_params')
      prev_b_shared_opt_state = ckpt.get('decomposed_b_shared_opt_state')
      prev_h_phi_params = ckpt.get('decomposed_h_phi_params')
      prev_h_phi_opt_state = ckpt.get('decomposed_h_phi_opt_state')
      prev_h_dyn_params = ckpt.get('decomposed_h_dyn_params')
      prev_h_dyn_opt_state = ckpt.get('decomposed_h_dyn_opt_state')
      prev_psi_params = ckpt.get('decomposed_psi_params')
      prev_psi_opt_state = ckpt.get('decomposed_psi_opt_state')
      if FLAGS.critic_mode == 'decomposed' and FLAGS.actor_mode == 'persistent':
        prev_dcc_policy_params = ckpt.get('decomposed_policy_params')
        prev_dcc_policy_opt_state = ckpt.get(
            'decomposed_policy_opt_state')
        prev_dcc_alpha_params = ckpt.get('decomposed_alpha_params')
        prev_dcc_alpha_opt_state = ckpt.get('decomposed_alpha_opt_state')

  for task_id in range(start_task, num_tasks):
    env_name = task_sequence[task_id]
    params['env_name'] = env_name
    goal_start, goal_end = goal_semantics.resolve_goal_slice(
        FLAGS.goal_conditioning_mode, env_name)
    params['start_index'] = goal_start
    params['end_index'] = goal_end

    # Per-task override for the dynamics auxiliary weight.
    # If --dyn_aux_after_task0 is set (>= 0) and task_id >= 1, use that
    # value instead of --dyn_aux_weight. This is the C2b ablation
    # (dynamics aux only at task 0, off afterward). See
    # docs/2026-05-14_c2_ldyn_interpretation.md.
    if FLAGS.dyn_aux_after_task0 >= 0.0 and task_id >= 1:
      effective_dyn_aux_weight = FLAGS.dyn_aux_after_task0
    else:
      effective_dyn_aux_weight = FLAGS.dyn_aux_weight
    continual_cfg.dyn_aux_weight = effective_dyn_aux_weight

    print(f'\n{"="*60}', flush=True)
    print(f'Task {task_id}/{num_tasks - 1}: {env_name}', flush=True)
    phase = 'BASE' if task_id == 0 else 'CONTINUAL'
    steps = continual_cfg.base_steps if task_id == 0 else continual_cfg.steps_per_task
    print(f'Phase: {phase} | Steps: {steps} | Pool: {len(pool)}/{continual_cfg.k_max} | dyn_aux_weight={effective_dyn_aux_weight}', flush=True)
    print(f'Critic: {FLAGS.critic_mode} | Task ID: {FLAGS.use_task_id} | '
          f'Heads only: {FLAGS.adapt_heads_only} | '
          f'Encoder base: {FLAGS.encoder_from_base}', flush=True)
    print(f'Actor mode: {FLAGS.actor_mode} | '
          f'Eval: {FLAGS.eval_episodes}ep, K={FLAGS.k_sample_k} | '
          f'20-task: {FLAGS.use_20_tasks}', flush=True)
    print(f'Goal contract: {FLAGS.goal_conditioning_mode} '
          f'(state[{goal_start}:{goal_end}])', flush=True)
    print(f'Sawyer success semantics: {FLAGS.sawyer_success_mode}', flush=True)
    print(f'{"="*60}\n', flush=True)

    config = contrastive.ContrastiveConfig(**params)

    # Initialise W&B run per task (matching lp_continual_contrastive.py).
    # WandbLogger in default.py assumes wandb.init() has already been called;
    # without this call all wandb.log() silently fail.
    if FLAGS.use_wandb and wandb is not None:
      wandb.init(
          project=FLAGS.wandb_project,
          group=FLAGS.wandb_group,
          config={**params, 'task_id': task_id, 'env_name': env_name,
                  'num_tasks': num_tasks, 'k_max': continual_cfg.k_max,
                  'git_commit': _git_commit_sha(),
                  'critic_mode': FLAGS.critic_mode,
                  'use_task_id': FLAGS.use_task_id,
                  'adapt_heads_only': FLAGS.adapt_heads_only,
                  'encoder_from_base': FLAGS.encoder_from_base,
                  'use_20_tasks': FLAGS.use_20_tasks,
                  'actor_mode': FLAGS.actor_mode,
                  'goal_conditioning_mode': FLAGS.goal_conditioning_mode,
                  'sawyer_success_mode': FLAGS.sawyer_success_mode,
                  'profile_runtime': FLAGS.profile_runtime,
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
                  'neg_bank_max_tasks': FLAGS.neg_bank_max_tasks,
                  'dyn_aux_weight': FLAGS.dyn_aux_weight,
                  'dyn_aux_after_task0': FLAGS.dyn_aux_after_task0,
                  'shared_repr_scale': FLAGS.shared_repr_scale,
                  'resume_checkpoint_dir': FLAGS.resume_checkpoint_dir,
                  'phi_task_width': FLAGS.phi_task_width,
                  'phi_task_depth': FLAGS.phi_task_depth,
                  'combine_mode': FLAGS.combine_mode,
                  'goal_encoder_mode': FLAGS.goal_encoder_mode,
                  'in_trajectory_negative_repeats':
                      FLAGS.in_trajectory_negative_repeats,
                  'interaction_weighted_relabeling':
                      FLAGS.interaction_weighted_relabeling,
                  'interaction_threshold': FLAGS.interaction_threshold,
                  'interaction_bandwidth': FLAGS.interaction_bandwidth,
                  'interaction_weight_floor': FLAGS.interaction_weight_floor,
                  'action_effect_enabled': FLAGS.action_effect_enabled,
                  'action_effect_loss_weight':
                      FLAGS.action_effect_loss_weight,
                  'action_effect_discount': FLAGS.action_effect_discount,
                  'action_effect_temperature':
                      FLAGS.action_effect_temperature,
                  'action_effect_actor_weight':
                      FLAGS.action_effect_actor_weight,
                  'action_effect_normalization_eps':
                      FLAGS.action_effect_normalization_eps,
                  'action_effect_q_scale_ema_decay':
                      FLAGS.action_effect_q_scale_ema_decay,
                  'action_effect_hidden_dim':
                      FLAGS.action_effect_hidden_dim,
                  'action_effect_actor_mode':
                      FLAGS.action_effect_actor_mode,
                  'action_effect_target_mode':
                      FLAGS.action_effect_target_mode,
                  'outcome_horizon': FLAGS.outcome_horizon,
                  'outcome_success_threshold':
                      FLAGS.outcome_success_threshold,
                  'outcome_progress_loss_weight':
                      FLAGS.outcome_progress_loss_weight,
                  'outcome_success_loss_weight':
                      FLAGS.outcome_success_loss_weight,
                  'outcome_success_actor_weight':
                      FLAGS.outcome_success_actor_weight,
                  'outcome_progress_ema_decay':
                      FLAGS.outcome_progress_ema_decay,
                  'outcome_progress_std_floor':
                      FLAGS.outcome_progress_std_floor,
                  'success_bc_weight': FLAGS.success_bc_weight,
                  'success_buffer_capacity':
                      FLAGS.success_buffer_capacity,
                  'success_bc_batch_size': FLAGS.success_bc_batch_size,
                  'counterfactual_rank_interval_steps':
                      FLAGS.counterfactual_rank_interval_steps,
                  'counterfactual_rank_num_anchors':
                      FLAGS.counterfactual_rank_num_anchors,
                  'counterfactual_rank_candidates_per_family':
                      FLAGS.counterfactual_rank_candidates_per_family,
                  'counterfactual_rank_rollout_horizon':
                      FLAGS.counterfactual_rank_rollout_horizon,
                  'counterfactual_rank_action_repeat':
                      FLAGS.counterfactual_rank_action_repeat,
                  'counterfactual_rank_local_noise_std':
                      FLAGS.counterfactual_rank_local_noise_std,
                  'counterfactual_rank_anchor_mode':
                      FLAGS.counterfactual_rank_anchor_mode,
                  'counterfactual_rank_anchor_search_steps':
                      FLAGS.counterfactual_rank_anchor_search_steps,
                  'counterfactual_rank_interaction_threshold':
                      FLAGS.counterfactual_rank_interaction_threshold,
                  'counterfactual_rank_contact_gain':
                      FLAGS.counterfactual_rank_contact_gain,
                  'counterfactual_rank_success_threshold':
                      FLAGS.counterfactual_rank_success_threshold,
                  'counterfactual_rank_success_bonus':
                      FLAGS.counterfactual_rank_success_bonus,
                  'counterfactual_rank_min_outcome_gap':
                      FLAGS.counterfactual_rank_min_outcome_gap,
                  'counterfactual_rank_buffer_capacity':
                      FLAGS.counterfactual_rank_buffer_capacity,
                  'counterfactual_rank_batch_anchors':
                      FLAGS.counterfactual_rank_batch_anchors,
                  'counterfactual_rank_updates_per_event':
                      FLAGS.counterfactual_rank_updates_per_event,
                  'counterfactual_rank_pairwise_temperature':
                      FLAGS.counterfactual_rank_pairwise_temperature,
                  'counterfactual_rank_l2_weight':
                      FLAGS.counterfactual_rank_l2_weight,
                  'counterfactual_rank_validation_anchors':
                      FLAGS.counterfactual_rank_validation_anchors,
                  'counterfactual_rank_success_mode':
                      FLAGS.counterfactual_rank_success_mode,
                  'counterfactual_rank_actor_enabled':
                      FLAGS.counterfactual_rank_actor_enabled,
                  'counterfactual_oracle_interval_steps':
                      FLAGS.counterfactual_oracle_interval_steps,
                  'counterfactual_oracle_num_anchors':
                      FLAGS.counterfactual_oracle_num_anchors,
                  'counterfactual_oracle_condition_set':
                      FLAGS.counterfactual_oracle_condition_set,
                  'counterfactual_oracle_max_events':
                      FLAGS.counterfactual_oracle_max_events,
                  'phase_gated_control': FLAGS.phase_gated_control,
                  'phase_gate_reach_mode': FLAGS.phase_gate_reach_mode,
                  'phase_gate_interaction_threshold':
                      FLAGS.phase_gate_interaction_threshold,
                  'phase_gate_chunk_length':
                      FLAGS.phase_gate_chunk_length,
                  'phase_gate_num_candidates':
                      FLAGS.phase_gate_num_candidates,
                  'phase_gate_local_noise_std':
                      FLAGS.phase_gate_local_noise_std,
                  'phase_gate_contact_gain':
                      FLAGS.phase_gate_contact_gain,
                  'bellman_loss_weight': FLAGS.bellman_loss_weight,
                  'bellman_residual_l2_weight':
                      FLAGS.bellman_residual_l2_weight,
                  'bellman_discount': FLAGS.bellman_discount,
                  'bellman_tau': FLAGS.bellman_tau,
                  'bellman_hidden_dim': FLAGS.bellman_hidden_dim,
                  'her_reward_threshold': FLAGS.her_reward_threshold,
                  'step_penalty_reward': FLAGS.step_penalty_reward,
                  'dcc_sac_q_loss_weight': FLAGS.dcc_sac_q_loss_weight,
                  'dcc_sac_q_learning_rate': FLAGS.dcc_sac_q_learning_rate,
                  'dcc_sac_discount': FLAGS.dcc_sac_discount,
                  'dcc_sac_tau': FLAGS.dcc_sac_tau,
                  'dcc_sac_q_hidden_dim': FLAGS.dcc_sac_q_hidden_dim,
                  'dcc_sac_beta_max': FLAGS.dcc_sac_beta_max,
                  'dcc_sac_q_warmup_updates':
                      FLAGS.dcc_sac_q_warmup_updates,
                  'dcc_sac_q_ramp_updates': FLAGS.dcc_sac_q_ramp_updates,
                  'dcc_sac_td_error_threshold':
                      FLAGS.dcc_sac_td_error_threshold,
                  'dcc_sac_twin_disagreement_threshold':
                      FLAGS.dcc_sac_twin_disagreement_threshold,
                  'dcc_sac_ema_decay': FLAGS.dcc_sac_ema_decay,
                  'dcc_sac_candidate_actions':
                      FLAGS.dcc_sac_candidate_actions,
                  'dcc_sac_normalization_eps':
                      FLAGS.dcc_sac_normalization_eps,
                  'dcc_sac_correction_clip': FLAGS.dcc_sac_correction_clip,
                  'action_contrast_weight': FLAGS.action_contrast_weight,
                  'action_contrast_temperature':
                      FLAGS.action_contrast_temperature,
                  'action_contrast_batch_size':
                      FLAGS.action_contrast_batch_size,
                  'shortcut_diagnostic_interval':
                      FLAGS.shortcut_diagnostic_interval,
                  'shortcut_diagnostic_batch_size':
                      FLAGS.shortcut_diagnostic_batch_size,
                  'shortcut_candidate_actions':
                      FLAGS.shortcut_candidate_actions,
                  'action_landscape_diagnostic_interval_steps':
                      FLAGS.action_landscape_diagnostic_interval_steps,
                  'action_landscape_num_anchors':
                      FLAGS.action_landscape_num_anchors,
                  'action_landscape_candidates_per_family':
                      FLAGS.action_landscape_candidates_per_family,
                  'action_landscape_rollout_horizon':
                      FLAGS.action_landscape_rollout_horizon,
                  'action_landscape_anchor_prefix_steps':
                      FLAGS.action_landscape_anchor_prefix_steps,
                  'action_landscape_local_noise_std':
                      FLAGS.action_landscape_local_noise_std,
                  'action_landscape_interaction_aware_anchor':
                      FLAGS.action_landscape_interaction_aware_anchor,
                  'action_landscape_anchor_search_steps':
                      FLAGS.action_landscape_anchor_search_steps,
                  'action_landscape_interaction_threshold':
                      FLAGS.action_landscape_interaction_threshold,
                  'action_landscape_action_repeat':
                      FLAGS.action_landscape_action_repeat,
                  'action_landscape_use_best_progress':
                      FLAGS.action_landscape_use_best_progress,
                  'action_landscape_success_threshold':
                      FLAGS.action_landscape_success_threshold,
                  'action_landscape_success_mode':
                      FLAGS.action_landscape_success_mode,
                  'post_task_eval_scope':
                      FLAGS.post_task_eval_scope},
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
     prev_psi_params, prev_psi_opt_state,
     prev_dcc_policy_params, prev_dcc_policy_opt_state,
     prev_dcc_alpha_params, prev_dcc_alpha_opt_state) = train_single_task(
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
        prev_dcc_policy_params=prev_dcc_policy_params,
        prev_dcc_policy_opt_state=prev_dcc_policy_opt_state,
        prev_dcc_alpha_params=prev_dcc_alpha_params,
        prev_dcc_alpha_opt_state=prev_dcc_alpha_opt_state,
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
        'goal_conditioning_mode': FLAGS.goal_conditioning_mode,
        'sawyer_success_mode': FLAGS.sawyer_success_mode,
        'goal_start_index': config.start_index,
        'goal_end_index': config.end_index,
        'in_trajectory_negative_repeats':
            FLAGS.in_trajectory_negative_repeats,
    }
    if FLAGS.critic_mode == 'cka':
      ckpt_data['q_base'] = q_base
      ckpt_data['critic_pool_vectors'] = critic_pool.state_dict()
    if FLAGS.critic_mode in _DECOMPOSED_CRITIC_MODES:
      ckpt_data['decomposed_b_shared_params'] = prev_b_shared_params
      ckpt_data['decomposed_b_shared_opt_state'] = prev_b_shared_opt_state
      ckpt_data['decomposed_h_phi_params'] = prev_h_phi_params
      ckpt_data['decomposed_h_phi_opt_state'] = prev_h_phi_opt_state
      ckpt_data['decomposed_h_dyn_params'] = prev_h_dyn_params
      ckpt_data['decomposed_h_dyn_opt_state'] = prev_h_dyn_opt_state
      ckpt_data['decomposed_psi_params'] = prev_psi_params
      ckpt_data['decomposed_psi_opt_state'] = prev_psi_opt_state
      if FLAGS.critic_mode == 'decomposed':
        ckpt_data['decomposed_policy_params'] = prev_dcc_policy_params
        ckpt_data['decomposed_policy_opt_state'] = (
            prev_dcc_policy_opt_state)
        ckpt_data['decomposed_alpha_params'] = prev_dcc_alpha_params
        ckpt_data['decomposed_alpha_opt_state'] = prev_dcc_alpha_opt_state
    save_ckpt(FLAGS.checkpoint_dir, task_id, seed, ckpt_data,
              critic_mode=FLAGS.critic_mode, use_task_id=FLAGS.use_task_id,
              adapt_heads_only=FLAGS.adapt_heads_only,
              actor_mode=FLAGS.actor_mode,
              dyn_aux_weight=FLAGS.dyn_aux_weight,
              phi_task_width=FLAGS.phi_task_width,
              phi_task_depth=FLAGS.phi_task_depth,
              rbc_config=(
                  _rbc_identity_config()
                  if FLAGS.critic_mode == 'rbc_decomposed' else None),
              in_trajectory_negative_repeats=
                  FLAGS.in_trajectory_negative_repeats,
              single_task=FLAGS.single_task,
              goal_conditioning_mode=FLAGS.goal_conditioning_mode,
              sawyer_success_mode=FLAGS.sawyer_success_mode)

    # ---- configurable task-boundary evaluation ---------------------------
    if (
        FLAGS.eval_episodes > 0
        and FLAGS.post_task_eval_scope != 'none'):
      if FLAGS.post_task_eval_scope == 'current':
        eval_task_ids = [task_id]
        print('\n  Evaluating only the task just trained...', flush=True)
      else:
        eval_task_ids = list(range(task_id + 1))
        print('\n  Evaluating on all tasks seen so far...', flush=True)
      eval_results = {}
      for eval_tid in eval_task_ids:
        eval_env_name_i = task_sequence[eval_tid]
        sr = evaluate_on_task(
            eval_env_name_i, eval_tid, composed_policy, prev_q, config,
            continual_cfg, seed,
            num_episodes=FLAGS.eval_episodes,
            k_sample_k=FLAGS.k_sample_k)
        eval_results[eval_env_name_i] = sr
        print(
            f'    Task {eval_tid} [{eval_env_name_i}]: {sr:.1%}',
            flush=True)
      mean_sr = np.mean(list(eval_results.values()))
      print(f'    Mean success: {mean_sr:.1%}', flush=True)
      if FLAGS.use_wandb and wandb is not None:
        wandb_eval = {
            f'eval/{name}': sr for name, sr in eval_results.items()}
        wandb_eval['eval/mean_success'] = mean_sr
        wandb_eval['eval/num_tasks_evaluated'] = len(eval_task_ids)
        wandb_eval['eval/scope_all_seen'] = float(
            FLAGS.post_task_eval_scope == 'all_seen')
        wandb.log(wandb_eval)

    # Close the W&B run for this task before starting the next one
    if FLAGS.use_wandb and wandb is not None:
      wandb.finish()

  print(f'\nAll {num_tasks} tasks complete.', flush=True)


if __name__ == '__main__':
  app.run(main)
