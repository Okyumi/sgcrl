"""Periodic diagnostics for contrastive shortcut learning and action sensitivity.

The diagnostics in this module are deliberately kept outside the learner's
hot update path.  They answer two separate questions:

1. Can the contrastive classifier keep solving its batch-classification
   objective when the action is shuffled or zeroed?
2. For a fixed state and goal, do DCC and Bellman Q meaningfully rank
   different candidate actions?

The returned function is JIT compiled once and should be called only at the
configured diagnostic interval.
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp


def _rankdata(values):
  """Ordinal ranks along the last axis (ties are rare for network scores)."""
  order = jnp.argsort(values, axis=-1)
  return jnp.argsort(order, axis=-1).astype(jnp.float32)


def _spearman(values_a, values_b, axis=-1, eps=1e-8):
  rank_a = _rankdata(values_a)
  rank_b = _rankdata(values_b)
  rank_a = rank_a - jnp.mean(rank_a, axis=axis, keepdims=True)
  rank_b = rank_b - jnp.mean(rank_b, axis=axis, keepdims=True)
  numerator = jnp.sum(rank_a * rank_b, axis=axis)
  denominator = jnp.sqrt(
      jnp.sum(jnp.square(rank_a), axis=axis)
      * jnp.sum(jnp.square(rank_b), axis=axis))
  return numerator / jnp.maximum(denominator, eps)


def _categorical_accuracy(logits):
  labels = jnp.arange(logits.shape[0])
  return jnp.mean(
      (jnp.argmax(logits, axis=1) == labels).astype(jnp.float32))


def make_shortcut_diagnostic_fn(
    *,
    decomp_nets,
    policy_network,
    sample_fn,
    obs_dim: int,
    start_index: int,
    end_index: int,
    diagnostic_batch_size: int = 32,
    candidate_actions: int = 16,
    q_network: Optional[object] = None,
):
  """Build a periodic diagnostic function.

  When q_network is None, only DCC metrics are returned.  Otherwise the same
  fixed-state candidate actions are scored by the independent twin Q critic,
  which makes the DCC-versus-Q ranking comparison directly matched.
  """
  diagnostic_batch_size = int(diagnostic_batch_size)
  candidate_actions = int(candidate_actions)
  has_q = q_network is not None

  def _achieved(state):
    if end_index == -1:
      return state[:, start_index:]
    return state[:, start_index:end_index]

  def diagnostic(
      b_shared_params,
      h_phi_params,
      phi_task_params,
      psi_params,
      policy_params,
      q_params,
      transitions,
      key,
  ):
    n = min(diagnostic_batch_size, transitions.action.shape[0])
    obs = transitions.observation[:n]
    next_obs = transitions.next_observation[:n]
    replay_action = transitions.action[:n]
    shuffled_action = jnp.roll(replay_action, 1, axis=0)
    zero_action = jnp.zeros_like(replay_action)

    def paired(actions):
      return decomp_nets.apply_paired_score(
          b_shared_params, h_phi_params, phi_task_params, psi_params,
          obs, actions)

    logits = decomp_nets.apply_score(
        b_shared_params, h_phi_params, phi_task_params, psi_params,
        obs, replay_action)
    shuffled_logits = decomp_nets.apply_score(
        b_shared_params, h_phi_params, phi_task_params, psi_params,
        obs, shuffled_action)
    zero_logits = decomp_nets.apply_score(
        b_shared_params, h_phi_params, phi_task_params, psi_params,
        obs, zero_action)

    # Coordinate interventions on the goal side.  Unified Sawyer states use
    # hand xyz at 0:3, gripper opening at 3:4, and the manipulated mechanism
    # or object at 4:7.  Shuffling one block across batch rows destroys that
    # cue while leaving all other goal coordinates untouched.  A large
    # categorical-accuracy drop identifies reliance on that block; it does
    # not by itself establish causality for control because the mixed goal is
    # off-manifold.
    state = obs[:, :obs_dim]
    goal = obs[:, obs_dim:]

    def coordinate_shuffled_logits(start, end):
      shuffled_goal = goal.at[:, start:end].set(
          jnp.roll(goal[:, start:end], 1, axis=0))
      shuffled_obs = jnp.concatenate([state, shuffled_goal], axis=-1)
      return decomp_nets.apply_score(
          b_shared_params, h_phi_params, phi_task_params, psi_params,
          shuffled_obs, replay_action)

    hand_goal_logits = coordinate_shuffled_logits(0, 3)
    gripper_goal_logits = coordinate_shuffled_logits(3, 4)
    mechanism_goal_logits = coordinate_shuffled_logits(4, 7)

    categorical_accuracy = _categorical_accuracy(logits)
    shuffled_accuracy = _categorical_accuracy(shuffled_logits)
    zero_accuracy = _categorical_accuracy(zero_logits)
    hand_goal_accuracy = _categorical_accuracy(hand_goal_logits)
    gripper_goal_accuracy = _categorical_accuracy(gripper_goal_logits)
    mechanism_goal_accuracy = _categorical_accuracy(mechanism_goal_logits)

    paired_replay = paired(replay_action)
    paired_shuffled = paired(shuffled_action)
    action_delta = paired_replay - paired_shuffled

    key_policy, key_uniform, key_grad = jax.random.split(key, 3)
    dist = policy_network.apply(policy_params, obs)
    policy_keys = jax.random.split(key_policy, candidate_actions)
    policy_actions = jax.vmap(lambda k: sample_fn(dist, k))(policy_keys)
    uniform_actions = jax.random.uniform(
        key_uniform,
        shape=(candidate_actions,) + replay_action.shape,
        minval=-1.0,
        maxval=1.0)

    dcc_policy_scores = jax.vmap(paired)(policy_actions)
    dcc_uniform_scores = jax.vmap(paired)(uniform_actions)

    def score_mean(actions):
      return jnp.mean(paired(actions))

    action_gradient = jax.grad(score_mean)(replay_action)

    state = obs[:, :obs_dim]
    next_state = next_obs[:, :obs_dim]
    goal = obs[:, obs_dim:]
    achieved = _achieved(state)
    achieved_next = _achieved(next_state)
    progress = (
        jnp.linalg.norm(achieved - goal, axis=-1)
        - jnp.linalg.norm(achieved_next - goal, axis=-1))
    dcc_progress_spearman = _spearman(
        paired_replay[None, :], progress[None, :])[0]

    eye = jnp.eye(n)
    positive = jnp.diag(logits)
    negative_mean = (
        jnp.sum(logits * (1.0 - eye))
        / jnp.maximum(jnp.sum(1.0 - eye), 1.0))

    metrics = {
        'shortcut/categorical_accuracy': categorical_accuracy,
        'shortcut/action_shuffled_categorical_accuracy': shuffled_accuracy,
        'shortcut/zero_action_categorical_accuracy': zero_accuracy,
        'shortcut/action_shuffle_retention': (
            shuffled_accuracy / jnp.maximum(categorical_accuracy, 1e-6)),
        'shortcut/zero_action_retention': (
            zero_accuracy / jnp.maximum(categorical_accuracy, 1e-6)),
        'shortcut/hand_goal_shuffled_categorical_accuracy':
            hand_goal_accuracy,
        'shortcut/gripper_goal_shuffled_categorical_accuracy':
            gripper_goal_accuracy,
        'shortcut/mechanism_goal_shuffled_categorical_accuracy':
            mechanism_goal_accuracy,
        'shortcut/hand_goal_shuffle_drop': (
            categorical_accuracy - hand_goal_accuracy),
        'shortcut/gripper_goal_shuffle_drop': (
            categorical_accuracy - gripper_goal_accuracy),
        'shortcut/mechanism_goal_shuffle_drop': (
            categorical_accuracy - mechanism_goal_accuracy),
        'shortcut/logit_saturation_fraction': jnp.mean(
            (jnp.abs(logits) > 10.0).astype(jnp.float32)),
        'shortcut/positive_negative_margin': (
            jnp.mean(positive) - negative_mean),
        'action/dcc_shuffle_delta_rms': jnp.sqrt(
            jnp.mean(jnp.square(action_delta))),
        'action/dcc_shuffle_delta_abs': jnp.mean(jnp.abs(action_delta)),
        'action/dcc_candidate_std_policy': jnp.mean(
            jnp.std(dcc_policy_scores, axis=0)),
        'action/dcc_candidate_std_uniform': jnp.mean(
            jnp.std(dcc_uniform_scores, axis=0)),
        'action/dcc_action_grad_norm': jnp.mean(
            jnp.linalg.norm(action_gradient, axis=-1)),
        'action/dcc_progress_spearman': dcc_progress_spearman,
    }

    if has_q:
      def q_score(actions):
        q_values = q_network.apply(q_params, obs, actions)
        return jnp.min(q_values, axis=-1)

      q_replay = q_score(replay_action)
      q_policy_scores = jax.vmap(q_score)(policy_actions)
      q_uniform_scores = jax.vmap(q_score)(uniform_actions)
      q_progress_spearman = _spearman(
          q_replay[None, :], progress[None, :])[0]
      per_state_rank_agreement = _spearman(
          jnp.swapaxes(dcc_policy_scores, 0, 1),
          jnp.swapaxes(q_policy_scores, 0, 1))
      q_twins = q_network.apply(q_params, obs, replay_action)
      metrics.update({
          'action/q_candidate_std_policy': jnp.mean(
              jnp.std(q_policy_scores, axis=0)),
          'action/q_candidate_std_uniform': jnp.mean(
              jnp.std(q_uniform_scores, axis=0)),
          'action/q_progress_spearman': q_progress_spearman,
          'action/dcc_q_candidate_spearman': jnp.mean(
              per_state_rank_agreement),
          'q/twin_disagreement_periodic': jnp.mean(
              jnp.abs(q_twins[:, 0] - q_twins[:, 1])),
      })

    del key_grad
    return metrics

  return jax.jit(diagnostic)
