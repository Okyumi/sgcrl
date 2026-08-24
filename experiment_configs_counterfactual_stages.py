#!/usr/bin/env python3
"""Staged falsification configs for phase-gated counterfactual chunk control."""
from __future__ import annotations

import argparse
import os
import shlex


TASKS = (
    ('sawyer_handle_press_side', 'task5', 0.02),
    ('sawyer_window_close', 'task8', 0.05),
)
SEEDS = (5, 6)
STAGES = ('A', 'B', 'C', 'D')


def _base(task, label, success_threshold, seed, steps, eval_every):
  return {
      'actor_mode': 'reset',
      'critic_mode': 'advantage_decomposed',
      'seed': seed,
      'single_task': task,
      'steps_per_task': steps,
      'base_steps': steps,
      'eval_every': eval_every,
      'dyn_aux_weight': 1.0,
      'combine_mode': 'add',
      'goal_encoder_mode': 'shared',
      'use_task_id': False,
      'actor_auto_reset': False,
      'in_trajectory_negative_repeats': 12,
      'interaction_weighted_relabeling': False,
      'action_effect_enabled': True,
      'action_effect_loss_weight': 1.0,
      'action_effect_actor_weight': 1.0,
      'action_effect_hidden_dim': 256,
      'action_effect_actor_mode': 'combined',
      'action_effect_target_mode': 'counterfactual_rank',
      'counterfactual_rank_actor_enabled': False,
      'counterfactual_rank_num_anchors': 4,
      'counterfactual_rank_candidates_per_family': 4,
      'counterfactual_rank_rollout_horizon': 100,
      'counterfactual_rank_action_repeat': 5,
      'counterfactual_rank_local_noise_std': 0.10,
      'counterfactual_rank_anchor_mode': 'scripted_contact',
      'counterfactual_rank_anchor_search_steps': 150,
      'counterfactual_rank_interaction_threshold': 0.09,
      'counterfactual_rank_contact_gain': 5.0,
      # The raw Sawyer wrappers emit 0/1. The learner's HER -1/0 shaping is a
      # separate internal signal and must not define simulator success.
      'counterfactual_rank_success_threshold': success_threshold,
      'counterfactual_rank_success_mode': 'positive_reward',
      'counterfactual_rank_success_bonus': 1.0,
      'counterfactual_rank_min_outcome_gap': 0.002,
      'counterfactual_rank_buffer_capacity': 128,
      'counterfactual_rank_batch_anchors': 16,
      'counterfactual_rank_updates_per_event': 25,
      'counterfactual_rank_pairwise_temperature': 1.0,
      'counterfactual_rank_l2_weight': 1e-4,
      'counterfactual_oracle_num_anchors': 4,
      'phase_gated_control': False,
      'phase_gate_reach_mode': 'policy',
      'phase_gate_interaction_threshold': 0.09,
      'phase_gate_chunk_length': 5,
      'phase_gate_num_candidates': 16,
      'phase_gate_local_noise_std': 0.10,
      'phase_gate_contact_gain': 5.0,
      'shortcut_diagnostic_interval': 50,
      'action_landscape_num_anchors': 4,
      'action_landscape_candidates_per_family': 4,
      'action_landscape_rollout_horizon': 100,
      'action_landscape_anchor_prefix_steps': 20,
      'action_landscape_local_noise_std': 0.10,
      'action_landscape_interaction_aware_anchor': True,
      'action_landscape_anchor_search_steps': 200,
      'action_landscape_interaction_threshold': 0.09,
      'action_landscape_action_repeat': 5,
      'action_landscape_use_best_progress': True,
      'action_landscape_success_threshold': success_threshold,
      'action_landscape_success_mode': 'positive_reward',
      'post_task_eval_scope': 'current',
      '_label': label,
  }


def build_configs(stage=None, reach_mode=None):
  stage = (stage or os.environ.get('COUNTERFACTUAL_STAGE', 'A')).upper()
  if stage not in STAGES:
    raise ValueError(f'COUNTERFACTUAL_STAGE must be one of {STAGES}')
  reach_mode = reach_mode or os.environ.get(
      'COUNTERFACTUAL_REACH_MODE', 'scripted_contact')
  if reach_mode not in ('policy', 'scripted_contact'):
    raise ValueError('COUNTERFACTUAL_REACH_MODE must be policy or scripted_contact')

  configs = []
  for task, label, success_threshold in TASKS:
    for seed in SEEDS:
      if stage == 'A':
        # Stage A2 only revalidates instrumentation after the reward fix. Two
        # post-prefill diagnostic events are sufficient; this is not a
        # performance comparison.
        config = _base(task, label, success_threshold, seed, 30_000, 10_000)
        config.update({
            'counterfactual_rank_interval_steps': 10_000,
            'counterfactual_rank_validation_anchors': 4,
            'counterfactual_oracle_interval_steps': 0,
            'action_landscape_diagnostic_interval_steps': 10_000,
            'wandb_group': f'CFR-STAGE-A2-positive-reward-{label}',
        })
      elif stage == 'B':
        config = _base(
            task, label, success_threshold, seed, 100_000, 25_000)
        config.update({
            'critic_mode': 'decomposed',
            'action_effect_enabled': False,
            'action_effect_actor_mode': 'combined',
            'action_effect_target_mode': 'psi_one_step',
            'counterfactual_rank_interval_steps': 0,
            'counterfactual_rank_validation_anchors': 0,
            'counterfactual_oracle_interval_steps': 25_000,
            'counterfactual_oracle_num_anchors': 8,
            'action_landscape_diagnostic_interval_steps': 0,
            'wandb_group': f'CFR-STAGE-B-oracle-decomposition-{label}',
        })
      elif stage == 'C':
        config = _base(
            task, label, success_threshold, seed, 250_000, 25_000)
        config.update({
            'counterfactual_rank_interval_steps': 25_000,
            'counterfactual_rank_validation_anchors': 8,
            'counterfactual_oracle_interval_steps': 50_000,
            'action_landscape_diagnostic_interval_steps': 50_000,
            'wandb_group': f'CFR-STAGE-C-heldout-chunk-ranker-{label}',
        })
      else:
        config = _base(
            task, label, success_threshold, seed, 1_000_000, 50_000)
        config.update({
            'counterfactual_rank_interval_steps': 50_000,
            'counterfactual_rank_validation_anchors': 4,
            'counterfactual_oracle_interval_steps': 250_000,
            'phase_gated_control': True,
            'phase_gate_reach_mode': reach_mode,
            'action_landscape_diagnostic_interval_steps': 250_000,
            'wandb_group': (
                f'PGC-DCC-{reach_mode.replace("_", "-")}-{label}'),
        })
      config['counterfactual_stage'] = stage
      config.pop('_label')
      configs.append(config)
  return configs


def _emit(config):
  for key, value in config.items():
    # Bookkeeping field is logged through the W&B group, not an absl flag.
    if key == 'counterfactual_stage':
      continue
    if isinstance(value, bool):
      value = 'true' if value else 'false'
    elif isinstance(value, str):
      value = shlex.quote(value)
    print(f'{key.upper()}={value}')


def main():
  parser = argparse.ArgumentParser()
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--setting', type=int)
  group.add_argument('--total', action='store_true')
  group.add_argument('--list', action='store_true')
  args = parser.parse_args()
  configs = build_configs()
  if args.total:
    print(len(configs))
    return
  if args.list:
    for index, config in enumerate(configs):
      print(index, config['counterfactual_stage'], config['wandb_group'],
            config['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
