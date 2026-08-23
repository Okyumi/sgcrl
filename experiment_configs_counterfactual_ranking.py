#!/usr/bin/env python3
"""One-million-step task-goal counterfactual-ranking falsification cells."""
import argparse
import shlex
import sys


TASKS = (
    ('sawyer_handle_press_side', 'task5'),
    ('sawyer_window_close', 'task8'),
)
SEEDS = (5, 6)


def build_configs():
  configs = []
  for task, label in TASKS:
    for seed in SEEDS:
      configs.append({
          'actor_mode': 'reset',
          'critic_mode': 'advantage_decomposed',
          'seed': seed,
          'single_task': task,
          'steps_per_task': 1_000_000,
          'base_steps': 1_000_000,
          'eval_every': 50_000,
          'dyn_aux_weight': 1.0,
          'combine_mode': 'add',
          'goal_encoder_mode': 'shared',
          'use_task_id': False,
          'actor_auto_reset': False,
          'in_trajectory_negative_repeats': 12,
          'interaction_weighted_relabeling': False,
          'action_effect_enabled': True,
          'action_effect_loss_weight': 1.0,
          'action_effect_temperature': 1.0,
          'action_effect_actor_weight': 1.0,
          'action_effect_hidden_dim': 256,
          'action_effect_actor_mode': 'effect_only',
          'action_effect_target_mode': 'counterfactual_rank',
          'success_bc_weight': 0.0,
          'counterfactual_rank_interval_steps': 50_000,
          'counterfactual_rank_num_anchors': 4,
          'counterfactual_rank_candidates_per_family': 4,
          'counterfactual_rank_rollout_horizon': 100,
          'counterfactual_rank_action_repeat': 5,
          'counterfactual_rank_local_noise_std': 0.10,
          'counterfactual_rank_anchor_mode': 'scripted_contact',
          'counterfactual_rank_anchor_search_steps': 150,
          'counterfactual_rank_interaction_threshold': 0.09,
          'counterfactual_rank_contact_gain': 5.0,
          'counterfactual_rank_success_threshold': 0.05,
          'counterfactual_rank_success_bonus': 1.0,
          'counterfactual_rank_min_outcome_gap': 0.002,
          'counterfactual_rank_buffer_capacity': 128,
          'counterfactual_rank_batch_anchors': 16,
          'counterfactual_rank_updates_per_event': 25,
          'counterfactual_rank_pairwise_temperature': 1.0,
          'counterfactual_rank_l2_weight': 1e-4,
          'shortcut_diagnostic_interval': 50,
          'action_landscape_diagnostic_interval_steps': 250_000,
          'action_landscape_num_anchors': 2,
          'action_landscape_candidates_per_family': 4,
          'action_landscape_rollout_horizon': 100,
          'action_landscape_anchor_prefix_steps': 20,
          'action_landscape_local_noise_std': 0.10,
          'action_landscape_interaction_aware_anchor': True,
          'action_landscape_anchor_search_steps': 200,
          'action_landscape_interaction_threshold': 0.09,
          'log_probe_data': False,
          'post_task_eval_scope': 'current',
          'wandb_group': f'CFRDCC-taskgoal-H100-chunk5-{label}',
      })
  return configs


def _emit(config):
  for key, value in config.items():
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
      print(index, config['wandb_group'], config['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    print(f'ERROR: setting {args.setting} out of range', file=sys.stderr)
    raise SystemExit(1)
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
