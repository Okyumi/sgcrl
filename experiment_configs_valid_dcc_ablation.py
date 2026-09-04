#!/usr/bin/env python3
"""Protocol-valid DCC/Advantage/SuccessBC experiments.

The continual cells deliberately include a corrected-wrapper plain-DCC
control: the old continual controls predate the Task-5/Task-8 wrapper repair.
"""
from __future__ import annotations

import argparse
import os
import shlex


SEEDS = (5, 6)
TASK58 = ('sawyer_handle_press_side', 'sawyer_window_close')
CONTINUAL_STEPS_PER_TASK = int(os.environ.get(
    'CONTINUAL_STEPS_PER_TASK', '4000000'))
if CONTINUAL_STEPS_PER_TASK not in (4_000_000, 8_000_000):
  raise ValueError('CONTINUAL_STEPS_PER_TASK must be 4000000 or 8000000.')
CONTINUAL_BUDGET_M = CONTINUAL_STEPS_PER_TASK // 1_000_000

PLAIN = {
    'variant': 'plain_dcc',
    'critic_mode': 'decomposed',
    'action_effect_enabled': False,
    'action_effect_target_mode': 'psi_one_step',
    'success_bc_weight': 0.0,
    'success_bc_label_mode': 'episode_sparse_reward',
}

ADVANTAGE_1STEP = {
    'variant': 'advantage_1step',
    'critic_mode': 'advantage_decomposed',
    'action_effect_enabled': True,
    'action_effect_target_mode': 'psi_one_step',
    'success_bc_weight': 0.0,
    'success_bc_label_mode': 'episode_sparse_reward',
}

TERMINAL_SUCCESS_BC = {
    'variant': 'terminal_success_bc_l0p1',
    'critic_mode': 'decomposed',
    'action_effect_enabled': False,
    'action_effect_target_mode': 'psi_one_step',
    'success_bc_weight': 0.1,
    'success_bc_label_mode': 'episode_sparse_reward',
}


def _common():
  return {
      'actor_mode': 'reset',
      'network_width': 1024,
      'critic_depth': 4,
      'actor_depth': 4,
      'dyn_aux_weight': 1.0,
      'phi_task_width': 256,
      'phi_task_depth': 4,
      'combine_mode': 'add',
      'goal_encoder_mode': 'shared',
      'in_trajectory_negative_repeats': 1,
      'action_effect_loss_weight': 1.0,
      'action_effect_actor_weight': 1.0,
      'action_effect_actor_mode': 'combined',
      'success_buffer_capacity': 4096,
      'success_bc_batch_size': 64,
      'sawyer_success_mode': 'corrected',
      'goal_conditioning_mode': 'full_state',
      'use_task_id': False,
      'actor_auto_reset': False,
      'counterfactual_rank_interval_steps': 0,
      'counterfactual_oracle_interval_steps': 0,
      'action_landscape_diagnostic_interval_steps': 0,
      'shortcut_diagnostic_interval': 0,
      'log_rl_metrics': False,
      'log_pool_cosine': False,
      'log_mixture_norm': False,
      'log_probe_data': False,
      'profile_runtime': True,
      'intra_eval_previous': False,
      'post_task_eval_scope': 'current',
  }


def build_configs():
  configs = []

  # Eight matched single-task cells: method x task x seed. These establish
  # whether the 1M SuccessBC result persists through 4M steps.
  for variant, group in (
      (PLAIN, 'TASK58-CORRECTED-4M-PLAIN-DCC'),
      (TERMINAL_SUCCESS_BC,
       'TASK58-CORRECTED-4M-TERMINAL-SUCCESS-BC-L0.1')):
    for env_name in TASK58:
      for seed in SEEDS:
        configs.append({
            **_common(), **variant,
            'suite': 'task58_4m',
            'name': f"task58_4m_{variant['variant']}_{env_name}_s{seed}",
            'seed': seed,
            'single_task': env_name,
            'num_tasks': 1,
            'steps_per_task': 4_000_000,
            'base_steps': 4_000_000,
            'eval_every': 100_000,
            'eval_episodes': 10,
            'wandb_group': group,
        })

  # Six full continual cells: corrected plain control plus the two requested
  # modifications. The default 4M-per-task pilot minimizes turnaround while
  # retaining a matched plain control; set CONTINUAL_STEPS_PER_TASK=8000000
  # for the established full paper budget.
  for variant, group in (
      (PLAIN, f'CONTINUAL10-CORRECTED-{CONTINUAL_BUDGET_M}M-PLAIN-DCC'),
      (ADVANTAGE_1STEP,
       f'CONTINUAL10-CORRECTED-{CONTINUAL_BUDGET_M}M-ADVANTAGE-1STEP'),
      (TERMINAL_SUCCESS_BC,
       f'CONTINUAL10-CORRECTED-{CONTINUAL_BUDGET_M}M-TERMINAL-SUCCESS-BC-L0.1')):
    for seed in SEEDS:
      configs.append({
          **_common(), **variant,
          'suite': f'continual10_{CONTINUAL_BUDGET_M}m',
          'name': (f"continual10_{CONTINUAL_BUDGET_M}m_"
                   f"{variant['variant']}_s{seed}"),
          'seed': seed,
          'single_task': '',
          'num_tasks': 10,
          'steps_per_task': CONTINUAL_STEPS_PER_TASK,
          'base_steps': CONTINUAL_STEPS_PER_TASK,
          # Forty measurements per task, at one quarter of the old 50k
          # evaluation overhead.
          'eval_every': 200_000,
          'eval_episodes': 10,
          'wandb_group': group,
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
      print(index, config['suite'], config['variant'],
            config['single_task'] or 'tasks0-9', config['seed'],
            config['wandb_group'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
