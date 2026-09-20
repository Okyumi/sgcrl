#!/usr/bin/env python3
"""Find one Success-BC label that helps Tasks 5/8 without hurting 4/7.

Episode-wide labels retain 5/8 but stall 4/7. This grid keeps λ=0.1 and
only changes which sparse-0/1 transitions enter D_succ:

  0  Task 4 stick_pull, terminal_episode, resume paper BC task_3, 1M
  1  Task 7 shelf_place, terminal_episode, resume paper BC task_6, 1M
  2  Task 5 handle_press_side, current_sparse_reward, 1M from scratch
  3  Task 8 window_close, current_sparse_reward, 1M from scratch

Tasks 4/7 use the paper stack (native_info, dyn=0). 1M is enough to see
the known episode-wide collapse window. Tasks 5/8 use the diagnosis stack
that already showed terminal-episode Success-BC retains.

W&B: PAPER-DCC-UNIVERSAL-SUCCESS-BC
"""
from __future__ import annotations

import argparse
import shlex


SEED = 6
STEPS_PER_TASK = 1_000_000
WANDB_GROUP = 'PAPER-DCC-UNIVERSAL-SUCCESS-BC'
PAPER_BC_DIR = (
    '/scratch/yd2247/sgcrl/logs/paper_dcc_success_bc_jubail_checkpoints/10seed'
    '/actor_reset_critic_decomposed_tid_False_heads_True'
    '_success_native_info_dyn0.000_pt256x4'
    '_bridge_d2186abe4046_ret_847a4481eb68'
)

VARIANTS = (
    {
        'variant': 'task4_stick_terminal',
        'kind': 'resume_paper',
        'start_task': 4,
        'num_tasks': 5,
        'resume_checkpoint_file': f'{PAPER_BC_DIR}/seed_{SEED}/task_3.pkl',
        'success_bc_label_mode': 'terminal_episode',
    },
    {
        'variant': 'task7_shelf_terminal',
        'kind': 'resume_paper',
        'start_task': 7,
        'num_tasks': 8,
        'resume_checkpoint_file': f'{PAPER_BC_DIR}/seed_{SEED}/task_6.pkl',
        'success_bc_label_mode': 'terminal_episode',
    },
    {
        'variant': 'handle_current_sparse',
        'kind': 'single_task',
        'single_task': 'sawyer_handle_press_side',
        'success_bc_label_mode': 'current_sparse_reward',
    },
    {
        'variant': 'window_current_sparse',
        'kind': 'single_task',
        'single_task': 'sawyer_window_close',
        'success_bc_label_mode': 'current_sparse_reward',
    },
)


def _paper_resume(start_task: int, num_tasks: int, resume_file: str,
                  label_mode: str) -> dict:
  return {
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
      'seed': SEED,
      'start_task': start_task,
      'num_tasks': num_tasks,
      'steps_per_task': STEPS_PER_TASK,
      'base_steps': STEPS_PER_TASK,
      'k_max': 5,
      'eval_every': 50_000,
      'eval_episodes': 10,
      'eval_record_video': False,
      'network_width': 1024,
      'critic_depth': 4,
      'actor_depth': 4,
      'use_residual': True,
      'sawyer_success_mode': 'native_info',
      'goal_conditioning_mode': 'full_state',
      'use_task_id': False,
      'adapt_heads_only': True,
      'encoder_from_base': False,
      'actor_auto_reset': False,
      'dyn_aux_weight': 0.0,
      'dyn_aux_after_task0': -1.0,
      'phi_task_width': 256,
      'phi_task_depth': 4,
      'combine_mode': 'add',
      'goal_encoder_mode': 'shared',
      'in_trajectory_negative_repeats': 1,
      'action_effect_enabled': False,
      'success_bc_weight': 0.1,
      'success_bc_label_mode': label_mode,
      'success_buffer_capacity': 4096,
      'success_bc_batch_size': 64,
      'actor_goal_mode': 'her',
      'actor_success_score_weight': 0.0,
      'truncate_on_success': False,
      'counterfactual_rank_interval_steps': 0,
      'counterfactual_oracle_interval_steps': 0,
      'action_landscape_diagnostic_interval_steps': 0,
      'shortcut_diagnostic_interval': 0,
      'log_rl_metrics': True,
      'rl_metrics_occasional_multiplier': 2,
      'log_pool_cosine': False,
      'log_mixture_norm': False,
      'log_probe_data': False,
      'profile_runtime': True,
      'num_actors': 2,
      'intra_eval_previous': False,
      'post_task_eval_scope': 'current',
      'her_phase_log_enabled': False,
      'success_trace_log_enabled': False,
      'mid_task_checkpoint_every': 50_000,
      'use_action_entropy': True,
      'resume_checkpoint_file': resume_file,
      'wandb_group': WANDB_GROUP,
  }


def _diagnosis_single(task: str, label_mode: str) -> dict:
  return {
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
      'seed': SEED,
      'single_task': task,
      'num_tasks': 1,
      'steps_per_task': STEPS_PER_TASK,
      'base_steps': STEPS_PER_TASK,
      'network_width': 1024,
      'critic_depth': 4,
      'actor_depth': 4,
      'dyn_aux_weight': 1.0,
      'phi_task_width': 256,
      'phi_task_depth': 4,
      'in_trajectory_negative_repeats': 1,
      'eval_every': 50_000,
      'eval_episodes': 10,
      'eval_record_video': False,
      'sawyer_success_mode': 'corrected',
      'goal_conditioning_mode': 'full_state',
      'use_task_id': False,
      'adapt_heads_only': False,
      'encoder_from_base': False,
      'actor_auto_reset': False,
      'counterfactual_rank_interval_steps': 0,
      'counterfactual_oracle_interval_steps': 0,
      'action_landscape_diagnostic_interval_steps': 0,
      'shortcut_diagnostic_interval': 0,
      'log_rl_metrics': True,
      'log_pool_cosine': True,
      'profile_runtime': True,
      'intra_eval_previous': False,
      'post_task_eval_scope': 'current',
      'interaction_weighted_relabeling': False,
      'action_effect_enabled': False,
      'success_bc_weight': 0.1,
      'success_bc_label_mode': label_mode,
      'success_buffer_capacity': 4096,
      'success_bc_batch_size': 64,
      'truncate_on_success': False,
      'actor_goal_mode': 'her',
      'actor_success_score_weight': 0.0,
      'combine_mode': 'add',
      'goal_encoder_mode': 'shared',
      'critic_phase_probe_enabled': True,
      'critic_phase_probe_episodes': 10,
      'her_phase_log_enabled': True,
      'success_trace_log_enabled': True,
      'mid_task_checkpoint_every': 50_000,
      'her_future_sampling_mode': 'discounted',
      'her_future_discount': -1.0,
      'freeze_critic_after_success_rate': -1.0,
      'use_action_entropy': True,
      'wandb_group': WANDB_GROUP,
  }


def build_configs():
  configs = []
  for variant in VARIANTS:
    if variant['kind'] == 'resume_paper':
      config = _paper_resume(
          variant['start_task'], variant['num_tasks'],
          variant['resume_checkpoint_file'],
          variant['success_bc_label_mode'])
    else:
      config = _diagnosis_single(
          variant['single_task'], variant['success_bc_label_mode'])
    config['variant'] = variant['variant']
    configs.append(config)
  return configs


_EXTRA = ('variant',)


def _emit(config):
  for key, value in config.items():
    if key in _EXTRA:
      continue
    if isinstance(value, bool):
      value = 'true' if value else 'false'
    elif isinstance(value, str):
      value = shlex.quote(value)
    print(f'{key.upper()}={value}')
  print(f"VARIANT={shlex.quote(config['variant'])}")


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
    for i, c in enumerate(configs):
      task = c.get('single_task', f"start={c.get('start_task')}")
      print(i, c['variant'], task, f"labels={c['success_bc_label_mode']}")
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
