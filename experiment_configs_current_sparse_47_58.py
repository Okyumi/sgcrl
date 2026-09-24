#!/usr/bin/env python3
"""Per-step Success-BC on Tasks 4/7 and 5/8.

Same λ=0.1 and the same 4096 ring. Only transitions whose own sparse
bit is 1 enter D_succ (not the whole episode). Paper stack everywhere
(native_info, dyn=0, 1024×4).

Cells:
  0  Task 4 stick_pull, 8M from paper BC task_3, seed 6
  1  Task 7 shelf_place, 8M from paper BC task_6, seed 6
  2  Task 5 handle_press_side, 1M from scratch, seed 6
  3  Task 8 window_close, 1M from scratch, seed 6
"""
from __future__ import annotations

import argparse
import shlex


SEED = 6
WANDB_GROUP = 'PAPER-DCC-CURRENT-SPARSE-4758'
PAPER_BC_DIR = (
    '/scratch/yd2247/sgcrl/logs/paper_dcc_success_bc_jubail_checkpoints/10seed'
    '/actor_reset_critic_decomposed_tid_False_heads_True'
    '_success_native_info_dyn0.000_pt256x4'
    '_bridge_d2186abe4046_ret_847a4481eb68'
)


def _paper(steps: int, eval_every: int, **extra) -> dict:
  return {
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
      'seed': SEED,
      'steps_per_task': steps,
      'base_steps': steps,
      'k_max': 5,
      'eval_every': eval_every,
      'eval_episodes': 10,
      'eval_record_video': False,
      'network_width': 1024,
      'critic_depth': 4,
      'actor_depth': 4,
      'use_residual': True,
      'sawyer_success_mode': 'native_info',
      'goal_conditioning_mode': 'full_state',
      'use_task_id': False,
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
      'success_bc_label_mode': 'current_sparse_reward',
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
      'mid_task_checkpoint_every': eval_every,
      'use_action_entropy': True,
      'wandb_group': WANDB_GROUP,
      **extra,
  }


def build_configs():
  return [
      _paper(
          8_000_000, 100_000,
          variant='task4_stick_current_sparse',
          start_task=4,
          num_tasks=5,
          adapt_heads_only=True,
          resume_checkpoint_file=f'{PAPER_BC_DIR}/seed_{SEED}/task_3.pkl',
      ),
      _paper(
          8_000_000, 100_000,
          variant='task7_shelf_current_sparse',
          start_task=7,
          num_tasks=8,
          adapt_heads_only=True,
          resume_checkpoint_file=f'{PAPER_BC_DIR}/seed_{SEED}/task_6.pkl',
      ),
      _paper(
          1_000_000, 50_000,
          variant='handle_current_sparse',
          single_task='sawyer_handle_press_side',
          num_tasks=1,
          adapt_heads_only=False,
      ),
      _paper(
          1_000_000, 50_000,
          variant='window_current_sparse',
          single_task='sawyer_window_close',
          num_tasks=1,
          adapt_heads_only=False,
      ),
  ]


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
      print(i, c['variant'], task,
            f"labels={c['success_bc_label_mode']}",
            f"steps={c['steps_per_task']}")
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
