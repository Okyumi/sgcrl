#!/usr/bin/env python3
"""8M from-scratch Tasks 5/8: whole-episode Success-BC + local-σ_s gate.

Same algorithm as PAPER-DCC-EPISODE-LOCALGATE-4758 cells 2–3, but the
full 8M budget used by paper DCC+BC. Separate log/ckpt dirs so the
finished 1M task_0.pkl is not auto-resumed.

Cells:
  0  Task 5 handle_press_side, 8M from scratch
  1  Task 8 window_close, 8M from scratch
"""
from __future__ import annotations

import argparse
import shlex


SEED = 6
WANDB_GROUP = 'PAPER-DCC-EPISODE-LOCALGATE-58-8M'


def _paper(**extra) -> dict:
  return {
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
      'seed': SEED,
      'steps_per_task': 8_000_000,
      'base_steps': 8_000_000,
      'k_max': 5,
      'eval_every': 100_000,
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
      'success_bc_label_mode': 'episode_sparse_reward',
      'success_bc_critic_gate': True,
      'success_bc_critic_gate_mode': 'local',
      'success_bc_critic_state_noise': 0.1,
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
      'mid_task_checkpoint_every': 100_000,
      'use_action_entropy': True,
      'wandb_group': WANDB_GROUP,
      'num_tasks': 1,
      'adapt_heads_only': False,
      **extra,
  }


def build_configs():
  return [
      _paper(
          variant='handle_episode_local_gate_8m',
          single_task='sawyer_handle_press_side',
      ),
      _paper(
          variant='window_episode_local_gate_8m',
          single_task='sawyer_window_close',
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
      print(i, c['variant'], c['single_task'],
            f"labels={c['success_bc_label_mode']}",
            f"gate={c['success_bc_critic_gate_mode']}",
            f"steps={c['steps_per_task']}")
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
