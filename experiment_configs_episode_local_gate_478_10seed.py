#!/usr/bin/env python3
"""10-seed gated whole-episode Success-BC on Tasks 4, 7, and 8.

Local-σ_s critic gate, episode_sparse_reward, λ=0.1. Resume paper
Success-BC predecessors (seeds 5–14). Paper stack, 8M, eval every 100k.

Cells:
  0–9   stick_pull from task_3, seeds 5–14
  10–19 shelf_place from task_6, seeds 5–14
  20–29 window_close from task_7, seeds 5–14
"""
from __future__ import annotations

import argparse
import shlex


SEEDS = tuple(range(5, 15))
STEPS_PER_TASK = 8_000_000
WANDB_GROUP = 'PAPER-DCC-EPISODE-LOCALGATE-478-10SEED'
PAPER_BC_DIR = (
    '/scratch/yd2247/sgcrl/logs/paper_dcc_success_bc_jubail_checkpoints/10seed'
    '/actor_reset_critic_decomposed_tid_False_heads_True'
    '_success_native_info_dyn0.000_pt256x4'
    '_bridge_d2186abe4046_ret_847a4481eb68'
)

VARIANTS = (
    {
        'variant': 'task4_stick_episode_local_gate',
        'start_task': 4,
        'num_tasks': 5,
        'resume_task': 3,
    },
    {
        'variant': 'task7_shelf_episode_local_gate',
        'start_task': 7,
        'num_tasks': 8,
        'resume_task': 6,
    },
    {
        'variant': 'task8_window_episode_local_gate',
        'start_task': 8,
        'num_tasks': 9,
        'resume_task': 7,
    },
)


def _resume_file(seed: int, resume_task: int) -> str:
  return f'{PAPER_BC_DIR}/seed_{seed}/task_{resume_task}.pkl'


def _shared(seed: int, start_task: int, num_tasks: int,
            resume_file: str) -> dict:
  return {
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
      'seed': seed,
      'start_task': start_task,
      'num_tasks': num_tasks,
      'steps_per_task': STEPS_PER_TASK,
      'base_steps': STEPS_PER_TASK,
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
      'resume_checkpoint_file': resume_file,
      'wandb_group': WANDB_GROUP,
  }


def build_configs():
  configs = []
  for variant in VARIANTS:
    for seed in SEEDS:
      config = _shared(
          seed, variant['start_task'], variant['num_tasks'],
          _resume_file(seed, variant['resume_task']))
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
      print(i, c['variant'], f"seed={c['seed']}",
            f"start={c['start_task']}", f"resume={c['resume_checkpoint_file']}")
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
