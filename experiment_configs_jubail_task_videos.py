#!/usr/bin/env python3
"""Jubail Task-5 / Task-6 / Task-8 rollout GIFs.

Cells:
  0  handle_nobc_gifs  — offline GIFs from seed-6 DCC w/o BC (job 17901349)
  1  push_gifs         — offline GIFs from seed-6 DCC push
  2  handle_bc_train   — train DCC + Success-BC 1M, save paced + first-success GIFs
  3  window_nobc_gifs  — offline GIFs from seed-6 DCC window (job 17954891)
  4  window_bc_train   — train DCC + Success-BC 1M on Task 8

W&B: TASK58-JUBAIL-TASK-VIDEOS
"""
from __future__ import annotations

import argparse
import shlex


SEED = 6
STEPS_PER_TASK = 1_000_000
WANDB_GROUP = 'TASK58-JUBAIL-TASK-VIDEOS'
SOURCE_CKPT_DIR = (
    '/scratch/yd2247/sgcrl/logs/jubail_task5_action_advice/checkpoints')

VARIANTS = (
    {
        'variant': 'handle_nobc_gifs',
        'job_mode': 'offline_gifs',
        'single_task': 'sawyer_handle_press_side',
        'success_bc_weight': 0.0,
        'success_bc_label_mode': 'raw_horizon',
        'eval_video_first_success': False,
        'label': 'dcc_nobc',
        'first_success_step': 150_000,
        'source_checkpoint_dir': SOURCE_CKPT_DIR,
    },
    {
        'variant': 'push_gifs',
        'job_mode': 'offline_gifs',
        'single_task': 'sawyer_push',
        'success_bc_weight': 0.0,
        'success_bc_label_mode': 'raw_horizon',
        'eval_video_first_success': False,
        'label': 'dcc',
        'first_success_step': 0,
        'source_checkpoint_dir': SOURCE_CKPT_DIR,
    },
    {
        'variant': 'handle_bc_train',
        'job_mode': 'train',
        'single_task': 'sawyer_handle_press_side',
        'success_bc_weight': 0.1,
        'success_bc_label_mode': 'terminal_episode',
        'eval_video_first_success': True,
        'label': 'dcc_bc',
        'first_success_step': 150_000,
        'source_checkpoint_dir': '',
    },
    {
        'variant': 'window_nobc_gifs',
        'job_mode': 'offline_gifs',
        'single_task': 'sawyer_window_close',
        'success_bc_weight': 0.0,
        'success_bc_label_mode': 'raw_horizon',
        'eval_video_first_success': False,
        'label': 'dcc_nobc',
        'first_success_step': 250_000,
        'source_checkpoint_dir': SOURCE_CKPT_DIR,
    },
    {
        'variant': 'window_bc_train',
        'job_mode': 'train',
        'single_task': 'sawyer_window_close',
        'success_bc_weight': 0.1,
        'success_bc_label_mode': 'terminal_episode',
        'eval_video_first_success': True,
        'label': 'dcc_bc',
        'first_success_step': 250_000,
        'source_checkpoint_dir': '',
    },
)


def _shared(seed: int, task: str, *, bc_weight: float, bc_mode: str,
            first_success: bool) -> dict:
  return {
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
      'seed': seed,
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
      'eval_record_video': True,
      'eval_video_every': 100_000,
      'eval_video_fps': 10,
      'eval_video_first_success': first_success,
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
      'success_bc_weight': bc_weight,
      'success_bc_label_mode': bc_mode,
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
    config = _shared(
        SEED, variant['single_task'],
        bc_weight=float(variant['success_bc_weight']),
        bc_mode=str(variant['success_bc_label_mode']),
        first_success=bool(variant['eval_video_first_success']))
    config['variant'] = variant['variant']
    config['job_mode'] = variant['job_mode']
    config['label'] = variant['label']
    config['first_success_step'] = int(variant['first_success_step'])
    config['source_checkpoint_dir'] = variant['source_checkpoint_dir']
    configs.append(config)
  return configs


_EXTRA = (
    'variant', 'job_mode', 'label', 'first_success_step',
    'source_checkpoint_dir',
)


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
  print(f"JOB_MODE={shlex.quote(config['job_mode'])}")
  print(f"LABEL={shlex.quote(config['label'])}")
  print(f"FIRST_SUCCESS_STEP={int(config['first_success_step'])}")
  print(f"SOURCE_CHECKPOINT_DIR={shlex.quote(config['source_checkpoint_dir'])}")


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
      print(i, c['variant'], c['job_mode'], c['single_task'],
            f"bc={c['success_bc_weight']}", c['label'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
