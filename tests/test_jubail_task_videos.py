#!/usr/bin/env python3
"""Tests for Task-5/6 GIF recording helpers and Jubail video configs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_jubail_task_videos as cfg
from contrastive import eval_video


def _load_recorder():
  path = REPO_ROOT / 'scripts' / 'record_checkpoint_rollout_gifs.py'
  spec = importlib.util.spec_from_file_location(
      'record_checkpoint_rollout_gifs', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_video_configs():
  configs = cfg.build_configs()
  assert len(configs) == 5
  names = [c['variant'] for c in configs]
  assert names == [
      'handle_nobc_gifs', 'push_gifs', 'handle_bc_train',
      'window_nobc_gifs', 'window_bc_train']
  assert [c['job_mode'] for c in configs] == [
      'offline_gifs', 'offline_gifs', 'train',
      'offline_gifs', 'train']
  assert configs[0]['single_task'] == 'sawyer_handle_press_side'
  assert configs[1]['single_task'] == 'sawyer_push'
  assert configs[3]['single_task'] == 'sawyer_window_close'
  assert configs[4]['single_task'] == 'sawyer_window_close'
  assert configs[0]['success_bc_weight'] == 0.0
  assert configs[2]['success_bc_weight'] == 0.1
  assert configs[4]['success_bc_weight'] == 0.1
  assert configs[4]['success_bc_label_mode'] == 'terminal_episode'
  assert configs[2]['eval_video_first_success'] is True
  assert configs[4]['eval_video_first_success'] is True
  assert configs[2]['eval_record_video'] is True
  assert configs[2]['eval_video_every'] == 100_000
  assert configs[0]['first_success_step'] == 150_000
  assert configs[1]['first_success_step'] == 0
  assert configs[3]['first_success_step'] == 250_000
  assert configs[0]['adapt_heads_only'] is False
  assert configs[2]['critic_mode'] == 'decomposed'
  assert configs[4]['critic_mode'] == 'decomposed'
  assert all(c['seed'] == 6 for c in configs)
  assert all(c['network_width'] == 1024 for c in configs)


def test_even_targets_are_ten_paced_steps():
  recorder = _load_recorder()
  tokens = recorder.even_target_tokens(10)
  assert tokens == [
      '100000', '200000', '300000', '400000', '500000',
      '600000', '700000', '800000', '900000', '1000000']
  ckpts = [(s, Path(f'task_0_step_{s}.pkl')) for s in (
      50100, 100200, 150300, 200400, 250500, 300600, 350700, 400800,
      450900, 501000, 551100, 601200, 651300, 701400, 751500, 801600,
      851700, 901800, 951900)]
  picked = recorder.pick_targets(ckpts, tokens)
  assert [s for s, _ in picked] == [
      100200, 200400, 300600, 400800, 501000,
      601200, 701400, 801600, 901800, 951900]


def test_list_step_ckpts_skips_inject_suffix(tmp_path: Path):
  recorder = _load_recorder()
  env = 'sawyer_handle_press_side'
  root = tmp_path / 'ckpts'
  good = root / (
      'actor_reset_critic_decomposed_tid_False_heads_False_success_'
      f'corrected_dyn1.000_pt256x4_env_{env}') / 'seed_6'
  inject = root / (
      'actor_reset_critic_decomposed_tid_False_heads_False_success_'
      f'corrected_dyn1.000_pt256x4_env_{env}_ret_798af817bfde') / 'seed_6'
  good.mkdir(parents=True)
  inject.mkdir(parents=True)
  (good / 'task_0_step_150300.pkl').write_bytes(b'x')
  (inject / 'task_0_step_150300.pkl').write_bytes(b'y')
  ckpts = recorder.list_step_ckpts(root, env, 6)
  assert len(ckpts) == 1
  assert ckpts[0][0] == 150300
  assert 'ret_' not in str(ckpts[0][1])


class _BareActor:
  def select_action(self, observation):
    del observation
    return np.zeros(4, dtype=np.float32)


class _FakeEnv:
  def __init__(self, rewards):
    self._rewards = list(rewards)
    self._index = 0

  def reset(self):
    self._index = 0
    return self._timestep(reward=0.0, last=False)

  def step(self, action):
    del action
    self._index += 1
    if self._index >= len(self._rewards):
      return self._timestep(reward=0.0, last=True)
    return self._timestep(
        reward=self._rewards[self._index - 1], last=False)

  def _timestep(self, reward, last):
    class _Ts:
      observation = np.zeros(8, dtype=np.float32)
    ts = _Ts()
    ts.reward = reward
    ts.last = lambda: last
    return ts


def test_record_without_observe_and_until_success():
  original = eval_video.render_rgb_array
  eval_video.render_rgb_array = lambda env: np.zeros((8, 8, 3), dtype=np.uint8)
  try:
    env = _FakeEnv(rewards=[0.0, 0.0, 0.0])
    frames, episode_return, success = eval_video.record_episode_frames(
        env, _BareActor())
    assert frames.shape[0] == 4
    assert success == 0.0
    env = _FakeEnv(rewards=[0.0, 1.0, 0.0])
    frames, episode_return, success, attempts = eval_video.record_until_success(
        env, _BareActor(), max_episodes=3)
    assert success == 1.0
    assert attempts == 1
    assert episode_return == 1.0
  finally:
    eval_video.render_rgb_array = original


def test_save_gif_downsamples(tmp_path: Path):
  frames = np.zeros((12, 48, 64, 3), dtype=np.uint8)
  frames[6:] = 255
  path = tmp_path / 'rollout.gif'
  written = eval_video.save_gif(
      frames, path, fps=8, max_side=32, max_frames=6)
  assert Path(written).is_file()
  small = eval_video.downsample_frames(frames, max_side=32, max_frames=6)
  assert small.shape[0] == 6
  assert max(small.shape[1], small.shape[2]) <= 32


def main():
  test_video_configs()
  test_even_targets_are_ten_paced_steps()
  test_record_without_observe_and_until_success()
  with tempfile.TemporaryDirectory() as tmp:
    test_list_step_ckpts_skips_inject_suffix(Path(tmp))
    test_save_gif_downsamples(Path(tmp))
  print('jubail task video tests passed')


if __name__ == '__main__':
  main()
