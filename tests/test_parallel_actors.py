#!/usr/bin/env python3
"""CPU actor pool: one learner-equivalent get() per completed episode."""
from __future__ import annotations

import ast
import time
from pathlib import Path
import sys
import threading

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.parallel_actors import ActorEpisodePool


class _FakeLoop:
  def __init__(self, delay, steps=7):
    self.delay = delay
    self.steps = steps
    self.calls = 0
    self.lock = threading.Lock()

  def run_episode(self):
    time.sleep(self.delay)
    with self.lock:
      self.calls += 1
      n = self.calls
    return {'episode_length': self.steps, 'n': n}


def test_pool_returns_one_episode_per_get():
  loops = [_FakeLoop(0.02), _FakeLoop(0.02)]
  pool = ActorEpisodePool(loops)
  try:
    lengths = []
    for _ in range(6):
      actor_id, result, duration = pool.get()
      assert actor_id in (0, 1)
      assert result['episode_length'] == 7
      assert duration >= 0.0
      lengths.append(result['episode_length'])
    assert lengths == [7] * 6
  finally:
    pool.stop()


def test_two_actors_overlap_collection_with_learner():
  loops = [_FakeLoop(0.05), _FakeLoop(0.05)]
  started = time.perf_counter()
  pool = ActorEpisodePool(loops)
  try:
    for _ in range(4):
      pool.get()
      time.sleep(0.05)  # fake learner.step
  finally:
    pool.stop()
  elapsed = time.perf_counter() - started
  sequential = 4 * (0.05 + 0.05)
  assert elapsed < sequential * 0.85, elapsed


def test_stop_joins_workers():
  pool = ActorEpisodePool([_FakeLoop(0.01)])
  pool.get()
  pool.stop()
  assert all(not thread.is_alive() for thread in pool._threads)


def test_runner_keeps_one_learner_step_per_episode():
  source = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  tree = ast.parse(source)
  found_pool = False
  found_learner = False
  for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
      if node.func.id == 'ActorEpisodePool':
        found_pool = True
  assert found_pool
  assert 'learner.step()' in source
  assert 'one learner.step per' in source
  assert '_make_actor_observers()' in source
  # Sequential path is preserved for the default flag.
  assert 'num_actors = max(1, int(FLAGS.num_actors))' in source
  found_learner = source.count('learner.step()') >= 1
  assert found_learner


def test_torch_draft_forwards_num_actors():
  draft = (REPO_ROOT / 'DRAFT.sh').read_text(encoding='utf-8')
  assert 'NUM_ACTORS="${NUM_ACTORS:-1}"' in draft
  assert '--num_actors=$NUM_ACTORS' in draft


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'parallel actor tests passed ({len(tests)})')
