"""Overlapped CPU actors for the sequential continual runner.

Each completed episode still triggers exactly one ``learner.step()`` on the
main thread, so ``num_sgd_steps_per_step`` per env-step (UTD) is unchanged.
Extra actors only hide environment latency behind that learner call, which is
the Launchpad ``local_mt`` collection idea without changing the loss.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, List, Tuple

EpisodeResult = Dict[str, Any]


class SilentLogger:
  """Drop-in logger for extra actors so only actor 0 writes W&B/CSV."""

  def write(self, data):
    return

  def close(self):
    return


class ActorEpisodePool:
  """Run ``EnvironmentLoop.run_episode`` on background threads."""

  def __init__(self, loops):
    loops = list(loops)
    if not loops:
      raise ValueError('ActorEpisodePool requires at least one loop.')
    self._loops = loops
    self._queue: queue.Queue = queue.Queue(maxsize=len(loops))
    self._stop = threading.Event()
    self._threads: List[threading.Thread] = []
    for actor_id, loop in enumerate(loops):
      thread = threading.Thread(
          target=self._worker,
          args=(actor_id, loop),
          name=f'actor-{actor_id}',
          daemon=True)
      self._threads.append(thread)
      thread.start()

  def _worker(self, actor_id, loop):
    while not self._stop.is_set():
      started = time.perf_counter()
      try:
        result = loop.run_episode()
      except Exception as exc:  # pylint: disable=broad-except
        if self._stop.is_set():
          return
        self._queue.put((actor_id, exc, 0.0))
        return
      duration = time.perf_counter() - started
      while not self._stop.is_set():
        try:
          self._queue.put((actor_id, result, duration), timeout=0.5)
          break
        except queue.Full:
          continue

  def get(self) -> Tuple[int, EpisodeResult, float]:
    """Block until the next completed episode.

    Raises the worker exception if a collector crashed.
    """
    while True:
      try:
        actor_id, payload, duration = self._queue.get(timeout=0.5)
      except queue.Empty:
        if self._stop.is_set():
          raise RuntimeError(
              'ActorEpisodePool stopped before an episode arrived.')
        self._raise_if_workers_died()
        continue
      if isinstance(payload, Exception):
        raise RuntimeError(f'Actor {actor_id} failed') from payload
      return actor_id, payload, duration

  def _raise_if_workers_died(self):
    for thread in self._threads:
      if not thread.is_alive() and not self._stop.is_set():
        raise RuntimeError(f'Actor thread {thread.name} exited early.')

  def stop(self):
    self._stop.set()
    while True:
      try:
        self._queue.get_nowait()
      except queue.Empty:
        break
    for thread in self._threads:
      thread.join(timeout=60.0)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    self.stop()
    return False
