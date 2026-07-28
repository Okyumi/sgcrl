"""Cross-task checkpoint I/O for the continual SAC baseline.

Mirrors ``run_continual_contrastive.py``'s ``_ckpt_path`` / ``save_ckpt`` /
``load_ckpt`` trio, with two SAC-specific differences:

* The config key carries a ``_rew_{steppen,sparse01}`` suffix.  The critic's TD
  target is trained against one specific reward shape, so a checkpoint trained
  under the other shape must never be loadable — the path key enforces that
  structurally rather than by a runtime check.
* The CRL decomposed-critic ``_dyn*_pt*`` suffix has no SAC analogue and is
  therefore absent.

Checkpoints are plain pickles of numpy pytrees, same as the CRL driver, so the
existing analysis scripts under ``results/scripts/`` can read them unchanged.

``jax.tree_util.tree_map`` is used instead of the ``jax.tree_map`` alias found
elsewhere in the repo so this module also imports on newer JAX releases (the
alias was removed in JAX 0.4.25); the two are identical under the pinned
``jax==0.3.13``.
"""
import os
import pickle
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


def reward_tag(step_penalty_reward: bool) -> str:
  """Path fragment identifying the reward shape."""
  return 'steppen' if step_penalty_reward else 'sparse01'


def config_key(
    critic_mode: str = 'persistent',
    use_task_id: bool = True,
    adapt_heads_only: bool = True,
    actor_mode: str = 'cka',
    step_penalty_reward: bool = True,
) -> str:
  """Directory name uniquely identifying an ablation cell."""
  return (f'actor_{actor_mode}_critic_{critic_mode}'
          f'_tid_{use_task_id}_heads_{adapt_heads_only}'
          f'_rew_{reward_tag(step_penalty_reward)}')


def ckpt_path(
    ckpt_dir: str,
    task_id: int,
    seed: int,
    critic_mode: str = 'persistent',
    use_task_id: bool = True,
    adapt_heads_only: bool = True,
    actor_mode: str = 'cka',
    step_penalty_reward: bool = True,
) -> str:
  """Checkpoint path keyed by ablation-relevant config (``alg=sac_her`` implicit)."""
  return os.path.join(
      ckpt_dir,
      config_key(critic_mode, use_task_id, adapt_heads_only, actor_mode,
                 step_penalty_reward),
      f'seed_{seed}',
      f'task_{task_id}.pkl')


def save_ckpt(ckpt_dir: str, task_id: int, seed: int, data: Dict[str, Any],
              critic_mode: str = 'persistent', use_task_id: bool = True,
              adapt_heads_only: bool = True, actor_mode: str = 'cka',
              step_penalty_reward: bool = True) -> str:
  """Pickle ``data`` (JAX arrays converted to numpy) and return the path."""
  path = ckpt_path(ckpt_dir, task_id, seed, critic_mode, use_task_id,
                   adapt_heads_only, actor_mode, step_penalty_reward)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  data_np = jax.tree_util.tree_map(
      lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, data)
  with open(path, 'wb') as f:
    pickle.dump(data_np, f)
  print(f'  [ckpt] Saved -> {path}', flush=True)
  return path


def load_ckpt(ckpt_dir: str, task_id: int, seed: int,
              critic_mode: str = 'persistent', use_task_id: bool = True,
              adapt_heads_only: bool = True, actor_mode: str = 'cka',
              step_penalty_reward: bool = True) -> Dict[str, Any]:
  """Load a checkpoint, converting numpy arrays back to JAX arrays."""
  path = ckpt_path(ckpt_dir, task_id, seed, critic_mode, use_task_id,
                   adapt_heads_only, actor_mode, step_penalty_reward)
  if not os.path.exists(path):
    raise FileNotFoundError(
        f'No checkpoint found at {path}. Make sure the previous run used '
        f'the same configuration (seed={seed}, actor_mode={actor_mode}, '
        f'critic_mode={critic_mode}, use_task_id={use_task_id}, '
        f'adapt_heads_only={adapt_heads_only}, '
        f'step_penalty_reward={step_penalty_reward}).')
  with open(path, 'rb') as f:
    data = pickle.load(f)
  data_jax = jax.tree_util.tree_map(
      lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, data)
  print(f'  [ckpt] Loaded <- {path}', flush=True)
  return data_jax


def find_resume_task(
    ckpt_dir: str,
    num_tasks: int,
    seed: int,
    critic_mode: str = 'persistent',
    use_task_id: bool = True,
    adapt_heads_only: bool = True,
    actor_mode: str = 'cka',
    step_penalty_reward: bool = True,
) -> Optional[int]:
  """Probe backwards for the newest checkpoint; return the task to resume at.

  Returns ``task_id + 1`` of the newest checkpoint found, or ``None`` when no
  checkpoint matches this exact config key.  A return value equal to
  ``num_tasks`` means the sequence is already finished, so re-running a
  completed configuration is a safe no-op.
  """
  for probe_tid in range(num_tasks - 1, -1, -1):
    probe = ckpt_path(ckpt_dir, probe_tid, seed, critic_mode, use_task_id,
                      adapt_heads_only, actor_mode, step_penalty_reward)
    if os.path.exists(probe):
      return probe_tid + 1
  return None
