"""Cross-task checkpoint I/O for the continual SAC baseline.

Mirrors ``run_continual_contrastive.py``'s ``_ckpt_path`` / ``save_ckpt`` /
``load_ckpt`` trio, with two SAC-specific differences:

* The config key carries both a ``_rew_{steppen,sparse01}`` suffix and the HER
  reach threshold. The critic's TD target is trained against one specific
  reward definition, so checkpoints with different reward shapes or thresholds
  must never collide.
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


def threshold_tag(her_reward_threshold: float) -> str:
  """Stable filesystem-safe fragment identifying the HER reach threshold."""
  value = format(float(her_reward_threshold), '.12g')
  return value.replace('-', 'm').replace('.', 'p').replace('+', '')


def _legacy_config_key(
    critic_mode: str = 'persistent',
    use_task_id: bool = True,
    adapt_heads_only: bool = True,
    actor_mode: str = 'cka',
    step_penalty_reward: bool = True,
) -> str:
  """Pre-threshold checkpoint key, used only for migration diagnostics."""
  return (f'actor_{actor_mode}_critic_{critic_mode}'
          f'_tid_{use_task_id}_heads_{adapt_heads_only}'
          f'_rew_{reward_tag(step_penalty_reward)}')


def config_key(
    critic_mode: str = 'persistent',
    use_task_id: bool = True,
    adapt_heads_only: bool = True,
    actor_mode: str = 'cka',
    step_penalty_reward: bool = True,
    her_reward_threshold: float = 0.05,
) -> str:
  """Directory name uniquely identifying an ablation cell."""
  return (_legacy_config_key(
      critic_mode, use_task_id, adapt_heads_only, actor_mode,
      step_penalty_reward)
          + f'_tau_{threshold_tag(her_reward_threshold)}')


def ckpt_path(
    ckpt_dir: str,
    task_id: int,
    seed: int,
    critic_mode: str = 'persistent',
    use_task_id: bool = True,
    adapt_heads_only: bool = True,
    actor_mode: str = 'cka',
    step_penalty_reward: bool = True,
    her_reward_threshold: float = 0.05,
) -> str:
  """Checkpoint path keyed by ablation-relevant config (``alg=sac_her`` implicit)."""
  return os.path.join(
      ckpt_dir,
      config_key(critic_mode, use_task_id, adapt_heads_only, actor_mode,
                 step_penalty_reward, her_reward_threshold),
      f'seed_{seed}',
      f'task_{task_id}.pkl')


def save_ckpt(ckpt_dir: str, task_id: int, seed: int, data: Dict[str, Any],
              critic_mode: str = 'persistent', use_task_id: bool = True,
              adapt_heads_only: bool = True, actor_mode: str = 'cka',
              step_penalty_reward: bool = True,
              her_reward_threshold: float = 0.05) -> str:
  """Pickle ``data`` (JAX arrays converted to numpy) and return the path."""
  path = ckpt_path(ckpt_dir, task_id, seed, critic_mode, use_task_id,
                   adapt_heads_only, actor_mode, step_penalty_reward,
                   her_reward_threshold)
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
              step_penalty_reward: bool = True,
              her_reward_threshold: float = 0.05) -> Dict[str, Any]:
  """Load a checkpoint, converting numpy arrays back to JAX arrays."""
  path = ckpt_path(ckpt_dir, task_id, seed, critic_mode, use_task_id,
                   adapt_heads_only, actor_mode, step_penalty_reward,
                   her_reward_threshold)
  if not os.path.exists(path):
    legacy_path = os.path.join(
        ckpt_dir,
        _legacy_config_key(
            critic_mode, use_task_id, adapt_heads_only, actor_mode,
            step_penalty_reward),
        f'seed_{seed}', f'task_{task_id}.pkl')
    if os.path.exists(legacy_path):
      raise FileNotFoundError(
          f'No threshold-keyed checkpoint found at {path}, but a legacy '
          f'checkpoint exists at {legacy_path}. The legacy path does not '
          f'record her_reward_threshold, so loading it automatically would '
          f'be ambiguous. If you know it used '
          f'her_reward_threshold={her_reward_threshold}, move it explicitly '
          f'to {path}; otherwise rerun the cell.')
    raise FileNotFoundError(
        f'No checkpoint found at {path}. Make sure the previous run used '
        f'the same configuration (seed={seed}, actor_mode={actor_mode}, '
        f'critic_mode={critic_mode}, use_task_id={use_task_id}, '
        f'adapt_heads_only={adapt_heads_only}, '
        f'step_penalty_reward={step_penalty_reward}, '
        f'her_reward_threshold={her_reward_threshold}).')
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
    her_reward_threshold: float = 0.05,
) -> Optional[int]:
  """Probe backwards for the newest checkpoint; return the task to resume at.

  Returns ``task_id + 1`` of the newest checkpoint found, or ``None`` when no
  checkpoint matches this exact config key.  A return value equal to
  ``num_tasks`` means the sequence is already finished, so re-running a
  completed configuration is a safe no-op.
  """
  for probe_tid in range(num_tasks - 1, -1, -1):
    probe = ckpt_path(ckpt_dir, probe_tid, seed, critic_mode, use_task_id,
                      adapt_heads_only, actor_mode, step_penalty_reward,
                      her_reward_threshold)
    if os.path.exists(probe):
      return probe_tid + 1
  return None
