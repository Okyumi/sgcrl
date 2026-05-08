"""Linear-probe task classifier (D6) — diagnoses task-identity leakage.

Plan section 3.4: a healthy ``b_shared`` (decomposed critic) should
produce features that do **not** easily classify task identity. If a
linear classifier can predict task index from ``b_shared(s, a)`` with
high accuracy, the body is absorbing task identity rather than learning
task-agnostic dynamics features.

The script runs entirely on CPU (a few minutes of work for 10 tasks at
the default 256 sample size) and produces:

- overall probe accuracy (train and test)
- per-task probe accuracy (test)
- chance level (1 / num_tasks)
- one-row-per-task confusion matrix

Inputs (all written by ``run_continual_contrastive.py`` when
``continual_config.log_probe_data=True``):

- ``probe_data_task{k}_seed{s}.npz`` for k = 0..K-1, holding
  ``obs (N, 2*obs_dim)``, ``action (N, A)``, ``task_id``, ``obs_dim``.
- ``task_{K-1}.pkl`` — the final-task checkpoint, used to rebuild the
  trained network and load its parameters.

Run from the project root (decomposed checkpoint):

    python eval_linear_probe.py \\
        --checkpoint_dir=logs/continual_goal_crl \\
        --seed=42 \\
        --num_tasks=10 \\
        --critic_mode=decomposed \\
        --actor_mode=reset

Run on a non-decomposed run (probes the existing critic's sa-encoder
hidden output, the natural analog of ``b_shared``):

    python eval_linear_probe.py \\
        --checkpoint_dir=logs/continual_goal_crl \\
        --seed=42 \\
        --num_tasks=10 \\
        --critic_mode=cka \\
        --actor_mode=cka
"""
from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

from absl import app, flags
import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from dm_env import specs as dm_specs

# Make sure the project root is importable when running from anywhere.
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
  sys.path.insert(0, _THIS_DIR)

from contrastive import config as contrastive_config
from contrastive import networks as contrastive_networks
from contrastive.continual_config import ContinualConfig
from contrastive.decomposed_networks import make_decomposed_networks


FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'logs/continual_goal_crl',
                    'Root directory for per-config checkpoints.')
flags.DEFINE_integer('seed', 42, 'Seed used during training.')
flags.DEFINE_integer('num_tasks', 10, 'Number of tasks to probe.')
flags.DEFINE_string('critic_mode', 'decomposed',
                    "'decomposed' | 'cka' | 'persistent' | 'reset'.")
flags.DEFINE_string('actor_mode', 'reset',
                    "'reset' | 'persistent' | 'cka'.")
flags.DEFINE_bool('use_task_id', False,
                  'Whether the run used the TaskIDGymWrapper one-hot.')
flags.DEFINE_bool('adapt_heads_only', True,
                  "Used in the checkpoint dir key.")
flags.DEFINE_float('test_fraction', 0.2,
                   'Fraction of each task\'s probe data held out for test.')
flags.DEFINE_float('ridge', 1e-4,
                   'L2 ridge added to the lstsq normal equations for '
                   'numerical stability. 0 disables ridging.')


# ---------------------------------------------------------------------------
# Checkpoint / data loading
# ---------------------------------------------------------------------------

def _config_dir(ckpt_dir: str, critic_mode: str, actor_mode: str,
                use_task_id: bool, adapt_heads_only: bool, seed: int) -> str:
  """Mirrors ``run_continual_contrastive._ckpt_path``'s directory layout."""
  config_key = (f'actor_{actor_mode}_critic_{critic_mode}'
                f'_tid_{use_task_id}_heads_{adapt_heads_only}')
  return os.path.join(ckpt_dir, config_key, f'seed_{seed}')


def _load_probe_data(config_dir: str, num_tasks: int, seed: int):
  """Return list of (obs, action, task_id) per task; raise if missing."""
  per_task = []
  for k in range(num_tasks):
    path = os.path.join(config_dir, f'probe_data_task{k}_seed{seed}.npz')
    if not os.path.exists(path):
      raise FileNotFoundError(
          f'Missing probe data for task {k}: {path}\n'
          f'Re-run training with continual_config.log_probe_data=True '
          f'to populate these files.')
    npz = np.load(path)
    obs = np.asarray(npz['obs'], dtype=np.float32)
    act = np.asarray(npz['action'], dtype=np.float32)
    tid = int(npz['task_id'])
    obs_dim = int(npz['obs_dim'])
    per_task.append((obs, act, tid, obs_dim))
  # Sanity: all tasks share the same obs_dim and full obs shape.
  obs_dims = {t[3] for t in per_task}
  shapes = {t[0].shape[1] for t in per_task}
  if len(obs_dims) != 1:
    raise ValueError(f'obs_dim varies across tasks: {obs_dims}')
  if len(shapes) != 1:
    raise ValueError(f'full-obs width varies across tasks: {shapes}')
  return per_task


def _load_final_ckpt(config_dir: str, last_task_id: int):
  path = os.path.join(config_dir, f'task_{last_task_id}.pkl')
  if not os.path.exists(path):
    raise FileNotFoundError(
        f'Final checkpoint missing: {path}. Did the run complete '
        f'task {last_task_id}?')
  with open(path, 'rb') as f:
    data = pickle.load(f)
  # Convert numpy arrays back to JAX
  return jax.tree_util.tree_map(
      lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
      data)


# ---------------------------------------------------------------------------
# Synthetic env spec (avoids loading mujoco / metaworld for eval-only)
# ---------------------------------------------------------------------------

def _make_env_spec(full_obs_dim: int, action_dim: int) -> specs.EnvironmentSpec:
  return specs.EnvironmentSpec(
      observations=dm_specs.Array(shape=(full_obs_dim,), dtype=np.float32),
      actions=dm_specs.BoundedArray(
          shape=(action_dim,), dtype=np.float32,
          minimum=-1.0, maximum=1.0),
      rewards=dm_specs.Array(shape=(), dtype=np.float32),
      discounts=dm_specs.BoundedArray(
          shape=(), dtype=np.float32, minimum=0.0, maximum=1.0),
  )


# ---------------------------------------------------------------------------
# Feature extraction: branches on critic_mode
# ---------------------------------------------------------------------------

def _build_decomposed_apply(env_spec, obs_dim: int, ckpt: dict, cfg):
  """Rebuild the decomposed b_shared and bind its trained params."""
  nets = make_decomposed_networks(
      env_spec, obs_dim=obs_dim,
      repr_dim=cfg.repr_dim,
      use_residual=cfg.use_residual,
      network_width=cfg.network_width,
      critic_depth=cfg.critic_depth,
      phi_task_width=getattr(cfg, 'phi_task_width', 256),
      phi_task_depth=getattr(cfg, 'phi_task_depth', 2),
      energy_fn=cfg.energy_fn,
      repr_norm=cfg.repr_norm,
  )
  params = ckpt.get('decomposed_b_shared_params')
  if params is None:
    raise ValueError(
        'Checkpoint has no decomposed_b_shared_params. Was the run '
        'trained with --critic_mode=decomposed?')
  apply = nets.apply_b_shared

  def feat_fn(obs, action):
    return apply(params, obs, action)
  return feat_fn, int(nets.hidden_dim)


def _build_qcritic_apply(env_spec, obs_dim: int, ckpt: dict, cfg):
  """Fallback: probe the existing critic's sa-encoder hidden output.

  Plan section 3.4 is decomposed-only, but the user opted into a
  fallback for non-decomposed columns so we can compare the decomposed
  ``b_shared`` against the existing critic body. Both share the
  ``ResidualMLP`` body shape (network_width / critic_depth), so the
  hidden_dim matches. The Haiku module name 'sa_encoder' is shared with
  q_network, so we can call ``critic_hidden_repr_fn`` with the same
  ``q_params`` pytree the learner saved.
  """
  nets = contrastive_networks.make_networks(
      env_spec, obs_dim=obs_dim,
      repr_dim=cfg.repr_dim, repr_norm=cfg.repr_norm,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      use_residual=cfg.use_residual,
      network_width=cfg.network_width,
      critic_depth=cfg.critic_depth,
      actor_depth=cfg.actor_depth,
      energy_fn=cfg.energy_fn,
  )
  q_params = ckpt.get('q_params')
  if q_params is None:
    raise ValueError(
        'Checkpoint has no q_params. Cannot probe a decomposed run '
        'with --critic_mode != decomposed; pass --critic_mode=decomposed '
        'if this is a decomposed run.')

  def feat_fn(obs, action):
    sa_hidden, _g_hidden = nets.critic_hidden_repr_fn(q_params, obs, action)
    return sa_hidden
  hidden_dim = int(cfg.network_width)
  return feat_fn, hidden_dim


# ---------------------------------------------------------------------------
# Linear probe: closed-form softmax via least-squares on one-hot targets
# ---------------------------------------------------------------------------

def _train_test_split(obs, act, tid, test_frac: float, rng: np.random.Generator):
  n = obs.shape[0]
  perm = rng.permutation(n)
  n_test = max(1, int(round(n * test_frac)))
  test_idx = perm[:n_test]
  train_idx = perm[n_test:]
  return ((obs[train_idx], act[train_idx]),
          (obs[test_idx], act[test_idx]),
          tid)


def _fit_linear_probe(features_train: jnp.ndarray,
                      labels_train: jnp.ndarray,
                      num_classes: int,
                      ridge: float) -> jnp.ndarray:
  """Fit ``W`` minimising ``|| Phi W - one_hot(y) ||_F^2 + ridge ||W||^2``.

  The solution is ``W = (Phi^T Phi + ridge * I)^{-1} Phi^T Y``. We use
  ``jnp.linalg.solve`` on the regularised normal equations rather than
  ``lstsq`` because the ridge keeps the system well-conditioned even
  when ``hidden_dim > N`` (rare here but cheap to guard against).
  Adds a bias column to ``features``.

  Returns:
    ``W`` of shape ``(D + 1, num_classes)``.
  """
  N, D = features_train.shape
  Phi = jnp.concatenate([features_train, jnp.ones((N, 1), dtype=features_train.dtype)], axis=1)
  Y = jax.nn.one_hot(labels_train, num_classes)
  A = Phi.T @ Phi + ridge * jnp.eye(D + 1, dtype=Phi.dtype)
  B = Phi.T @ Y
  W = jnp.linalg.solve(A, B)
  return W


def _predict(features: jnp.ndarray, W: jnp.ndarray) -> jnp.ndarray:
  N = features.shape[0]
  Phi = jnp.concatenate([features, jnp.ones((N, 1), dtype=features.dtype)], axis=1)
  scores = Phi @ W
  return jnp.argmax(scores, axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_confusion_matrix(cm: np.ndarray) -> None:
  """Pretty-print a square confusion matrix with row-normalised counts."""
  K = cm.shape[0]
  header = '       ' + ' '.join(f'  pr{j:>2d}' for j in range(K))
  print(header)
  print('       ' + '-' * (len(header) - 7))
  for i in range(K):
    row = cm[i]
    rs = row.sum() or 1
    cells = ' '.join(f'{c / rs:5.2f}' for c in row)
    print(f'  t{i:>2d} | {cells}  (n={int(row.sum())})')


def main(_):
  if FLAGS.test_fraction <= 0 or FLAGS.test_fraction >= 1:
    raise ValueError(f'--test_fraction must be in (0, 1), got {FLAGS.test_fraction}')

  config_dir = _config_dir(
      FLAGS.checkpoint_dir, FLAGS.critic_mode, FLAGS.actor_mode,
      FLAGS.use_task_id, FLAGS.adapt_heads_only, FLAGS.seed)
  print(f'Config dir: {config_dir}')

  per_task = _load_probe_data(config_dir, FLAGS.num_tasks, FLAGS.seed)
  obs_dim = per_task[0][3]
  full_obs_dim = per_task[0][0].shape[1]
  action_dim = per_task[0][1].shape[1]
  print(f'Loaded probe data for {FLAGS.num_tasks} tasks; '
        f'obs_dim={obs_dim}, full_obs_dim={full_obs_dim}, '
        f'action_dim={action_dim}, batch_size={per_task[0][0].shape[0]}')

  # The ContrastiveConfig defaults match the training defaults that
  # made it into the checkpoint (network_width, critic_depth, etc.).
  # ContinualConfig's phi_task_* defaults are read from the dataclass
  # in the same way. For configurations that override these defaults,
  # users should pass the same flags they used at training time —
  # currently we accept the project defaults; flagging this in docs.
  cfg = contrastive_config.ContrastiveConfig()
  cfg.obs_dim = obs_dim  # required by make_networks code paths
  cont_cfg = ContinualConfig()

  ckpt = _load_final_ckpt(config_dir, last_task_id=FLAGS.num_tasks - 1)
  env_spec = _make_env_spec(full_obs_dim, action_dim)

  if FLAGS.critic_mode == 'decomposed':
    feat_fn, hidden_dim = _build_decomposed_apply(env_spec, obs_dim, ckpt, cfg)
    probe_target = 'decomposed_b_shared'
  else:
    feat_fn, hidden_dim = _build_qcritic_apply(env_spec, obs_dim, ckpt, cfg)
    probe_target = 'q_network_sa_encoder_hidden'
  print(f'Probe target: {probe_target} (hidden_dim={hidden_dim})')

  # Compute features per task; split into train / test in a reproducible way.
  rng = np.random.default_rng(FLAGS.seed + 4242)
  feats_train_chunks, labels_train_chunks = [], []
  feats_test_chunks, labels_test_chunks = [], []
  per_task_test = []  # for per-task accuracy

  for (obs, act, tid, _od) in per_task:
    train, test, _ = _train_test_split(obs, act, tid, FLAGS.test_fraction, rng)
    obs_tr, act_tr = train
    obs_te, act_te = test
    f_tr = np.asarray(feat_fn(jnp.asarray(obs_tr), jnp.asarray(act_tr)))
    f_te = np.asarray(feat_fn(jnp.asarray(obs_te), jnp.asarray(act_te)))
    feats_train_chunks.append(f_tr)
    labels_train_chunks.append(np.full(f_tr.shape[0], tid, dtype=np.int32))
    feats_test_chunks.append(f_te)
    labels_test_chunks.append(np.full(f_te.shape[0], tid, dtype=np.int32))
    per_task_test.append((tid, f_te, np.full(f_te.shape[0], tid, dtype=np.int32)))

  X_tr = jnp.asarray(np.concatenate(feats_train_chunks, axis=0))
  y_tr = jnp.asarray(np.concatenate(labels_train_chunks, axis=0))
  X_te = jnp.asarray(np.concatenate(feats_test_chunks, axis=0))
  y_te = jnp.asarray(np.concatenate(labels_test_chunks, axis=0))

  print(f'Train set: {X_tr.shape}, test set: {X_te.shape}')

  W = _fit_linear_probe(X_tr, y_tr, num_classes=FLAGS.num_tasks,
                        ridge=FLAGS.ridge)

  # Train accuracy (sanity)
  pred_tr = _predict(X_tr, W)
  acc_tr = float(jnp.mean(pred_tr == y_tr))
  pred_te = _predict(X_te, W)
  acc_te = float(jnp.mean(pred_te == y_te))
  chance = 1.0 / FLAGS.num_tasks

  print('')
  print(f'==== Linear-probe results ({probe_target}) ====')
  print(f'Chance:        {chance:.4f}  (1 / {FLAGS.num_tasks})')
  print(f'Train accuracy: {acc_tr:.4f}')
  print(f'Test  accuracy: {acc_te:.4f}')
  print('')
  print('Per-task test accuracy:')
  for (tid, f_te, y_te_t) in per_task_test:
    p = _predict(jnp.asarray(f_te), W)
    a = float(jnp.mean(p == jnp.asarray(y_te_t)))
    print(f'  task {tid:>2d}: {a:.4f}')

  # Confusion matrix on the test set
  cm = np.zeros((FLAGS.num_tasks, FLAGS.num_tasks), dtype=np.int64)
  pred_te_np = np.asarray(pred_te)
  y_te_np = np.asarray(y_te)
  for t, p in zip(y_te_np, pred_te_np):
    cm[int(t), int(p)] += 1
  print('')
  print('Row-normalised confusion matrix (rows = true, cols = predicted):')
  _print_confusion_matrix(cm)

  print('')
  print('Interpretation (plan section 3.4):')
  if probe_target.startswith('decomposed'):
    if acc_te < chance + 0.10:
      print(f'  Test accuracy {acc_te:.3f} is near chance ({chance:.3f}). '
            f'b_shared is task-agnostic. PASS.')
    elif acc_te > 0.50:
      print(f'  Test accuracy {acc_te:.3f} is well above chance '
            f'({chance:.3f}). b_shared has absorbed task identity. FAIL.')
    else:
      print(f'  Test accuracy {acc_te:.3f} is intermediate. Borderline; '
            f'check per-task accuracy and pool cosine to disambiguate.')
  else:
    print('  Comparison baseline only; the decomposed plan does not set '
          'a target on this column.')


if __name__ == '__main__':
  app.run(main)
