"""Task checkpoint atomicity and continuation discovery without simulator deps."""
import ast
import importlib.util
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location('vanilla_launch', REPO / 'scripts/vanilla_crl_gate_bc.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class CheckpointTest(unittest.TestCase):
  def test_failed_write_preserves_previous_checkpoint(self):
    module = ast.parse((REPO / 'run_continual_contrastive.py').read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'save_ckpt')
    with tempfile.TemporaryDirectory() as temp:
      path = Path(temp) / 'task_0.pkl'
      scope = dict(os=os, tempfile=tempfile, pickle=pickle, jax=jax, jnp=jnp, np=np,
                   _ckpt_path=lambda *a, **kw: str(path))
      exec(compile(ast.Module(body=[function], type_ignores=[]), '<save_ckpt>', 'exec'), scope)
      save = scope['save_ckpt']
      save(temp, 0, 5, {'value': jnp.array([1.])})
      def fail(data, handle):
        handle.write(b'partial')
        raise OSError('simulated write interruption')
      with mock.patch.object(pickle, 'dump', side_effect=fail):
        with self.assertRaises(OSError):
          save(temp, 0, 5, {'value': jnp.array([2.])})
      with path.open('rb') as handle:
        np.testing.assert_array_equal(pickle.load(handle)['value'], [1.])
      self.assertEqual(list(Path(temp).iterdir()), [path])

  def test_progress_requires_contiguous_tasks(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      self.assertEqual(launch.completed_tasks(root, 5), 0)
      directory = root / 'checkpoints/actor_reset_critic_reset_tid_False_heads_True_success_native_info_bridge_test/seed_5'
      directory.mkdir(parents=True)
      (directory / 'task_0.pkl').touch()
      self.assertEqual(launch.completed_tasks(root, 5), 1)
      (directory / 'task_2.pkl').touch()
      with self.assertRaises(RuntimeError):
        launch.completed_tasks(root, 5)


if __name__ == '__main__':
  unittest.main()
