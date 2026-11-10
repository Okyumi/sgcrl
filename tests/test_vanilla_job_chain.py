"""Exercise Slurm success/failure paths using a fake scheduler, never submitting."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class JobChainTest(unittest.TestCase):
  def run_job(self, run_rc=0, sbatch_rc=0, complete=False):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      (root / 'bin').mkdir()
      (root / 'set_up').mkdir()
      (root / 'set_up/jubail_hpc_env.sh').write_text('true\n')
      (root / 'logs/slurm').mkdir(parents=True)
      commands = {
          'git': '#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo abc; fi\n',
          'module': '#!/bin/bash\nexit 0\n',
          'conda': '#!/bin/bash\nif [[ "$1" == shell.bash ]]; then echo :; fi\n',
          'sbatch': f'#!/bin/bash\necho "sbatch $*" >> "$EVENTS"\necho 222\nexit {sbatch_rc}\n',
          'scancel': '#!/bin/bash\necho "scancel $*" >> "$EVENTS"\n',
          'python': f'''#!/bin/bash
echo "python $*" >> "$EVENTS"
case "$*" in
  *--mode=preflight*) exit 0;;
  *--mode=complete*) exit {0 if complete else 1};;
  *--mode=run*) exit {run_rc};;
esac
exit 99
''',
      }
      for name, content in commands.items():
        p = root / 'bin' / name
        p.write_text(content)
        p.chmod(0o755)
      env = dict(os.environ, PATH=f'{root}/bin:' + os.environ['PATH'],
                 VANILLA_REPO_DIR=str(root), VANILLA_RUN_ROOT=str(root / 'logs'),
                 VANILLA_EXPECTED_COMMIT='abc', VANILLA_MODE='run',
                 VANILLA_CHAIN_INDEX='0', VANILLA_MAX_JOBS='20',
                 SLURM_ARRAY_JOB_ID='111', SLURM_ARRAY_TASK_ID='3',
                 EVENTS=str(root / 'events'), WANDB_API_KEY='dummy')
      result = subprocess.run(['bash', str(REPO / 'scripts/jubail_vanilla_crl_gate_bc.sbatch')],
                              env=env, capture_output=True, text=True)
      events = (root / 'events').read_text()
      return result.returncode, events

  def test_prequeues_next_seed_allocation(self):
    rc, events = self.run_job()
    self.assertEqual(rc, 0)
    self.assertIn('--dependency=afterany:111_3', events)
    self.assertIn('--array=3', events)
    self.assertIn('--seed=8', events)
    self.assertLess(events.index('sbatch'), events.index('--mode=run'))
    self.assertNotIn('scancel', events)

  def test_python_failure_cancels_successor(self):
    rc, events = self.run_job(run_rc=2)
    self.assertEqual(rc, 2)
    self.assertIn('scancel 222', events)

  def test_scheduler_failure_prevents_training(self):
    rc, events = self.run_job(sbatch_rc=1)
    self.assertNotEqual(rc, 0)
    self.assertNotIn('--mode=run', events)

  def test_completed_seed_does_not_queue_or_train(self):
    rc, events = self.run_job(complete=True)
    self.assertEqual(rc, 0)
    self.assertNotIn('sbatch', events)
    self.assertNotIn('--mode=run', events)


if __name__ == '__main__':
  unittest.main()
