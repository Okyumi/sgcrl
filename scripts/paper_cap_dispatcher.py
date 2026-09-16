#!/usr/bin/env python3
"""Fill paper GPU jobs without exceeding qos gpu48 MaxTRESPU=16.

Priority: incomplete ``paper_fs`` arrays, then leftover contrastive
packs, then 1-per-GPU SAC. First-seed packing is two learners per L40S
(11 GPUs for 22 runs). Continuations are off (``MAX_CHAIN=0``): 11
running plus 11 pending hops would exceed the 16-GPU TRES cap. This
dispatcher resubmits timed-out array tasks.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CAP = 16
SPARE = 1
SLEEP_SEC = 120
USER = os.environ.get('USER', 'yd2247')

FIRST_LAUNCHER = REPO_ROOT / 'DRAFT_paper_first_seeds.sh'
REMAINING_LAUNCHER = REPO_ROOT / 'DRAFT_paper_remaining_seeds.sh'
SAC_LAUNCHER = REPO_ROOT / 'DRAFT_paper_sparse_sac_10seed.sh'
FIRST_N_ARRAY = 11
REMAINING_N_ARRAY = 34
SAC_N_ARRAY = 20


def _run(args, check=True):
  return subprocess.run(
      args, check=check, capture_output=True, text=True)


def parse_array_ids(job_id):
  """Return array task ids represented by a squeue JOBID token."""
  match = re.search(r'_\[(.+)\]$', job_id)
  if not match:
    tail = job_id.rsplit('_', 1)
    if len(tail) == 2 and tail[1].isdigit():
      return {int(tail[1])}
    return set()
  ids = set()
  for part in match.group(1).split(','):
    if '-' in part:
      start, end = part.split('-', 1)
      ids.update(range(int(start), int(end) + 1))
    elif part.isdigit():
      ids.add(int(part))
  return ids


def gpu_queue():
  result = _run([
      'squeue', '-u', USER, '-t', 'RUNNING,PENDING', '-h',
      '-o', '%i %j %T %b',
  ], check=False)
  jobs = []
  for line in result.stdout.splitlines():
    parts = line.split()
    if len(parts) < 4:
      continue
    job_id, name, state, tres = parts[0], parts[1], parts[2], parts[3]
    if 'gpu' not in tres.lower():
      continue
    array_ids = parse_array_ids(job_id)
    jobs.append({
        'job_id': job_id,
        'name': name,
        'state': state,
        'n': max(1, len(array_ids)),
        'array_ids': array_ids,
    })
  return jobs


def gpu_task_count(jobs):
  return sum(job['n'] for job in jobs)


def queued_array_ids(jobs, name):
  ids = set()
  for job in jobs:
    if job['name'] == name:
      ids.update(job['array_ids'])
  return ids


def parse_incomplete_ids(text):
  if not text.strip():
    return []
  return [int(part) for part in text.strip().split(',') if part.strip()]


def incomplete_status_ids(script_name):
  result = _run([
      sys.executable,
      str(REPO_ROOT / 'scripts' / script_name),
      '--incomplete-array-ids',
  ], check=False)
  if result.returncode != 0:
    raise RuntimeError(
        result.stderr.strip() or result.stdout.strip()
        or f'{script_name} failed')
  return parse_incomplete_ids(result.stdout)


def sbatch_array(launcher, array_id, extra=None):
  cmd = ['sbatch', f'--array={array_id}', str(launcher)]
  if extra:
    cmd[1:1] = extra
  result = _run(cmd, check=False)
  print(result.stdout.strip() or result.stderr.strip(), flush=True)
  return result.returncode == 0


def _submit_missing(name, launcher, ids, extra, jobs, slots, submitted):
  queued = queued_array_ids(jobs, name)
  for array_id in ids:
    if submitted >= slots:
      break
    if array_id in queued:
      continue
    print(f'[dispatcher] submit {name} array {array_id}', flush=True)
    if sbatch_array(launcher, array_id, extra=extra):
      submitted += 1
      queued.add(array_id)
  return submitted, queued


def fill_slots(jobs):
  used = gpu_task_count(jobs)
  slots = GPU_CAP - SPARE - used
  print(f'[dispatcher] gpu tasks={used} cap={GPU_CAP} spare={SPARE} '
        f'slots={slots}', flush=True)
  if slots <= 0:
    return 0

  submitted = 0
  fs_ids = incomplete_status_ids('paper_first_seeds_status.py')
  submitted, fs_queued = _submit_missing(
      'paper_fs', FIRST_LAUNCHER, fs_ids,
      ['--export=ALL,PAPER_FIRST_SEEDS_MAX_CHAIN=0'],
      jobs, slots, submitted)
  missing_fs = sorted(set(fs_ids) - fs_queued)
  if missing_fs:
    print(f'[dispatcher] still need first-seed arrays {missing_fs}',
          flush=True)
    return submitted

  rest_ids = incomplete_status_ids('paper_remaining_seeds_status.py')
  submitted, rest_queued = _submit_missing(
      'paper_rest', REMAINING_LAUNCHER, rest_ids,
      ['--nice=100', '--export=ALL,PAPER_REMAINING_MAX_CHAIN=0'],
      jobs, slots, submitted)
  missing_rest = sorted(set(rest_ids) - rest_queued)
  if missing_rest:
    print(f'[dispatcher] still need contrastive arrays {missing_rest}',
          flush=True)
    return submitted

  sac_queued = queued_array_ids(jobs, 'paper_sac')
  for array_id in range(SAC_N_ARRAY):
    if submitted >= slots:
      break
    if array_id in sac_queued:
      continue
    extra = [
        '--nice=200',
        '--export=ALL,TASKS_PER_GPU=1,PAPER_SPARSE_SAC_MAX_CHAIN=0',
    ]
    print(f'[dispatcher] submit paper_sac array {array_id}', flush=True)
    if sbatch_array(SAC_LAUNCHER, array_id, extra=extra):
      submitted += 1
      sac_queued.add(array_id)
  return submitted


def once():
  jobs = gpu_queue()
  fill_slots(jobs)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--loop', action='store_true')
  parser.add_argument('--sleep', type=int, default=SLEEP_SEC)
  args = parser.parse_args()
  os.chdir(REPO_ROOT)
  if not args.loop:
    once()
    return
  while True:
    try:
      once()
    except Exception as exc:  # noqa: BLE001
      print(f'[dispatcher] error: {exc}', flush=True)
    time.sleep(args.sleep)


if __name__ == '__main__':
  main()
