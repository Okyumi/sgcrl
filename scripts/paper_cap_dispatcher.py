#!/usr/bin/env python3
"""Fill leftover paper GPU jobs without exceeding qos gpu48 MaxTRESPU=16.

Waits until ``paper_fs`` is actually running (so its 48h continuations are
already in the pending count), then submits leftover contrastive packs
and 1-per-GPU SAC only into free slots. Remaining jobs are launched with
``MAX_CHAIN=0`` so they do not spawn extra pending GPUs; this dispatcher
resubmits timed-out array tasks.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CAP = 16
SPARE = 1
SLEEP_SEC = 120
USER = os.environ.get('USER', 'yd2247')

REMAINING_LAUNCHER = REPO_ROOT / 'DRAFT_paper_remaining_seeds.sh'
SAC_LAUNCHER = REPO_ROOT / 'DRAFT_paper_sparse_sac_10seed.sh'
REMAINING_N_ARRAY = 17
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


def paper_fs_running(jobs):
  return any(job['name'] == 'paper_fs' and job['state'] == 'RUNNING'
             for job in jobs)


def paper_fs_present(jobs):
  return any(job['name'] == 'paper_fs' for job in jobs)


def sbatch_array(launcher, array_id, extra=None):
  cmd = ['sbatch', f'--array={array_id}', str(launcher)]
  if extra:
    cmd[1:1] = extra
  result = _run(cmd)
  print(result.stdout.strip() or result.stderr.strip(), flush=True)
  return result.returncode == 0


def fill_slots(jobs):
  used = gpu_task_count(jobs)
  slots = GPU_CAP - SPARE - used
  print(f'[dispatcher] gpu tasks={used} cap={GPU_CAP} spare={SPARE} '
        f'slots={slots}', flush=True)
  if slots <= 0:
    return 0

  submitted = 0
  rest_queued = queued_array_ids(jobs, 'paper_rest')
  for array_id in range(REMAINING_N_ARRAY):
    if submitted >= slots:
      return submitted
    if array_id in rest_queued:
      continue
    extra = [
        '--nice=100',
        '--export=ALL,PAPER_REMAINING_MAX_CHAIN=0',
    ]
    print(f'[dispatcher] submit paper_rest array {array_id}', flush=True)
    if sbatch_array(REMAINING_LAUNCHER, array_id, extra=extra):
      submitted += 1
      rest_queued.add(array_id)

  if rest_queued != set(range(REMAINING_N_ARRAY)):
    missing = sorted(set(range(REMAINING_N_ARRAY)) - rest_queued)
    print(f'[dispatcher] still need contrastive arrays {missing}',
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
  return submitted


def once():
  jobs = gpu_queue()
  if not paper_fs_running(jobs):
    if paper_fs_present(jobs):
      print('[dispatcher] paper_fs pending; not filling yet', flush=True)
    else:
      print('[dispatcher] paper_fs not in queue; filling leftover only',
            flush=True)
      fill_slots(jobs)
    return
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
