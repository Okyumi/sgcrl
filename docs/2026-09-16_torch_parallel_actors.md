# Torch paper batch: Jubail-style overlapped collectors

Date: 2026-09-16  
Status: wired for the remaining Torch 9-baseline wave. Cluster was in
maintenance when this was written, so jobs may still need `sbatch`
after `sbatch` is enabled again.

## Motivation

The Jubail DCC batch hid MuJoCo time with `--num_actors=2` while keeping
one `learner.step()` (64 SGD × batch 256) per completed episode. Torch
paper runs were still sequential (`num_actors=1`) and packed **four**
learners per L40S. Copying `num_actors=2` on top of that packing would
be **eight** MuJoCo sims on 12 CPUs, which the Jubail note already
rejected.

## Decision

Follow the Jubail **idea**, not the 4-pack arithmetic:

- `--num_actors=2` (extra collectors only; UTD unchanged)
- `TASKS_PER_GPU=2` (two learners share the L40S)
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`

That is **four MuJoCo processes per GPU**, the same as original SGCRL
and the Jubail 2×2 pack. Expected ~1.3–1.6× env steps/s versus
sequential collection once a hop starts. Default `--num_actors=1` is
unchanged for every other experiment.

## Code and configuration changes

- `DRAFT.sh` forwards `--num_actors=$NUM_ACTORS` (default 1). Config
  `_emit` prints `NUM_ACTORS=2` for the paper first-seed and leftover
  waves.
- `experiment_configs_paper_first_seeds.py` and
  `experiment_configs_paper_remaining_seeds.py` set `num_actors=2`.
- First-seed array is `0-10` (22 runs / 2). Remaining array is `0-33`
  (68 runs / 2). Dispatcher `FIRST_N_ARRAY=11`, `REMAINING_N_ARRAY=34`.
- First-seed `MAX_CHAIN=0`: 11 running GPUs plus 11 pending hops would
  exceed `qos gpu48` MaxTRESPU=16. The CPU dispatcher resubmits timed-out
  `paper_fs` arrays before leftover packs.

Checkpoints and W&B groups are unchanged, so auto-resume still loads
`task_{k}.pkl`.

## Launch command

After maintenance lifts:

```bash
sbatch DRAFT_paper_first_seeds.sh
sbatch DRAFT_paper_cap_dispatcher.sh
```

## Validation

```bash
python tests/test_parallel_actors.py
python tests/test_paper_first_seeds.py
python tests/test_paper_remaining_seeds.py
python tests/test_paper_cap_dispatcher.py
```

## Known limitations

- Policy can lag by one in-flight episode, as in Launchpad `local_mt`.
- 11 first-seed GPUs leave 4 spare slots under the 16-GPU cap (one held
  back). Leftover 10-seed jobs fill those with `nice=100`.
- SAC is still 1 process per GPU (TensorFlow packing crash).
