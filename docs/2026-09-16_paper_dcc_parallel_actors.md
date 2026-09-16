# Overlapped CPU actors for the Jubail DCC vs Success-BC batch

Date: 2026-09-16  
Status: code on disk; in-flight hop 2 keeps the old sequential process.
The queued 48h continuation loads this runner.

## Motivation

The paper batch (`run_continual_contrastive.py`) collects with one MuJoCo
env, then runs 64 SGD steps, then collects again. Runtime profiles show
collection is ~60% of wall time and the learner ~40%. Original SGCRL hid
that collect time with Launchpad `local_mt` and `num_actors=4`.

This change uses the same idea **without changing DCC, Success-BC, the
dynamics-off setting, network width, or UTD**. Extra actors only collect.
The main thread still calls `learner.step()` once per completed episode
(`num_sgd_steps_per_step=64`, batch 256).

## Mathematical objective

Unchanged. Per completed episode the learner still consumes

\[
64 \times 256
\]

gradient samples. Environment steps per update stay ~one episode
(`max_episode_steps`). The loss, Success-BC buffer, and `μ=0` dyn skip
are untouched.

Policy parameters seen by collectors can lag the learner by up to
`num_actors` in-flight episodes, as in Launchpad `local_mt`.

## Code / config

- `contrastive/parallel_actors.py`: thread pool around Acme
  `EnvironmentLoop.run_episode`. Queue bound = number of actors.
- `run_continual_contrastive.py`: `--num_actors` (default **1**, so every
  other experiment stays sequential and seed-identical). Actor 0 reuses
  the original env/actor seeds.
- `experiment_configs_paper_dcc_success_bc_jubail.py`: `num_actors=2`.
- `DRAFT_jubail.sh`: forwards `--num_actors=$NUM_ACTORS` (default 1).

Two packed jobs × two actors = **four** MuJoCo sims per GPU, the original
SGCRL actor count. Four actors per packed job would be eight sims on 12
CPUs and would not help: once collection overlaps the learner, the GPU
update is the bound.

Expected wall-clock for this packed batch: about **1.3–1.6×** env
steps/s once a hop restarts, not 4×. Exclusive GPU + four actors would
be closer to 2.5× per run but would pause 10 of the 20 jobs.

## Launch command

No resubmit. Hop 3 already queued with `afterany` will exec

```bash
python -u run_continual_contrastive.py ... --num_actors=2
```

from this tree. Do **not** scancel hop 2: mid-task replay is not saved,
so a kill repeats the current 8M-step task.

If a later hop must be launched by hand:

```bash
ARRAY=$(python scripts/paper_dcc_success_bc_jubail_status.py --incomplete-array-ids)
sbatch --array="$ARRAY" DRAFT_paper_dcc_success_bc_jubail.sh
```

## Logged metrics

Same paper metrics. New W&B config field `num_actors`. Runtime profile
`actor_seconds` on the main thread becomes wait-for-episode time, so it
should drop relative to `learner_seconds` if overlap works.

## Validation

```bash
python tests/test_parallel_actors.py
python tests/test_paper_dcc_success_bc_jubail.py
```

## Known limitations

- Threads, not Launchpad processes (`local_mt` style).
- `num_actors=1` is still the default outside this paper config.
- Hop 2 (already running) will not speed up until it times out or
  finishes a 48h slice and the continuation starts.
