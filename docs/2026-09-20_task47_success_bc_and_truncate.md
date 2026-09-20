# Task 4/7 Success-BC diagnosis and Task 5/8 success truncation

Date: 2026-09-20  
Status: Task-5/8 freeze-on-success finished (`18032606`). Truncation
does not retain like Success-BC.

## Motivation

Paper 10-seed DCC (`dyn=0`, `native_info`, residual 1024×4) looked worse on
Task 4 (`sawyer_stick_pull`) and Task 7 (`sawyer_shelf_place`) after adding
Success-BC (`λ_succ=0.1`) than plain DCC. Need (1) learning curves, (2)
rollout GIFs from the seed-6 final policies, and (3) a first test of the
professor’s alternative: **truncate collection on first success**, put the
truncated episode in replay, and **do not** use a success buffer.

## Why λ=0.1 can still dominate the actor

The actor update is

\[
L_{\pi}
=
\mathbb{E}\bigl[-f(s,a,g)\bigr]
+
\lambda_{\mathrm{succ}}\,
\mathbf{1}[|\mathcal{D}_{\mathrm{succ}}|>0]\,
\bigl(-\mathbb{E}_{(s,a)\sim\mathcal{D}_{\mathrm{succ}}}\log\pi(a\mid s,g_{\mathrm{task}})\bigr).
\]

`λ_succ=0.1` is a weight on **NLL**, not on a unit-scale penalty. A
4-D tanh-normal can have NLL of order 5–20, while the DCC term
`mean(-Q)` is often O(0.3–2). The logged ratio

`retention/bc_to_dcc_loss_ratio = |λ NLL| / |mean(-Q)|`

is therefore allowed to be O(0.2–2) even though λ looks small. The BC
scalar is also **broadcast onto every DCC sample** in the batch, so it
shifts the whole actor step, not a tiny auxiliary.

## Why Tasks 4 and 7 are the fragile ones

`success_bc_label_mode=episode_sparse_reward` clones **every** transition
in a 150-step episode that ever saw a sparse reward, stored under the
**task goal**. On latch tasks (5/8) lingering after success is cheap. On
stick-pull / shelf-place it is not:

- post-success actions can knock the stick out or drop the object;
- early lucky / sloppy successes fill the 4096 ring buffer and keep
  being cloned;
- prefixes of a multi-phase skill are cloned as if they were already
  the final-goal policy.

So the actor can lock onto a suboptimal clone rather than keep improving
via DCC/HER. That is the working hypothesis; the seed-6 GIFs and the
mean±std curves are the check.

## Task 4/7 videos

Paper cells only keep `task_{k}.pkl` (no mid-task snapshots). Seed 6,
10 eval rollouts + first-success hunt, 320px GIFs:

```
logs/jubail_task47_videos/gifs/{stick_plain,stick_bc,shelf_plain,shelf_bc}/
```

Checkpoints:

```
logs/paper_dcc_success_bc_jubail_checkpoints/10seed/
  ..._dyn0.000_pt256x4/seed_6/task_{4,7}.pkl
  ..._dyn0.000_pt256x4_bridge_d2186abe4046_ret_847a4481eb68/seed_6/task_{4,7}.pkl
```

Launch:

```bash
sbatch DRAFT_jubail_task47_videos.sh
```

## Success truncation (Tasks 5 and 8)

Collection episodes stay 150 steps (Reverb batches fixed-length
episodes), but after the first sparse success the wrapper **freezes** the
last successful observation instead of letting the arm linger. HER then
sees a success-state future with no post-goal physics. Eval is unwrapped.

Diagnosis stack matches the existing Task-5/8 GIF jobs: decomposed DCC,
actor reset, corrected wrapper, seed 6, 1M, 1024×4, `dyn_aux_weight=1`.

```bash
sbatch DRAFT_jubail_success_truncate.sh
```

W&B group `TASK58-JUBAIL-SUCCESS-TRUNCATE`. Job `18032606` finished
(~1h55m). Seed-6 1M finals vs the same diagnosis stack:

| task | freeze-on-success | plain DCC | Success-BC |
|---|---|---|---|
| 5 handle_press_side | **20%** (peak 40% @ 300k) | **0%** (peak 20%) | **100%** |
| 8 window_close | **10%** (peak 50% @ 250k) | **0%** (peak 60% @ 250k) | **100%** |

Truncation discovers then mostly forgets, like plain DCC. It is not a
substitute for the success buffer on latch tasks.

Compare jobs: freeze `18032606`; plain handle `17901349_0`; plain window
`local_0_0` under action-mass; Success-BC handle `17982404_2`, window
`17988666_4`.

## Code / config

- `env_utils.SuccessTruncateGymWrapper` + `load(..., truncate_on_success)`
- collection `make_environment` in `run_continual_contrastive.py` only
- `--truncate_on_success` (illegal together with Success-BC)
- `scripts/record_checkpoint_rollout_gifs.py` can load `composed_policy`
- `scripts/plot_paper_task47_success_bc.py`

## Logged metrics

W&B 10-seed mean±std of `evaluator/success_rate` (final ~7.9M):

| task | plain DCC | Success-BC λ=0.1 |
|---|---|---|
| 4 stick_pull | **0.80 ± 0.13** | **0.44 ± 0.39** |
| 7 shelf_place | **0.61 ± 0.23** | **0.23 ± 0.20** |

Success-BC is bimodal, not uniformly stuck. Task 4 BC seeds 8/9 collapse to
0; seeds 5/7 finish at 1.0. Plot:

```
logs/paper_task47_curves/task47_curves.png
logs/paper_task47_curves/task47_curves.csv
```

Seed-6 GIFs (10 eval rollouts + first-success hunt):

```
logs/jubail_task47_videos/gifs/stick_plain/   9/10 success
logs/jubail_task47_videos/gifs/stick_bc/      7/10 success
logs/jubail_task47_videos/gifs/shelf_plain/   8/10 success
logs/jubail_task47_videos/gifs/shelf_bc/      5/10 success
```

SLURM GIF job `18032586`. Truncation relaunch `18032606` after Reverb
rejected mixed-length episodes; collection now freezes the success state
instead of ending the episode early.


- Paper Task 4/7 GIFs are **final policies**, not training snapshots.
- Truncation test is 1-seed 1M on Tasks 5/8 only.
- If MetaWorld `info['success']` is missing, truncation falls back to
  sparse reward ≥ 1.
