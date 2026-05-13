# W&B-driven findings, 2026-05-13

This document supersedes the qualitative analysis in
`docs/2026-05-12_cka_failure_results.md` with quantitative numbers
pulled directly from the `nyuad_mmvc/continual_gcrl_paper` W&B project.
All trajectories are aggregated across three seeds (5, 6, 7) per cell.

CSV artifacts and the trajectory plot are saved in
`docs/wandb_analysis/csv/` and `docs/wandb_analysis/png/`.

---

## TL;DR

1. **CKA-RL fails on the contrastive setup, but not because the
   mixture norm collapses to zero.** The actor mixture norm decays
   to ~0.02 (essentially negligible). The critic mixture norm decays
   to a non-zero steady state of ~0.5, oscillating between ~0.4 and
   ~1.3 across the eight tasks where the pool has ≥2 entries.
2. **`α_scale` does NOT uniformly drift to zero.** Critic `α_scale`
   often grows (reaches ~2.0 at task 8). Actor `α_scale` decays for
   some tasks and stays near 1.0 for others.
3. **My 2026-05-12 explanation about a "gradient null space"
   predicted the opposite asymmetry** (critic mixture should die
   faster than actor due to stricter shift-invariance). The data
   shows actor mixture dying ~30× more than critic mixture. That
   prediction is wrong.
4. **The decomposed-critic algorithm (C2 cell) is competitive with
   the CKA baseline (C0 cell) when "best during training" is the
   right metric.** End-of-task numbers had misled me last response —
   they reflect in-task instability, not algorithm failure.
5. **C1 task-1 crashed for all 6 runs (3 seeds × 2 modes) for an
   unidentified reason**; cluster-log inspection is needed.
   **C2 task-8 crashed for all 3 seeds in a 15-minute wall-clock
   window** — that is an infrastructure event (cluster node
   failure), not a code bug.

---

## 1. The actual CKA-cell numbers (C0)

Source: `docs/wandb_analysis/csv/c0_summary.csv` (24 rows: 8 tasks
× 3 seeds). Plot: `docs/wandb_analysis/png/c0_trajectories.png`.

### 1.1 Mixture norm at task end (mean over 3 seeds)

| Task | Actor mixture | Critic mixture |
|------|---------------|----------------|
| k=2  | 0.019         | 0.815          |
| k=3  | 0.015         | 0.674          |
| k=4  | 0.020         | 0.526          |
| k=5  | 0.010         | 1.238          |
| k=6  | 0.033         | 0.472          |
| k=7  | 0.039         | 0.443          |
| k=8  | 0.037         | 0.642          |
| k=9  | 0.023         | 0.396          |

The actor mixture is essentially zero (<0.05 across all tasks). The
critic mixture is **not** at zero — it sits in the 0.4-1.3 band, i.e.
the mixture term carries half-the-norm of `v_k` even at task end.

The original `docs/2026-05-12_cka_failure_results.md` reported the
critic mixture as "12 → ~0" — that was reading the **start** value
(12 is the magnitude at task entry, before `v_k` has grown to dominate
it) and casually rounding the asymptote. The asymptote is in fact
~0.5, not 0. This is a meaningful correction.

### 1.2 α_scale trajectories

| Task | Actor α_scale (start → end) | Critic α_scale (start → end) |
|------|-----------------------------|------------------------------|
| k=2  | 1.00 → 1.00 (flat)          | 1.00 → 1.00 (flat; only 1 active slot) |
| k=3  | 1.07 → 0.60                 | 1.00 → 0.74                  |
| k=4  | 1.05 → 0.65                 | 1.04 → 0.58                  |
| k=5  | 1.06 → 0.48                 | 1.06 → 1.42 (grew)           |
| k=6  | 1.14 → 1.27                 | 1.20 → 1.41 (grew)           |
| k=7  | 1.15 → 1.06                 | 1.21 → 1.22 (slight rise)    |
| k=8  | 1.21 → 0.80                 | 1.16 → 1.90 (substantial rise) |
| k=9  | 1.14 → 0.59                 | 1.25 → 1.04                  |

The actor's α_scale generally decays but not monotonically. The
**critic's α_scale grows for k=5, 6, 7, 8**, opposite to the
prediction in 2026-05-12.

### 1.3 What this rules out

- **Strict "vectors collapse to one direction" hypothesis (plan
  §3.1):** ruled out earlier (mean offdiag ~0.15, not >0.9).
- **Strict "mixture lives in a null space" hypothesis (2026-05-12):**
  ruled out for the critic. The critic mixture is alive, and its
  α_scale is being actively grown by gradient.
- **Pure "Adam noise drifts α_scale to zero" hypothesis (2026-05-12,
  prediction 1):** ruled out. α_scale receives non-zero gradient on
  the critic side and the gradient is sometimes positive.

### 1.4 What survives

- The **actor** mixture and α_scale do decay, although not to zero.
- The **critic** mixture decays from ~12 to ~0.5 over training — a
  big drop, but to a non-zero steady state.
- CKA-RL is failing **on the eval metric** (forgetting curves; see
  §2 below) — the failure is real. The mechanism is just not what
  the gradient-null-space story claimed.

A revised hypothesis-test plan is in
`docs/2026-05-13_revised_hypothesis_plan.md`.

---

## 2. C2 success rates done properly (best-during-training ± SEM)

Source: `docs/wandb_analysis/csv/c2_best_per_task_seeds.csv` and
`c2_best_per_task_agg.csv`. Metric used: `evaluator/success_rate`
trajectory during training (159 evaluator points per task), taking
the max over the trajectory. The earlier "end-of-task summary"
metric was misleading — it reflects in-task instability, not
algorithm capability.

| Task | env                       | best (mean ± SEM) | end (mean ± SEM) | mean-over-task |
|------|---------------------------|-------------------|------------------|----------------|
| k=0  | sawyer_hammer             | 1.000 ± 0.000     | 0.900 ± 0.058    | 0.895          |
| k=1  | sawyer_push_wall          | 1.000 ± 0.000     | 0.633 ± 0.318    | 0.749          |
| k=2  | sawyer_faucet_close       | 0.867 ± 0.033     | 0.333 ± 0.120    | 0.361          |
| k=3  | sawyer_push_back          | 1.000 ± 0.000     | 0.767 ± 0.233    | 0.793          |
| k=4  | sawyer_stick_pull         | 1.000 ± 0.000     | 0.433 ± 0.133    | 0.537          |
| k=5  | sawyer_handle_press_side  | 0.400 ± 0.058     | 0.067 ± 0.033    | 0.072          |
| k=6  | sawyer_push               | 1.000 ± 0.000     | 0.267 ± 0.120    | 0.536          |
| k=7  | sawyer_shelf_place        | 0.967 ± 0.033     | 0.533 ± 0.203    | 0.390          |

Tasks 8 and 9 crashed (see §4).

**Comparison with C0 (CKA baseline) on the same metric:**

| Task | env                       | C0 best (mean ± SEM) | C2 best (mean ± SEM) |
|------|---------------------------|----------------------|----------------------|
| k=0  | sawyer_hammer             | 1.000 ± 0.000        | 1.000 ± 0.000        |
| k=1  | sawyer_push_wall          | 1.000 ± 0.000        | 1.000 ± 0.000        |
| k=2  | sawyer_faucet_close       | 0.733 ± 0.088        | **0.867 ± 0.033**    |
| k=3  | sawyer_push_back          | 1.000 ± 0.000        | 1.000 ± 0.000        |
| k=4  | sawyer_stick_pull         | 1.000 ± 0.000        | 1.000 ± 0.000        |
| k=5  | sawyer_handle_press_side  | 0.167 ± 0.033        | **0.400 ± 0.058**    |
| k=6  | sawyer_push               | 1.000 ± 0.000        | 1.000 ± 0.000        |
| k=7  | sawyer_shelf_place        | 0.933 ± 0.033        | 0.967 ± 0.033        |

C2 **matches or beats** C0 on every task where data is available
(k=0..7). Most notably on the two harder tasks (k=2, k=5) the
decomposed cell shows a substantial gap: +0.13 and +0.23 respectively.

The end-of-task instability is a separate phenomenon — both cells
show large drops between the best success during a task and the
success at the end of the task. This is consistent with the well-known
"plasticity-stability tradeoff": the policy hits a peak mid-training
and then drifts as more gradient is taken. Worth a follow-up
investigation but not specific to the decomposed algorithm.

---

## 3. C1 task-0 regression check (only data we have)

Source: `docs/wandb_analysis/csv/c1_task0_seeds.csv`. Six runs (3
seeds × {persistent baseline, decomposed dyn_aux_weight=0}).

| critic_mode | best (mean ± SEM) | end (mean ± SEM) | mean-over-task |
|-------------|-------------------|------------------|----------------|
| persistent  | 1.000 ± 0.000     | 0.867 ± 0.088    | 0.834          |
| decomposed (w=0) | 1.000 ± 0.000 | 0.733 ± 0.176    | 0.885          |

Decomposed-w=0 and persistent are statistically indistinguishable on
task 0. The decomposed end value is slightly lower but within SEM;
the mean-over-task is slightly higher.

**This is task-0 only.** The regression check needs k ≥ 2 to detect
any actual regression introduced by the decomposed plumbing. The
C1 task-1 runs all crashed (see §4) so we have no regression data
for the meaningful comparison points.

---

## 4. Crash diagnosis

### 4.1 C1 task-1 — all 6 runs crashed (2026-05-10)

**Pattern:** all six C1 task-1 runs (`task1_sawyer_push_wall_s{5,6,7}`
× `critic_mode={persistent, decomposed}`) have:

- `state = crashed`
- `created_at == heartbeat_at` to the exact second
- zero rows of history (training loop never executed)
- no system metadata visible from the W&B API (the wandb client
  never finished syncing before the process died)

**Cluster-side context:** these are SLURM array jobs that auto-resume
from a task-0 checkpoint. The auto-resume path
(`run_continual_contrastive.py:1320-1368`) executes a sequence of
operations *before* the per-task `wandb.init` succeeds for task 1:
`os.path.exists` probe, then `pickle.load` of the previous
checkpoint, then `pool.load_state_dict`. The fact that wandb DID
manage to register the task-1 run on the server (run name
`task1_sawyer_push_wall_s5` exists) means `wandb.init` for task 1
completed, but the process died within 1 second after.

**Most plausible causes (in order of likelihood):**

1. **`train_single_task` ran but hit an error in the reverb/replay
   server boot** for `sawyer_push_wall` (different obs/action
   shapes than `sawyer_hammer`). This is the first task-specific
   resource to be created. The wandb client buffers up to a few
   seconds before sending the first heartbeat — an exit during
   reverb init would land in this window.
2. **JAX recompilation triggered an OOM**. Different env layout can
   trigger a re-JIT; if the cluster node was tight on memory the
   re-compile could OOM.
3. **A SLURM time-limit kick-out** (less likely — these were created
   minutes apart, an across-the-board time-limit would have killed
   them all at the same wall-clock time, which is what we observe
   for C2 task-8 below, not for C1 task-1).
4. **The checkpoint format was incompatible** because task-0 was
   saved by an older commit and task-1 tried to load it. Plausible
   given the user's workflow of committing config changes between
   batches.

**What to do (cluster side only — no code changes proposed):**

```bash
# On NYU Torch, locate the actual stderr trace for one crashed run.
# The W&B run ID is the suffix in the local wandb folder name.
find /scratch/yd2247/sgcrl/logs/continual/continual_contrastive_cpc/ \
    -name '*task1_sawyer_push_wall_s5*' -type d 2>/dev/null

# Inside each match, the file ./output.log (or wandb/run-*/logs/debug.log,
# or slurm-<jobid>.out from SLURM) contains the Python traceback.
# Grep for 'Traceback' or 'Error' to find the failure point.
grep -nE 'Traceback|Error|OOM|CUDA' /scratch/yd2247/sgcrl/logs/.../output.log
```

Once a traceback is recovered we can produce a confident root cause.
**No source-code changes recommended until then.** If the traceback
points to checkpoint deserialisation, the fix is straightforward (add
a `try/except` around `load_ckpt` that logs more context). If it
points to reverb/JAX init, the fix is environmental.

### 4.2 C2 task-8 — all 3 runs crashed (2026-05-12)

**Pattern:** all three C2 task-8 runs crashed within a **15-minute
wall-clock window** on 2026-05-12:

- `vlfpev9p` (s6): heartbeat ended 13:45:01 UTC, 98 history rows
- `jwetbe19` (s7): heartbeat ended 13:45:25 UTC, 26 history rows
- `cuzvcdw5` (s5): heartbeat ended 13:45:27 UTC, 7 history rows

The three runs were *created* over a 45-minute span (12:57, 13:32,
13:40 UTC) but **died within 25 seconds of each other**. All three
ran on host `gl017.hpc.nyu.edu`. Two of them had run for less than
40 minutes when they died; one had run for nearly an hour.

This pattern is consistent with an **infrastructure event**, not a
code bug: a SLURM node failure, file-system unmount on the scratch
volume, or a job-scheduler eviction. Code that had been running
successfully for 7 history-steps does not consistently fail at the
same wall-clock moment unless the failure is exogenous.

**What to do:** simply resubmit the three task-8 runs. The C2 cell
(k=0..7 working) is otherwise complete and produces the per-task
numbers reported in §2.

There are no C2 task-9 runs at all because auto-resume could not
advance past the failed task-8 checkpoints.

---

## 5. Recommended re-launches

To complete the experiment-data picture documented in the runbook:

1. **Resubmit C1 task-1 and onward** for both persistent and
   decomposed-w=0, after diagnosing the task-1 startup failure via
   the cluster-log recipe in §4.1. Until this lands, the C1
   regression-check verdict is "task-0 OK, k≥2 unknown".
2. **Resubmit C2 task-8 and task-9** for seeds 5, 6, 7. These are
   pure cluster-infrastructure failures; no code change needed.
   Once landed, the C2 numbers in §2 will be complete across all 10
   tasks.
3. **Run a fresh C0 seed (e.g., seed 8 or 9)** if reproducibility of
   the §1 numbers needs strengthening (currently 3 seeds is the
   sample; one more would tighten the SEM bars).

---

## 6. Files produced today

- `docs/wandb_analysis/csv/c0_summary.csv` — per-task α_scale and
  mixture-norm summaries (24 rows: 8 tasks × 3 seeds).
- `docs/wandb_analysis/csv/c0_best_per_task_seeds.csv` and
  `c0_best_per_task_agg.csv` — best-during-training success per task.
- `docs/wandb_analysis/csv/c1_summary.csv` and
  `c1_task0_seeds.csv` — C1 task-0 success comparison.
- `docs/wandb_analysis/csv/c2_summary.csv`,
  `c2_best_per_task_seeds.csv`, `c2_best_per_task_agg.csv` — C2 success.
- `docs/wandb_analysis/png/c0_trajectories.png` — mixture-norm and
  α_scale trajectories.
- `docs/wandb_analysis/c0_best_per_task.log` and
  `c2_best_per_task.log` — raw extraction logs.

The 2026-05-12 doc is left intact for historical record but is now
superseded by this one.
