# Runbook: what to launch next (decomposed-critic + CKA-failure diagnostics)

Date: 2026-05-08

Single source of truth for the cluster experiments that consume the
code work shipped in this session (N1–N4b, N2b, D1–D6). Each cell
here maps directly to a row in
`2026-05-08_implementation_tracking.md`.

This runbook assumes:

- branch `section3_done` is checked out at `/scratch/yd2247/sgcrl`
  (pull the latest before submitting)
- `conda activate contrastive_rl` is the env
- W&B is enabled by default

---

## How experiments are configured

All four submit scripts read **per-cell overrides from
`experiment_configs.py`** and forward them to
`run_continual_contrastive.py` as absl flags. There is no need to edit
dataclass defaults — every relevant knob is now a CLI flag.

The launchers:

| Script | HPC | Mode | Purpose |
|---|---|---|---|
| `draft_3.sh` | NYUAD Jubail | single-job | one-off cell, env vars on the sbatch line |
| `draft_4.sh` | NYUAD Jubail | array, 2/GPU | sweep multiple cells from `experiment_configs.py` |
| `DRAFT.sh` | NYU Torch | array, 3/GPU | sweep multiple cells from `experiment_configs.py` |
| `submit_continual_torch.sh` | NYU Torch | single-job | one-off cell, env vars on the sbatch line |

The flags exposed for the new work (all default to the dataclass /
absl defaults so omitting them preserves prior behaviour):

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `dyn_aux_weight` | float | `1.0` | `mu` on `L_dyn` (decomposed critic only). `0.0` for the regression-check cell. |
| `phi_task_width` | int | `256` | width of the per-task additive encoder |
| `phi_task_depth` | int | `2` | depth of the per-task additive encoder |
| `log_pool_cosine` | bool | `true` | per-task pool cosine matrices for D1–D4 |
| `log_mixture_norm` | bool | `false` | per-step `||sum_j a_j v_j|| / ||v_k||` for D5 |
| `log_probe_data` | bool | `false` | per-task `(obs, action)` dump for D6 |

### Two ways to set them

1. **Via env vars on the sbatch line** (single-job scripts):

   ```bash
   ACTOR_MODE=reset CRITIC_MODE=decomposed SEED=5 \
   DYN_AUX_WEIGHT=0.0 LOG_PROBE_DATA=true \
     sbatch submit_continual_torch.sh
   ```

2. **Via `CELLS` in `experiment_configs.py`** (array scripts; the
   recommended pattern for sweeps):

   ```python
   # experiment_configs.py
   CELLS = [
       {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 5,
        'dyn_aux_weight': 0.0, 'log_probe_data': True},
       {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 6,
        'dyn_aux_weight': 0.0, 'log_probe_data': True},
       ...
   ]
   ```

   Then submit:

   ```bash
   N=$(python experiment_configs.py --total)
   sbatch --array=0-$((N - 1)) draft_4.sh   # NYUAD
   sbatch --array=0-$((N - 1)) DRAFT.sh     # NYU Torch
   ```

   With `TASKS_PER_GPU > 1` (default 2 for `draft_4.sh`, 3 for
   `DRAFT.sh`), the array size is
   `ceil(N / TASKS_PER_GPU) - 1`. Use
   `python experiment_configs.py --list` to inspect the resolved
   schedule before submitting.

`experiment_configs.py` already validates that every CELLS entry
declares the required keys (`actor_mode`, `critic_mode`, `seed`); a
missing key fails fast at submission time with a clear error.

---

## Cell-by-cell launch

Cells are listed in dependency order. Each has its `experiment_configs.py`
CELLS entries (drop into the file, push, pull on the cluster, submit
the array), the W&B / file output paths, and the pass criterion.

### C0 — D7: CKA-failure diagnostic on the existing CKA cell

**Implements:** D7. **Gates:** none. **Runs in parallel with C1.**

This is the negative-result figure. Zero new algorithm code; uses
D1–D5 logging only.

```python
# experiment_configs.py — replace CELLS = []
CELLS = [
    {'actor_mode': 'cka', 'critic_mode': 'cka', 'seed': 5,
     'log_pool_cosine': True, 'log_mixture_norm': True},
    {'actor_mode': 'cka', 'critic_mode': 'cka', 'seed': 6,
     'log_pool_cosine': True, 'log_mixture_norm': True},
    {'actor_mode': 'cka', 'critic_mode': 'cka', 'seed': 7,
     'log_pool_cosine': True, 'log_mixture_norm': True},
]
# Also: ACTOR_MODES = []; CRITIC_MODES = []; SEEDS = [] to disable the
# Cartesian grid for this batch.
```

Submit:

```bash
cd /scratch/yd2247/sgcrl
git pull origin section3_done

N=$(python experiment_configs.py --total)        # 3
LAST=$(( (N + 2 - 1) / 2 - 1 ))                  # ceil(N/2) - 1 = 1 for draft_4
sbatch --array=0-$LAST draft_4.sh                 # NYUAD: 2 tasks per GPU
# or
LAST_T=$(( (N + 3 - 1) / 3 - 1 ))                # ceil(N/3) - 1 = 0 for DRAFT
sbatch --array=0-$LAST_T DRAFT.sh                 # NYU Torch: 3 tasks per GPU
```

**Result location:**

- W&B project `continual_gcrl_paper`, runs `task{k}_<env>_s{seed}`,
  keys `pool_cosine_actor/{n_active, mean_offdiag, max_offdiag,
  min_offdiag}`, `pool_cosine_critic/...`, and per-step
  `cka/actor_mixture_norm` / `cka/critic_mixture_norm`.
- Per-task `.npy` matrices in
  `/scratch/yd2247/sgcrl/logs/continual_checkpoints/actor_cka_critic_cka_tid_False_heads_True/seed_<S>/pool_cosine_{actor,critic}_task{k}.npy`.

**Pass criterion (REVISED 2026-05-12 after one run; see
`docs/2026-05-12_cka_failure_results.md`):**

- task 1: `n_active=0`, alpha metrics flat.
- task 2: `n_active=1`, off-diagonal stats are NaN.
- task 3+: `n_active >= 2`. **`mixture_norm < 0.1`** sustained
  through training (this is the core failure signature).

The original criterion also required `pool_cos_mean_offdiag > 0.9`,
on the conjecture that pool vectors collapse to one direction. The
first seed of C0 falsified that piece: actor `mean_offdiag` ~ 0.17
(stable across tasks 3-9), critic ~ 0. Pool vectors are diverse; the
failure mode is in the **gradient flow**, not the pool geometry.
Updated paper narrative is in
`docs/2026-05-12_cka_failure_results.md`.

One additional seed (seed 6 or 7) is needed to confirm the cosine
numbers reproduce. If they do, the negative-result figure shows
(i) `mixture_norm` decaying from non-trivial values to ~0 within
each task, and (ii) `mean_offdiag` staying well below 0.9 — the
two together being the quantitative signature of "mixture lives in
the gradient null space of contrastive losses".

**Paper mapping:** negative-result figure in the analysis section.

---

### C1 — N5: decomposed regression check (`dyn_aux_weight=0`)

**Implements:** N5. **Gates:** none experimental (depends only on
N1–N4b, all shipped).

Verify the decomposed code path with the dynamics auxiliary disabled
reduces to a contrastive-only baseline within seed noise. If this
fails, the issue is in the decomposed plumbing, not in the algorithm.

```python
# experiment_configs.py — CELLS for C1 (decomposed at w=0 + persistent baseline)
CELLS = [
    # decomposed at dyn_aux_weight=0 (regression check)
    {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 5,
     'dyn_aux_weight': 0.0, 'log_probe_data': True},
    {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 6,
     'dyn_aux_weight': 0.0, 'log_probe_data': True},
    {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 7,
     'dyn_aux_weight': 0.0, 'log_probe_data': True},
    # persistent baseline at the same seeds (for the comparison)
    {'actor_mode': 'reset', 'critic_mode': 'persistent', 'seed': 5,
     'log_probe_data': True},
    {'actor_mode': 'reset', 'critic_mode': 'persistent', 'seed': 6,
     'log_probe_data': True},
    {'actor_mode': 'reset', 'critic_mode': 'persistent', 'seed': 7,
     'log_probe_data': True},
]
ACTOR_MODES = []; CRITIC_MODES = []; SEEDS = []   # disable Cartesian grid
```

Submit (NYUAD):

```bash
cd /scratch/yd2247/sgcrl
git pull origin section3_done

N=$(python experiment_configs.py --total)               # 6
LAST=$(( (N + 2 - 1) / 2 - 1 ))                         # 2
sbatch --array=0-$LAST --time=72:00:00 draft_4.sh
```

Submit (NYU Torch):

```bash
cd /scratch/yd2247/sgcrl
git pull origin section3_done

N=$(python experiment_configs.py --total)
LAST_T=$(( (N + 3 - 1) / 3 - 1 ))                       # 1
sbatch --array=0-$LAST_T --time=72:00:00 DRAFT.sh
```

**Pass criterion** (gates C2):

- For task `k >= 2`: decomposed-w0 mean success >= persistent mean
  success - 0.05 across 3 seeds.
- `decomp/L_dyn` is logged but does not affect gradients (sanity that
  the dyn head still runs in the no-grad path).

If pass: tag `git tag n5_passed_<DATE>` and proceed to C2.

If fail: the decomposed plumbing has a regression. First-pass
diagnosis — compare the following metrics between decomposed-w0 and
persistent at the same seed:

- `actor_loss`, `critic_loss`, `entropy_mean`, `categorical_accuracy`.
  These should be statistically indistinguishable.
- The first divergence is most likely in the actor goal-rolling block
  (verified in N2b) or the InfoNCE form branch under `use_cpc`.

**Paper mapping:** appears as the `dyn_aux_weight=0` cell in the
ablation table; demonstrates structural neutrality.

---

### C2 — N6: single-cell sanity (`dyn_aux_weight=1.0`)

**Implements:** N6. **Gates:** C1 passes. **Runs in parallel with C0.**

Plan §8 first-cell config: `actor_mode='reset',
critic_mode='decomposed', dyn_aux_weight=1.0`, K_max=5,
ResidualMLP 1024×4, 8M steps/task, three seeds.

```python
# experiment_configs.py — CELLS for C2
CELLS = [
    {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 5,
     'dyn_aux_weight': 1.0, 'log_probe_data': True},
    {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 6,
     'dyn_aux_weight': 1.0, 'log_probe_data': True},
    {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 7,
     'dyn_aux_weight': 1.0, 'log_probe_data': True},
]
ACTOR_MODES = []; CRITIC_MODES = []; SEEDS = []
```

Submit (use the same array commands as C1; `N=3`).

Set the run config to match plan §8 via env vars on the sbatch line
or in the script's defaults:

```bash
K_MAX=5 NETWORK_WIDTH=1024 sbatch --array=0-1 draft_4.sh
```

(`K_MAX` and `NETWORK_WIDTH` are already env-var-overridable in
`draft_3.sh` / `draft_4.sh` / `DRAFT.sh` /
`submit_continual_torch.sh`; defaults are 10 / 256 respectively, so
override.)

**Pass criteria (plan §8):**

1. **`L_dyn` decreases monotonically across tasks** (`decomp/L_dyn`
   in W&B). Sawyer dynamics are consistent across the 10 tasks at
   the masked indices, so the body should learn them once.
2. **Average task-`k` success rate at least as good as the
   persistent baseline for `k >= 2`** on all three seeds. Compare
   against the persistent runs from C1.
3. **Linear-probe task-classifier accuracy on `b_shared` output
   is below `1/N + 5%`** (i.e., < 15% for 10 tasks). Run the probe
   once cluster jobs finish (see "After C2" below).

If all three pass: tag `n6_passed_<DATE>`, proceed to C3.

If criterion 3 fails: `b_shared` is absorbing task identity → spike
to N8 (mixed-task dynamics buffer) before launching the grid.

**After C2 (D6 linear probe; runs on the cluster login node or any
CPU box with the repo):**

```bash
cd /scratch/yd2247/sgcrl

for SEED in 5 6 7; do
  python eval_linear_probe.py \
    --checkpoint_dir=/scratch/yd2247/sgcrl/logs/continual_checkpoints \
    --seed=$SEED \
    --num_tasks=10 \
    --critic_mode=decomposed \
    --actor_mode=reset \
    --use_task_id=false \
    --adapt_heads_only=true
done
```

For the comparison baseline in the appendix, also probe the persistent
runs from C1 at the same seeds:

```bash
for SEED in 5 6 7; do
  python eval_linear_probe.py \
    --checkpoint_dir=/scratch/yd2247/sgcrl/logs/continual_checkpoints \
    --seed=$SEED --num_tasks=10 \
    --critic_mode=persistent --actor_mode=reset \
    --use_task_id=false --adapt_heads_only=true
done
```

The script prints test accuracy, per-task accuracy, and a
row-normalised confusion matrix; the PASS / FAIL line at the bottom
references plan §3.4 thresholds.

**Paper mapping:**
- Criterion 1 (`L_dyn` decreases) → side panel on the dynamics-loss
  curve.
- Criterion 2 (success rate parity) → main result table, decomposed
  column.
- Criterion 3 (linear probe) → appendix table, row "decomposed
  `b_shared`" vs "persistent critic body".

---

### C3 — N7: full ablation grid (5 cells × 5 seeds × 10 tasks)

**Implements:** N7. **Gates:** C2 passes all three criteria.

Plan §8 ablation grid:

| ID | dyn_aux_weight | actor_mode | critic_mode  | comment             |
|----|----------------|------------|--------------|---------------------|
| G1 | -              | reset      | persistent   | existing baseline   |
| G2 | 0.0            | reset      | decomposed   | dyn-aux off         |
| G3 | 0.1            | reset      | decomposed   | dyn-aux weak        |
| G4 | 1.0            | reset      | decomposed   | dyn-aux full        |
| G5 | 1.0            | reset      | reset        | dyn-aux + reset crit |

Five seeds: 5, 6, 7, 8, 9. **G5 is held** until a small plumbing
change adds an "decomposed body + reset carry" config (tracked as
N7b).

For G1, G2, G3, G4 with 5 seeds each = 20 runs. C1 already produces
the G2 column (3 seeds) and the G1 column (3 seeds); add seeds 8 and
9 for those, plus full 5 seeds for G3 and G4. Net new: 4 + 5 + 5 = 14
runs (or 20 if you redo C1).

```python
# experiment_configs.py — CELLS for C3
import itertools as _it
SEEDS_C3 = [5, 6, 7, 8, 9]

_g1 = [{'actor_mode': 'reset', 'critic_mode': 'persistent', 'seed': s}
       for s in SEEDS_C3]
_g2 = [{'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': s,
        'dyn_aux_weight': 0.0}
       for s in SEEDS_C3]
_g3 = [{'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': s,
        'dyn_aux_weight': 0.1}
       for s in SEEDS_C3]
_g4 = [{'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': s,
        'dyn_aux_weight': 1.0}
       for s in SEEDS_C3]

CELLS = _g1 + _g2 + _g3 + _g4   # 20 cells total
ACTOR_MODES = []; CRITIC_MODES = []; SEEDS = []
```

Submit (NYUAD): `N=20`, `LAST = ceil(20/2) - 1 = 9`. NYU Torch:
`LAST = ceil(20/3) - 1 = 6`. Roughly ~5h/run on a Torch H100 → ~100
GPU-hours.

```bash
cd /scratch/yd2247/sgcrl && git pull origin section3_done
N=$(python experiment_configs.py --total)
echo "Total: $N"

# NYUAD
LAST=$(( (N + 2 - 1) / 2 - 1 ))
sbatch --array=0-$LAST --time=72:00:00 draft_4.sh

# NYU Torch
LAST_T=$(( (N + 3 - 1) / 3 - 1 ))
sbatch --array=0-$LAST_T --time=72:00:00 DRAFT.sh
```

**Pass criterion:**

- G2 ≈ G1 within seed noise (regression check, mirrors C1 result).
- G3, G4 ≥ G1 on `k >= 2` average success.
- G4 ≥ G3 on later tasks (dyn-aux helps as `mu` grows).

**Paper mapping:** main result table (5 columns × 10 tasks ×
last-task success rate, mean ± stderr over 5 seeds).

---

### C4 — Final aggregation and paper-figure assembly

**Implements:** the negative-result figure (from C0) and the main
ablation table (from C3) for the paper. No new code; only analysis
notebooks consuming the W&B run data.

Suggested structure:

- Negative-result figure (CKA failure):
  - subplot 1: `pool_cosine_actor/mean_offdiag` over tasks
    (3 seeds, mean ± stderr).
  - subplot 2: `cka/actor_mixture_norm` over training (one curve per
    task boundary).
  - data: from C0 W&B runs.
- Main table:
  - rows: G1, G2, G3, G4 (G5 if N7b ships).
  - columns: avg success rate, forgetting metric, linear-probe
    accuracy on `b_shared`.
  - data: C2 + C3 + post-N6 linear-probe runs.
- Appendix table:
  - rows: persistent (existing critic), decomposed (b_shared).
  - column: linear-probe test accuracy.

---

## Quick reference: dependency / parallelism

| Cell | Runs in parallel with | Blocks on |
|------|------------------------|-----------|
| C0 (D7 diagnostic) | C1 | nothing |
| C1 (N5 regression) | C0 | nothing |
| C2 (N6 sanity)     | C0 | C1 pass |
| C3 (N7 grid)       | nothing | C2 pass |
| C4 (paper figs)    | nothing | C3 finish |

Worst-case full path C0/C1 → C2 → C3 → C4 takes ~4–5 days of GPU
time. C0 + C1 can run on day 1; C2 on day 2 if C1 passes; C3 on
days 3–4; C4 is analysis only.

---

## Common pitfalls (from prior runs in this codebase)

- **`USE_TASK_ID` default is `false` in absl FLAGS** but `true` in
  the older `submit_continual_torch.sh` defaults. Project convention
  is `false`. If you submit a single-job script without explicitly
  setting `USE_TASK_ID=false`, you may be silently using one-hot
  task ids; this changes obs layout and breaks comparison with
  earlier non-task-id runs.
- **`NETWORK_WIDTH=256` is the dataclass default** but the project
  run config is `1024`. Always set `NETWORK_WIDTH=1024` (and
  `CRITIC_DEPTH=4`, `ACTOR_DEPTH=4`) for the scaling-CRL reference.
- **`K_MAX=10` is the dataclass default** but plan §8 says 5 for the
  decomposed cells. Set `K_MAX=5` for C2 and C3.
- **`actor_mode='reset'` is required for `critic_mode='decomposed'`**
  (the runner enforces this with an early FLAG-side guard).
- **`log_pool_cosine` is on by default** (plan §3.1 logging is
  cheap; gives the negative-result figure for free on every CKA
  cell).
- **`log_mixture_norm` and `log_probe_data` are off by default**;
  flip them on per cell via the `CELLS` entry in
  `experiment_configs.py` or the env-var pipeline.
- **Auto-resume reads `decomposed_*` keys from the checkpoint** when
  `--critic_mode=decomposed` is passed (N4b). Cells from different
  configurations live under different `_ckpt_path` config keys, so
  there is no accidental cross-pollination.
- **`log_probe_data=true` adds one `next(iterator)` and a ~10 KB
  npz dump per task** — negligible relative to the 8M-step training.
- **`log_mixture_norm=true` adds one extra norm per inner step** in
  the CKA path; ~1% step-time overhead.

---

## Files referenced

- `2026-05-08_implementation_tracking.md` (status table)
- `2026-05-08_plan_proposal1_dyn_aux.md` (plan, §3.1, §3.2, §3.4, §8, §9, §10)
- `2026-05-08_decomposed_critic_implementation.md` (N1–N4 design)
- `2026-05-08_decomposed_critic_verification.md` (N2b SGCRL audit)
- `2026-05-08_decomposed_critic_n4b.md` (runner-glue log)
- `2026-05-08_d5_mixture_norm.md` (mixture_norm helper + wiring)
- `2026-05-08_d6_linear_probe.md` (probe data dump + eval script)
- `experiment_configs.py` (CELLS list + Cartesian grid)
- `draft_3.sh`, `draft_4.sh`, `DRAFT.sh`, `submit_continual_torch.sh`
  (the four launchers)

When the runbook gets stale (cells move, criteria change, new cells
added), update `implementation_tracking.md` first, then propagate
here.
