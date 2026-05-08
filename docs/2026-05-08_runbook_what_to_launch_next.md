# Runbook: what to launch next (decomposed-critic + CKA-failure diagnostics)

Date: 2026-05-08

Single source of truth for the cluster experiments that consume the
code work shipped in this session (N1–N4b, N2b, D1–D6). Each cell here
maps directly to a row in `2026-05-08_implementation_tracking.md`.

This runbook assumes:

- branch `section3_done` is checked out at `/scratch/yd2247/sgcrl`
  (pull the latest before submitting; commits up to `c9ed303` are
  required)
- `submit_continual_torch.sh` is the launcher (NYU Torch HPC)
- `conda activate contrastive_rl` is the env
- W&B is enabled by default (`USE_WANDB=true` in the submit script)

The cells below are listed in dependency order. Each cell has:

- the row in `implementation_tracking.md` it satisfies
- gating: which earlier cells must pass before launching it
- exact submit command(s)
- where to find the result and how to read it
- pass / fail criterion drawn from `2026-05-08_plan_proposal1_dyn_aux.md`
- which paper figure / table the result populates

---

## Note on flags vs dataclass defaults

The five new diagnostic / decomposed-critic knobs live in
`ContinualConfig` (not absl FLAGS), so they cannot currently be set
from the submit script's environment variables:

| Field | Default | Used by | Plan ref |
|---|---|---|---|
| `dyn_aux_weight` | `1.0` | decomposed critic | §6 |
| `phi_task_width` | `256` | decomposed critic | §6 |
| `phi_task_depth` | `2` | decomposed critic | §6 |
| `log_pool_cosine` | `True` | CKA diagnostics (D1–D4) | §3.1 |
| `log_mixture_norm` | `False` | CKA diagnostics (D5) | §3.2 |
| `log_probe_data` | `False` | linear-probe (D6) | §3.4 |

Two ways to set these per cell:

- **A. Edit the dataclass default** in `contrastive/continual_config.py`,
  commit on `section3_done`, push, pull on the cluster, submit. This
  matches the existing pattern (commit `c700e55` flipped
  `log_pool_cosine` to `True` this way). One short commit per change;
  cells are reproducible from the SHA.
- **B. Add absl flags + thread them into the `ContinualConfig(...)`
  constructor in `run_continual_contrastive.py`, then expose them in
  `submit_continual_torch.sh`. Cleaner long-term, requires a one-time
  code change.

The cells below assume **option A** (per-cell commit) because that is
what is in place today; switch to option B once the first decomposed
cell has produced a clean N5 result and the pattern is justified.

---

## C0 — D7: CKA-failure diagnostic on the existing CKA cell (workstream A)

**Implements:** D7 (table row).

**Gates:** D1–D6 are all shipped. No experimental dependency.

**Why first:** This is the negative-result figure. It uses zero new
algorithm code (just the diagnostic logging shipped in D1–D5). Run it
in parallel with C1; data lands without blocking the decomposed work.

**One-time edit on `section3_done`:** flip `log_mixture_norm` to
`True` so the per-step ratio gets logged for this run only.
`log_pool_cosine` is already `True`. `log_probe_data` is not needed
for this cell (no `b_shared` to probe).

```python
# contrastive/continual_config.py
log_mixture_norm: bool = True   # was False
```

Commit, push, pull on Torch.

**Submit (3 seeds):**

```bash
cd /scratch/yd2247/sgcrl

for SEED in 5 6 7; do
  ACTOR_MODE=cka \
  CRITIC_MODE=cka \
  USE_TASK_ID=false \
  K_MAX=5 \
  NETWORK_WIDTH=1024 \
  CRITIC_DEPTH=4 \
  ACTOR_DEPTH=4 \
  STEPS_PER_TASK=8000000 \
  BASE_STEPS=8000000 \
  SEED=$SEED \
  sbatch --job-name=cka_diag_s$SEED submit_continual_torch.sh
done
```

**Where the result lands:**

- W&B project `continual_gcrl_paper`, group `measure_similarityof_vk`,
  task-by-task runs `task{k}_sawyer_*_s{seed}` with keys
  `pool_cosine_actor/{n_active, mean_offdiag, max_offdiag,
  min_offdiag}`, `pool_cosine_critic/...`, and per-step
  `cka/actor_mixture_norm` / `cka/critic_mixture_norm`.
- Per-task `.npy` matrices in
  `/scratch/yd2247/sgcrl/logs/continual_checkpoints/actor_cka_critic_cka_tid_False_heads_True/seed_<S>/pool_cosine_{actor,critic}_task{k}.npy`.

**Pass criterion (plan §3.1, §3.2):**

- task 1: `n_active=0`, alpha metrics flat (single-slot degeneracy).
- task 2: `n_active=1`, off-diagonal stats are NaN.
- task 3+: `n_active >= 2`. **`pool_cos_mean_offdiag > 0.9`** AND
  **`mixture_norm < 0.1`** sustained through training.

If both hold, the CKA-failure narrative for the paper is empirically
supported. If `mean_offdiag < 0.5` or `mixture_norm > 0.5`, the audit
hypothesis is wrong and the paper needs a different explanation.

**Paper mapping:** negative-result figure in the analysis section
(replaces "CKA failed qualitatively" with quantitative data).

---

## C1 — N5: decomposed-critic regression check (`dyn_aux_weight=0`)

**Implements:** N5 (table row).

**Gates:** N1, N2, N2b, N3 (partial), N4, N4b all shipped. No
experimental dependency on C0.

**Purpose:** Verify the decomposed code path with the dynamics
auxiliary disabled reduces to a contrastive-only baseline within seed
noise. If this fails, the issue is in the decomposed plumbing (N4b
glue or N4 sibling learner), not in the algorithmic idea.

**One-time edit on `section3_done`:** flip `dyn_aux_weight` to `0.0`
and turn on `log_probe_data` so we can run D6 afterwards.

```python
# contrastive/continual_config.py
dyn_aux_weight: float = 0.0     # was 1.0; disables L_dyn for the regression cell
log_probe_data: bool = True     # was False; needed for the post-hoc D6 probe
```

Commit, push, pull. Tag this commit so the regression cell is
reproducible (`git tag n5_regression`).

**Submit (3 seeds):**

```bash
cd /scratch/yd2247/sgcrl

for SEED in 5 6 7; do
  ACTOR_MODE=reset \
  CRITIC_MODE=decomposed \
  USE_TASK_ID=false \
  K_MAX=5 \
  NETWORK_WIDTH=1024 \
  CRITIC_DEPTH=4 \
  ACTOR_DEPTH=4 \
  STEPS_PER_TASK=8000000 \
  BASE_STEPS=8000000 \
  SEED=$SEED \
  sbatch --job-name=decomp_w0_s$SEED submit_continual_torch.sh
done

# Submit the persistent baseline at the same seeds (existing 9-cell
# row — no code changes, only the decomp config flip is reverted).
# Use a SECOND clone or a worktree so the dataclass defaults don't
# clash, OR run the persistent baseline first then the decomposed.
# Easiest: revert the dataclass edits (or check out main from before
# c700e55) and re-submit the persistent baseline.
for SEED in 5 6 7; do
  ACTOR_MODE=reset \
  CRITIC_MODE=persistent \
  USE_TASK_ID=false \
  K_MAX=5 \
  NETWORK_WIDTH=1024 \
  CRITIC_DEPTH=4 \
  ACTOR_DEPTH=4 \
  STEPS_PER_TASK=8000000 \
  BASE_STEPS=8000000 \
  SEED=$SEED \
  sbatch --job-name=persistent_s$SEED submit_continual_torch.sh
done
```

**Where the result lands:**

- W&B `eval/mean_success` per task and `eval/<env_name>` for each
  seen task (forgetting curve).
- Final per-task accuracy printed at end of training under the
  `[intra-eval]` log prefix.

**Pass criterion (this is the gate for N6):**

- For task `k >= 2`: decomposed `dyn_aux_weight=0` mean success >=
  persistent baseline mean success - 0.05 across 3 seeds.
- `decomp/L_dyn` is logged but not used for grad (sanity check that
  the dyn head still runs).

If pass: tag `git tag n5_passed`, proceed to C2.

If fail: the decomposed plumbing has a regression. Cross-check
`actor_loss`, `critic_loss`, `entropy_mean`, and `categorical_accuracy`
between decomposed-w0 and persistent. The most likely culprits are
the actor goal-rolling (N2b verification fixes) and the InfoNCE form
branch (`use_cpc`).

**Paper mapping:** appears as the `dyn_aux_weight=0` cell in the
ablation table. Demonstrates that the decomposed structure is
neutral when the auxiliary is off.

---

## C2 — N6: single-cell sanity (`dyn_aux_weight=1.0`)

**Implements:** N6 (table row).

**Gates:** C1 (N5) passes. C0 can be running in parallel.

**One-time edit on `section3_done`:** flip `dyn_aux_weight` back to
`1.0`. Keep `log_probe_data=True` so the linear probe runs after.

```python
# contrastive/continual_config.py
dyn_aux_weight: float = 1.0     # back to default; enables L_dyn
log_probe_data: bool = True     # keep True for D6 probe
```

Commit, push, pull, tag (`git tag n6_smoke`).

**Submit (3 seeds, plan §8):**

```bash
cd /scratch/yd2247/sgcrl

for SEED in 5 6 7; do
  ACTOR_MODE=reset \
  CRITIC_MODE=decomposed \
  USE_TASK_ID=false \
  K_MAX=5 \
  NETWORK_WIDTH=1024 \
  CRITIC_DEPTH=4 \
  ACTOR_DEPTH=4 \
  STEPS_PER_TASK=8000000 \
  BASE_STEPS=8000000 \
  SEED=$SEED \
  sbatch --job-name=decomp_w1_s$SEED submit_continual_torch.sh
done
```

**Pass criteria (plan §8):**

1. **`L_dyn` decreases monotonically across tasks** (logged as
   `decomp/L_dyn`). Sawyer dynamics are consistent across the 10
   tasks at the masked indices, so the body should learn them once.
2. **Average task-`k` success rate at least as good as the
   persistent baseline for `k >= 2`** on all three seeds. Compare to
   the persistent runs from C1.
3. **Linear-probe task-classifier accuracy on `b_shared` output is
   below `1/N + 5%`** (i.e., for 10 tasks, below ~15%). Run the probe
   once the cluster jobs finish (see "After C2" below).

If all three pass: tag `git tag n6_passed`, proceed to C3 (full
ablation grid). If criterion 3 fails (probe accuracy high), the body
is absorbing task identity → spike to N8 (mixed-task dynamics buffer)
before the grid.

**After C2 (D6 linear probe; runs locally on the cluster login node
or any CPU box with the repo):**

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

The script prints the test accuracy, per-task accuracy, and a
row-normalised confusion matrix. The PASS / FAIL line at the bottom
references plan §3.4 thresholds. For the "comparison baseline" claim
in the paper, also probe the existing persistent column for the same
3 seeds:

```bash
for SEED in 5 6 7; do
  python eval_linear_probe.py \
    --checkpoint_dir=/scratch/yd2247/sgcrl/logs/continual_checkpoints \
    --seed=$SEED \
    --num_tasks=10 \
    --critic_mode=persistent \
    --actor_mode=reset \
    --use_task_id=false \
    --adapt_heads_only=true
done
```

This probes `q_network`'s sa-encoder hidden output (the natural
analog of `b_shared`) on the persistent run. The paper claim is
"`b_shared` probe accuracy is lower than the existing critic body's
probe accuracy"; this gives the comparison data.

**Paper mapping:**
- Criterion 1 (`L_dyn` decreases) → side panel on the dynamics-loss
  curve.
- Criterion 2 (success rate parity) → main result table, decomposed
  column.
- Criterion 3 (linear probe) → appendix table, row "decomposed
  `b_shared`" (compared to "persistent critic body").

---

## C3 — N7: full ablation grid (5 cells × 5 seeds × 10 tasks)

**Implements:** N7 (table row).

**Gates:** C2 (N6) passes all three pass criteria.

**Five cells (plan §8):**

| ID | dyn_aux_weight | actor_mode | critic_mode | comment             |
|----|----------------|------------|-------------|---------------------|
| G1 | -              | reset      | persistent  | existing baseline   |
| G2 | 0.0            | reset      | decomposed  | dyn-aux off         |
| G3 | 0.1            | reset      | decomposed  | dyn-aux weak        |
| G4 | 1.0            | reset      | decomposed  | dyn-aux full        |
| G5 | 1.0            | reset      | reset       | dyn-aux + reset crit |

Five seeds: 5, 6, 7, 8, 9. Total 25 runs. At 8M steps/task and
~5h/run on a Torch H100, plan for ~125 GPU-hours.

**Edits per cell:**

- G1: revert dataclass edits, then submit `CRITIC_MODE=persistent`.
- G2: `dyn_aux_weight=0.0`, `CRITIC_MODE=decomposed`.
- G3: `dyn_aux_weight=0.1`, `CRITIC_MODE=decomposed`.
- G4: `dyn_aux_weight=1.0`, `CRITIC_MODE=decomposed` (this is the C2
  config; if you keep checkpoints, the 3 seeds from C2 already
  populate this cell — submit only the missing 2 seeds).
- G5: `dyn_aux_weight=1.0`, `CRITIC_MODE=reset` (decomposed body but
  the per-task delta is fully reset rather than carried; this is
  effectively a reset-critic cell with the dynamics auxiliary).

For G5, note that `CRITIC_MODE=reset` is **not** the same as
`CRITIC_MODE=decomposed` with a reset of the carry — `CRITIC_MODE`
controls which learner gets instantiated. Plan §8 row G5 needs a
small extra: a flag to opt the decomposed learner into resetting all
four critic groups at task boundaries. **This is not currently
implemented.** Treat G5 as held until a small plumbing change adds an
`actor_mode='reset' + critic_mode='decomposed' + reset_decomposed=True`
config; document as N7b in the tracker.

**Submit (4 cells × 5 seeds = 20 runs; G5 deferred):**

```bash
cd /scratch/yd2247/sgcrl

# G1 baseline (only run seeds not already in C1)
for SEED in 8 9; do
  ACTOR_MODE=reset CRITIC_MODE=persistent USE_TASK_ID=false \
  K_MAX=5 NETWORK_WIDTH=1024 STEPS_PER_TASK=8000000 BASE_STEPS=8000000 \
  SEED=$SEED \
  sbatch --job-name=g1_pers_s$SEED submit_continual_torch.sh
done

# G2 / G3 / G4: each requires its own dataclass edit + commit + push
# + cluster pull. Iterate per cell:
#
#   1. Edit dyn_aux_weight in continual_config.py
#   2. git commit / push / pull on cluster
#   3. for SEED in 5 6 7 8 9; do sbatch ...; done
#   4. Wait for first task to land, then move to the next cell
#
# (Tag each commit: g2_decomp_w0, g3_decomp_w01, g4_decomp_w1.)
```

**Where the result lands:**

- W&B `eval/mean_success` per cell-seed-task triple.
- Final per-task forgetting curves and final-task summary in
  `[intra-eval] Mean success`.

**Pass criterion:**

- G2 ≈ G1 within seed noise (regression check, mirrors C1 result).
- G3, G4 ≥ G1 on `k >= 2` average success.
- G4 ≥ G3 on later tasks (dyn-aux helps as `mu` grows).

**Paper mapping:** main result table (5 columns × 10 tasks ×
last-task success rate, mean ± stderr over 5 seeds).

---

## C4 — Final aggregation and paper-figure assembly

**Implements:** the negative-result figure (from C0) and the main
ablation table (from C3) for the paper. No new code; only analysis
notebooks consuming the W&B run data.

Suggested structure:

- Negative-result figure (CKA failure):
  - subplot 1: `pool_cos_mean_offdiag` over tasks (3 seeds, mean
    line + shaded stderr).
  - subplot 2: `cka/actor_mixture_norm` over training (one curve per
    task boundary).
  - data: from the C0 W&B runs.
- Main table:
  - rows: G1, G2, G3, G4 (G5 if it's been added by then).
  - columns: avg success rate, forgetting metric, linear-probe
    accuracy on `b_shared`.
  - data: from C3 + C2 + the post-N6 linear-probe runs.
- Appendix table:
  - rows: persistent (existing critic), decomposed (b_shared).
  - column: linear-probe test accuracy.

---

## Quick reference: what's already running, what's gated

| Cell | Runs in parallel with | Blocks on |
|------|------------------------|-----------|
| C0 (D7 diagnostic) | C1 | nothing |
| C1 (N5 regression) | C0 | nothing |
| C2 (N6 sanity)     | C0 | C1 pass |
| C3 (N7 grid)       | nothing | C2 pass |
| C4 (paper figs)    | nothing | C3 finish |

Worst-case full path C0/C1 → C2 → C3 → C4 takes ~4-5 days of GPU
time. C0 + C1 can run on day 1; C2 on day 2 if C1 passes; C3 on
days 3-4; C4 is analysis only.

---

## Common pitfalls (from prior runs in this codebase)

- **`USE_TASK_ID` default is `true` in the submit script** but
  `false` in absl FLAGS. Always set `USE_TASK_ID=false` explicitly
  for new experiments per the project default; otherwise the obs
  layout differs and you can't compare against earlier non-task-id
  runs.
- **`NETWORK_WIDTH=256` is the dataclass default** but the project
  run config is `1024`. Always set `NETWORK_WIDTH=1024` (and
  `CRITIC_DEPTH=4`, `ACTOR_DEPTH=4`) to match the scaling-CRL
  reference and the existing 9-cell runs.
- **`K_MAX=10` is the dataclass default** but plan §8 says 5. Always
  set `K_MAX=5` for new experiments.
- **`actor_mode='reset'` is required for `critic_mode='decomposed'`**
  (the runner enforces this with an early FLAG-side guard). Do not
  pair decomposed with `actor_mode='cka'` or `actor_mode='persistent'`.
- **Auto-resume reads `decomposed_*` keys from the checkpoint** when
  `--critic_mode=decomposed` is passed (N4b). If you cherry-pick a
  partial run from a different cell, the auto-resume will fail
  cleanly with a clear error.
- **`log_probe_data=True` doubles the I/O slightly** at end-of-task
  (one `next(iterator)` and one ~10 KB npz dump). Negligible relative
  to the 8M-step training, but worth knowing if you sweep many cells.
- **`log_mixture_norm=True` adds one extra norm per inner step** in
  the CKA path. Roughly 1% step-time overhead; bit-identical when
  off (gated as a Python `if`).

---

## Files referenced

- `2026-05-08_implementation_tracking.md` (status table)
- `2026-05-08_plan_proposal1_dyn_aux.md` (plan, §3.1, §3.2, §3.4, §8, §9, §10)
- `2026-05-08_decomposed_critic_implementation.md` (N1–N4 design)
- `2026-05-08_decomposed_critic_verification.md` (N2b SGCRL-conventions audit)
- `2026-05-08_decomposed_critic_n4b.md` (runner-glue log)
- `2026-05-08_d5_mixture_norm.md` (mixture_norm helper + wiring)
- `2026-05-08_d6_linear_probe.md` (probe data dump + eval script)
- `submit_continual_torch.sh` (the launcher)

When the runbook gets stale (cells move, criteria change, new cells
added), update the table at the top of `implementation_tracking.md`
first, then propagate the change here.
