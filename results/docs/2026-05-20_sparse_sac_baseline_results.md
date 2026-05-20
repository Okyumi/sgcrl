# Sparse goal-conditioned SAC baselines (continual_sac project)

**Date.** 2026-05-20
**Source.** W&B project `d_konoki/continual_sac`, 144 finished runs.
**Author.** Pulled by paper-side tooling for the RLC 2026 workshop submission.

## Why these runs

The paper needs an explicit sparse-reward continual-RL baseline that does *not* use the contrastive critic. The sparse goal-conditioned SAC variants reported here are that baseline: SAC + HER, with a per-step `-1`/`0` penalty reward, run on the same ten-task Continual World V2 Sawyer curriculum and with the same per-task budget as the contrastive cells. They give a sparse-reward, non-contrastive reference point against which to claim that contrastive GCRL is a strong solver for sparse-reward continual RL.

The user confirmed that the `-1`/`0` penalty variant (`step_penalty_reward=True`) is the regime under which sparse SAC is most competitive on this benchmark, so this writeup only reports that variant. Two transfer modes are reported:

- **SAC reset / reset (R/R)** — actor and critic re-initialised at every task boundary.
- **SAC persistent / persistent (P/P)** — actor and critic carried forward.

## Run grouping and filtering

The `continual_sac` W&B project has no group names. Effective grouping is built from the config: `(task_id, actor_mode, critic_mode, step_penalty_reward)`. After filtering to `step_penalty_reward=True` and the two SAC variants above, the layout is one finished run per `(actor_mode, critic_mode, task_id, seed)` for `task_id ∈ {0,…,9}` and three seeds per task. Total: 60 runs.

Headline metric: per-task **best mean success**, defined as the maximum of `evaluator/success_rate` over training, averaged across seeds. This matches the headline metric used for the contrastive cells in `2026-05-14_results_overview.md` and in the RLC paper.

## Headline numbers

Curriculum averages across the ten tasks:

| Cell | Avg. best success | n seeds (per task) |
|---|---|---|
| Sparse SAC R/R, `-1`/`0` penalty | **0.417** | 3 |
| Sparse SAC P/P, `-1`/`0` penalty | **0.437** | 3 |
| Contrastive R/R (from-scratch)   | 0.832 | 5 |
| Contrastive P/P (persistent)     | 0.791 | 5 |
| Contrastive CKA on actor (C/R)   | 0.841 | 4–5 |
| Decomposed contrastive (ours)    | **0.873** | 3 |

The contrastive cells span `[0.79, 0.87]` on curriculum-average best success; both sparse SAC variants sit in `[0.42, 0.44]`. The gap is large enough to support the paper's claim that contrastive GCRL is a strong sparse-reward continual-RL solver irrespective of which contrastive transfer mode is used.

## Per-task numbers

Mean best success across 3 seeds per task (s.d. in parentheses).

| k | task | SAC R/R | SAC P/P |
|---|---|---|---|
| 0 | hammer            | 1.00 (0.00) | 1.00 (0.00) |
| 1 | push-wall         | 0.00 (0.00) | 0.13 (0.23) |
| 2 | faucet-close      | 0.70 (0.20) | 0.80 (0.10) |
| 3 | push-back         | 0.00 (0.00) | 0.00 (0.00) |
| 4 | stick-pull        | 1.00 (0.00) | 0.67 (0.58) |
| 5 | handle-press-side | 0.53 (0.06) | 0.50 (0.10) |
| 6 | push              | 0.07 (0.06) | 0.30 (0.35) |
| 7 | shelf-place       | 0.00 (0.00) | 0.00 (0.00) |
| 8 | window-close      | 0.80 (0.00) | 0.80 (0.00) |
| 9 | peg-unplug-side   | 0.07 (0.06) | 0.17 (0.06) |
| | **avg** | **0.417** | **0.437** |

## Observations

The SAC vs contrastive comparison is **not** uniformly in favour of contrastive. SAC under the `-1`/`0` penalty actually reaches **0.80 on window-close (k=8)** for both transfer modes, well above the contrastive baselines (0.37 for R/R and P/P, 0.40 for CKA) and above the decomposed cell (0.63). It also outperforms contrastive cells on `handle-press-side` (k=5): both SAC variants reach 0.50–0.53, while the contrastive baselines sit at 0.20–0.26 and the decomposed cell reaches 0.40.

SAC, on the other hand, completely fails on three tasks that all contrastive cells solve effortlessly: `push-wall` (k=1), `push-back` (k=3), and `shelf-place` (k=7), each at 0.00 success. It also collapses on `peg-unplug-side` (k=9) to 0.07–0.17, while every contrastive cell sits at 0.80–0.87.

A reasonable reading: the per-step penalty acts as dense reward shaping on tasks where the success geometry is well-aligned with stepwise distance reduction (window closing, handle pressing), and SAC handles those fine; on tasks where the geometry is multi-modal or requires precise late-stage manipulation, the penalty shaping does not point at the goal and SAC has no useful signal. Contrastive GCRL avoids this trap because its critic is goal-conditioned and learns reachability from data rather than from a stepwise distance proxy.

This nuance does not change the headline — contrastive cells dominate on curriculum average — but it should appear in the paper's discussion so we do not over-claim universal dominance.

## Files

- `results/data/processed/sac_per_seed_per_task.csv` — one row per (cell, task, seed) with best and end success.
- `results/data/processed/sac_per_task_aggregate.csv` — mean/std/sem/n per (cell, task).
- `results/data/processed/sac_cell_summary.csv` — curriculum-average per cell.

## Reproducing the pull

```python
import wandb
api = wandb.Api()
runs = list(api.runs('d_konoki/continual_sac', per_page=300))
# Filter: actor_mode in {'reset','persistent'}, critic_mode == actor_mode,
# step_penalty_reward == True. For each run, pull history(keys=['evaluator/success_rate'])
# and take the max as best_success.
```

Authentication uses a personal W&B API key; the script lives at the paper-side tooling and was run once on 2026-05-20.
