# Implementation tracking

Single source of truth for which pieces of the new
decomposed-critic + CKA-diagnostics work have shipped, which are in
flight, and which are still on paper. Updated on every push.

Workflow reminders (per the user's preferences in memory):
- code goes to both `section3_done` (with docs) and `clean` (code-only)
- docs go to `section3_done` only
- author always Okyumi (yd2247@nyu.edu)
- existing run configurations stay intact; new behaviour is gated
  behind config flags

---

## Plan reference

`docs/plan_proposal1_dyn_aux.md` is the master plan for two parallel
workstreams.

- **A. CKA-failure diagnostics.** Empirically explain why CKA does not
  help in this setting, so the paper has a defensible negative
  result. Sections 3 and 9 of the plan.
- **B. New `critic_mode='decomposed'` algorithm.** Proposal 1 from the
  research-question session: split `phi(s, a)` into a shared body
  (with two heads, contrastive and dynamics) plus a per-task encoder.
  Sections 1, 4, 5, 6 of the plan.

Both workstreams preserve existing 9-cell ablation columns; the
decomposed column is added, not substituted.

---

## Status table

| ID | Item | Workstream | Status | Notes |
|----|------|------------|--------|-------|
| D1 | `pool_cosine_matrix`, `pool_cosine_summary` on `CKAPool` | A | shipped | new functions in `contrastive/knowledge_pool.py` |
| D2 | `cosine_matrix_from_vectors`, `cosine_summary_from_vectors` for legacy `KnowledgePool` | A | shipped | same file |
| D3 | `continual_config.log_pool_cosine` flag (default `False`) | A | shipped | `contrastive/continual_config.py` |
| D4 | Post-task cosine logging in `run_continual_contrastive.py` for actor and critic pools (W&B + per-task `.npy`) | A | shipped | gated on `D3` |
| D5 | `mixture_norm` metric in CKA inner loop | A | not started | section 3.2 of plan |
| D6 | Linear-probe task classifier (`eval_linear_probe.py`) | A & B | not started | section 3.4 / section 10 of plan |
| D7 | CKA diagnostic run on `actor_mode='cka', critic_mode='cka'` with `log_pool_cosine=True` | A | ready to launch | depends only on D1-D4 (shipped) |
| N1 | `state_mask.py` with `STABLE_INDICES = (0, 1, 2, 3)` | B | not started | section 2 of plan |
| N2 | `decomposed_networks.py` (`b_shared`, `h_phi`, `h_dyn`, `phi_task`, `psi`) | B | not started | section 4 of plan |
| N3 | `critic_mode='decomposed'` config option + new flags | B | not started | section 6 of plan |
| N4 | Decomposed-critic learner path in `continual_learning.py` | B | not started | section 5 of plan |
| N5 | Smoke test: one-task run with `critic_mode='decomposed'` and `dyn_aux_weight=0` (regression check) | B | not started | section 11 step 7 of plan |
| N6 | Single-cell sanity experiment with `dyn_aux_weight=1.0` | B | not started | section 8 of plan |
| N7 | Full ablation grid (5 cells × 5 seeds × 10 tasks) | B | not started | section 8 of plan |
| N8 | Mixed-task dynamics buffer (option B) — only if N6 fails the linear probe | B | held | section 7 of plan |

---

## What can run now

Item D7 is unblocked. To launch:

```
python run_continual_contrastive.py \
    --actor_mode=cka \
    --critic_mode=cka \
    [other defaults preserved] \
    [continual_config.log_pool_cosine=True]
```

Three seeds is enough to confirm the cosine pattern from the audit
hypothesis. Expected pattern:

- task 1: `n_active=0`, off-diagonal stats are NaN (no pairs)
- task 2: `n_active=1`, off-diagonal stats are NaN
- task 3 onward: `n_active=2..5`, `mean_offdiag` and
  `max_offdiag` should sit close to 1.0 if the audit hypothesis is
  right. If they are below 0.5, the hypothesis is wrong and we need
  a different explanation for CKA's failure.

Outputs:

- per-step W&B keys `pool_cosine_actor/{n_active, mean_offdiag,
  max_offdiag, min_offdiag}` and `pool_cosine_critic/...` (the latter
  only when `critic_mode='cka'`).
- per-task `.npy` matrices in the checkpoint directory:
  `pool_cosine_actor_task{k}.npy`, `pool_cosine_critic_task{k}.npy`.

These feed the negative-result figure in the paper's analysis section.

---

## Order of next pushes

1. (this push) D1, D2, D3, D4, plan + tracking docs.
2. (next) Implement N1, N2, N3, N4. Single PR-style commit per
   component; smoke test after each.
3. (after N4) Run N5; verify regression to baseline.
4. (after N5 passes) Run N6 single-cell sanity experiment.
5. (parallel with N6) Implement D5, D6.
6. (after N6 passes) Launch N7 ablation grid + D7 diagnostic run.

---

## Branches and remotes

- `Okyumi/sgcrl` `section3_done`: this branch. Code + docs.
- `Okyumi/sgcrl` `clean`: code-only mirror; cherry-pick from
  `section3_done`.
- `Okyumi/builderbench` `main`: appendix port; not affected by this
  workstream unless a result here motivates an appendix experiment.
- `Okyumi/NeurIPS-2026---RL` `main`: paper repo. Updated when method
  / experiments sections need to reflect new content.
