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
| D5 | `mixture_norm` metric in CKA inner loop | A | shipped (2026-05-08) | helper `mixture_to_vk_ratio` in `knowledge_pool.py`; emitted as `cka/actor_mixture_norm` / `cka/critic_mixture_norm` per inner step when `continual_config.log_mixture_norm=True` (default False). See `2026-05-08_d5_mixture_norm.md`. |
| D6 | Linear-probe task classifier (`eval_linear_probe.py`) | A & B | not started | section 3.4 / section 10 of plan |
| D7 | CKA diagnostic run on `actor_mode='cka', critic_mode='cka'` with `log_pool_cosine=True` | A | ready to launch | depends only on D1-D4 (shipped) |
| N1 | `state_mask.py` with `STABLE_INDICES = (0, 1, 2, 3)` | B | shipped (2026-05-08) | section 2 of plan |
| N2 | `decomposed_networks.py` (`b_shared`, `h_phi`, `h_dyn`, `phi_task`, `psi`) | B | shipped (2026-05-08) | smoke-tested, gradient isolation verified |
| N2b| Verify decomposed implementation matches SGCRL conventions (score, hyperparameters, actor loss) | B | shipped (2026-05-08) | see `2026-05-08_decomposed_critic_verification.md`; fixed 4 bugs (score=neg-L2, adaptive_entropy field, InfoNCE form, actor goal-rolling+entropy gate) |
| N3 | `critic_mode='decomposed'` config option + new flags | B | partial | new fields `dyn_aux_weight`, `phi_task_width`, `phi_task_depth` shipped; `critic_mode='decomposed'` accepted at the runner edge but not yet dispatched |
| N4 | Decomposed-critic learner path in `continual_learning.py` | B | shipped as sibling | new file `continual_learning_decomposed.py` with `ContinualDecomposedLearner` |
| N4b| Runner glue in `train_single_task` (network branch + learner branch + post-task extraction) | B | shipped (2026-05-08) | three blocks landed: (1) `make_decomposed_networks` build under `critic_mode=='decomposed'`; (2) learner branch constructs `ContinualDecomposedLearner` with all 8 prev_* carry kwargs; (3) early actor-pool / critic-q skip with new 17-tuple return + checkpoint extension. Early FLAG-side guards reject use_td / twin_q / use_image_obs / neg_bank / actor_mode='cka' / k_sample_k>0. Smoke at obs_dim=21 (10-task shape) passes. |
| N5 | Smoke test: one-task run with `critic_mode='decomposed'` and `dyn_aux_weight=0` (regression check) | B | unblocked (N4b shipped) | run on cluster |
| N6 | Single-cell sanity experiment with `dyn_aux_weight=1.0` | B | blocked on N5 | section 8 of plan |
| N7 | Full ablation grid (5 cells × 5 seeds × 10 tasks) | B | blocked on N6 | section 8 of plan |
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

1. ~~D1, D2, D3, D4, plan + tracking docs.~~ shipped 2026-05-08.
2. ~~N1, N2, N3 (partial), N4.~~ shipped 2026-05-08. Foundations (state
   mask, decomposed networks, sibling learner, config flags) are done
   and locally smoke-tested. Runner glue (N4b) is the remaining
   blocker.
3. ~~N2b: verify SGCRL conventions and fix bugs.~~ shipped 2026-05-08.
   Switched the score to inner product (was neg-L2 by mistake), fixed
   `adaptive_entropy` detection to match `entropy_coefficient is None`,
   reshaped the critic loss to use sigmoid-BCE / softmax-CE branches
   under `use_cpc`, added goal rolling and `use_action_entropy` gating
   to the actor loss. Smoke test re-run; gradient isolation preserved.
4. ~~N4b: apply the three-block runner patch.~~ shipped 2026-05-08. The
   network-construction, learner-construction, and post-task-extraction
   blocks are all in `train_single_task`; the main loop threads the
   eight decomposed `prev_*` kwargs across tasks and persists them in
   the per-task checkpoint under `decomposed_*` keys. Early FLAG-side
   guards reject incompatible flags before the replay server boots.
5. (next) **N5** regression check: persistent cell still matches the
   existing baseline. **N5b** decomposed cell with `dyn_aux_weight=0.0`
   should also match the baseline within seed noise (the shared body +
   h_phi + psi reduce to the existing critic structure when the dyn
   gradient is zeroed; phi_task adds a per-task additive head, so the
   match will be close but not exact — expect a small offset).
6. (after N5 passes) **N6** single-cell sanity experiment with
   `dyn_aux_weight=1.0`.
7. (parallel with N6) ~~D5~~ shipped 2026-05-08. **D6** linear-probe
   diagnostic remains (workstream A; code-only; can be done locally
   ahead of N5 if needed).
8. (after N6 passes) **N7** ablation grid + **D7** CKA diagnostic
   run.

---

## Branches and remotes

- `Okyumi/sgcrl` `section3_done`: this branch. Code + docs.
- `Okyumi/sgcrl` `clean`: code-only mirror; cherry-pick from
  `section3_done`.
- `Okyumi/builderbench` `main`: appendix port; not affected by this
  workstream unless a result here motivates an appendix experiment.
- `Okyumi/NeurIPS-2026---RL` `main`: paper repo. Updated when method
  / experiments sections need to reflect new content.
