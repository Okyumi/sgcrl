# Plan: dynamics-aux + decomposed (s,a) embedding on sgcrl

This plan turns proposal 1 into concrete code changes against the
existing sgcrl tree. Two pieces are interleaved:

1. **A new `critic_mode='decomposed'` variant** that adds `phi_shared`
   with two heads (contrastive + masked dynamics) plus `phi_task`,
   matching proposal 1.
2. **Diagnostic instrumentation that explains why CKA underperformed**
   on the existing `actor_mode='cka'` runs, so the paper has both a
   diagnosis and a positive new result.

Existing configs stay intact. The 9-cell `actor_mode x critic_mode`
ablation grows to a 12-cell grid: the new column
`critic_mode='decomposed'` is added; existing columns
(`reset`, `persistent`, `cka`) and the existing
`actor_mode x critic_mode` cells are not touched.

---

## 1. What we are building

The actor / critic forward pass under `critic_mode='decomposed'` becomes:

```
phi(s, a) = phi_shared(s, a) + phi_task(s, a)              # contrastive embedding
psi(g)    = psi(g)                                          # goal encoder, unchanged
score     = - || phi(s, a) - psi(g) ||_2                    # current sgcrl form
```

`phi_shared` is one network with two heads:

- `h_phi`: contrastive head, `b_shared([s; a]) -> R^repr_dim`
- `h_dyn`: dynamics head, `b_shared([s; a]) -> R^d_M`

`phi_task` is a separate (smaller) network whose output dim is
`repr_dim`, added elementwise to `h_phi(b_shared([s; a]))`.

Three losses run inside the inner JIT step:

- `L_InfoNCE` on the sum embedding (current task only)
- `L_dyn = || h_dyn(b_shared(s, a)) - M . s' ||_2^2`  (Hadamard mask)
- `L_actor` (unchanged from current code)

Gradient routing inside `update_step`:

| component       | InfoNCE | dyn | actor |
|-----------------|:-------:|:---:|:-----:|
| `b_shared`      |  yes    | yes |  no   |
| `h_phi`         |  yes    | no  |  no   |
| `h_dyn`         |  no     | yes |  no   |
| `phi_task`      |  yes    | no  |  no   |
| `psi`           |  yes    | no  |  no   |
| actor           |  no     | no  | yes   |

At task `k > 0` boundary:

- `b_shared, h_phi, h_dyn, psi` carry over from task `k-1` (transfer)
- `phi_task` is reset (fresh weights every task)
- actor is reset every task (`actor_mode='reset'`, default)

---

## 2. The mask M (no engineering required)

`docs/STATE_AND_GOAL_INDEX_SEMANTICS.md` already lists per-task index
semantics. With `STATE_DIM_UNIFIED = 11` and consistent zero-padding,
the cross-task-stable indices are:

| index | quantity                  | stable across all 10 tasks? |
|-------|---------------------------|:---------------------------:|
| 0     | end-effector x            | yes                         |
| 1     | end-effector y            | yes                         |
| 2     | end-effector z            | yes                         |
| 3     | gripper distance apart    | yes                         |
| 4..10 | object slots / quat / pad | no                          |

So `M = (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)` over the 11 state indices,
`d_M = 4`. Stored as a constant in a new module
`contrastive/state_mask.py`.

The dynamics target is `next_observation[:, 0:11] * M`. The
`next_observation` field is already in the transition (see
`continual_builder.py:155`); no buffer changes are needed.

---

## 3. CKA-failure diagnostics (added to existing CKA runs)

Three diagnostics, runnable on the current `actor_mode='cka'` /
`critic_mode='cka'` cells without changing the algorithm. These exist
to explain *why* CKA underperformed and to make the paper's negative
result defensible.

### 3.1 Pairwise cosine similarity of pool vectors

Hypothesis from the audit: the per-task knowledge vectors `v_j` produced
by the contrastive objective are highly correlated, so a softmax
mixture over them collapses to rank one. We have not yet measured this.

New utility in `contrastive/knowledge_pool.py`:

```python
def pool_cosine_matrix(pool: CKAPool) -> jnp.ndarray:
    """Return the pairwise cosine similarity matrix of active pool vectors.

    Shape (n_active, n_active). Inactive slots are excluded. Output is
    symmetric; diagonal is one. NaN-safe for zero-norm vectors.
    """
```

Implementation: flatten each active slot to a single vector (reuse the
flattening already in `_merge_most_similar_pair_host`), compute
`(flat @ flat.T) / (norms[:, None] * norms[None, :])`. Pure JAX, no
host syncs except the active-mask read.

Logging: at the end of every task `k > 0` for any CKA cell, the
orchestrator computes the actor-pool and critic-pool cosine matrices
on the host and logs:

- `pool_cos_mean_offdiag` (scalar)
- `pool_cos_max_offdiag` (scalar)
- `pool_cos_min_offdiag` (scalar)
- `pool_cos_matrix` (a small array, written to a per-task `.npy` for
  paper figures)

If `pool_cos_mean_offdiag > 0.9` consistently, the audit's hypothesis
is confirmed and the paper has its empirical reason for CKA's failure.

**REVISED 2026-05-12 (after running the C0 diagnostic):** the pool
vectors do NOT collapse. Observed actor `mean_offdiag` ~ 0.17,
critic ~ 0. The hypothesis above is refuted.

**FURTHER REVISED 2026-05-13 (after pulling all 3-seed W&B data):**
the "gradient null space" explanation in
`docs/2026-05-12_cka_failure_results.md` is also wrong. The actor
mixture does decay to ~0.02 (near zero), but the critic mixture
decays only to ~0.5 (not zero) and the critic `α_scale` actively
**grows** on later tasks (up to ~2.0 at k=8). Both halves of the
original collapse hypothesis are refuted. See
`docs/2026-05-13_wandb_findings.md` for the quantitative table and
`docs/2026-05-13_revised_hypothesis_plan.md` for a fresh
hypothesis-test plan.

### 3.2 Per-task contribution of `v_k` vs the mixture term

Hypothesis: even when `alpha` is technically trainable
(task `k >= 3` under canonical CKA-RL hand-off), most of the actor
update is being absorbed by `v_k` rather than the mixture
`sum_j alpha_j v_j`, because the mixture term is approximately
constant in that regime.

New scalar metric inside `update_step` for any CKA path:

- `mixture_norm = || sum_j alpha_j v_j || / || v_k ||`

If this is small (< 0.1) throughout training, the mixture term is
effectively a small bias that the actor learns to ignore. The paper
can then say that CKA degrades to per-task residual finetuning, which
is what we observe.

### 3.3 Actor / critic alpha trajectories already exist

The current code already logs `actor_alpha_max`, `actor_alpha_scale`,
`actor_alpha_entropy` (and critic equivalents). These showed
flat-zero / flat-one at tasks 1 and 2 due to the masked-softmax
degeneracy with `<= 1` active slot. We keep these as-is.

### 3.4 Linear-probe task classifier on `b_shared` (decomposed runs only)

Healthy `b_shared` should produce features that do *not* easily
classify task identity. Failure mode: `b_shared` absorbs task identity
when exposed to mixed-task gradients.

Eval-time-only diagnostic (no training cost): freeze `b_shared`, fit a
linear classifier on `b_shared(s, a)` to predict task index, report
test accuracy. Target is near-chance (1/N for N tasks); a probe
accuracy above ~50% in the 10-task setting flags the body absorbing
task identity. Implementation lives in a new `eval_linear_probe.py`
script that loads a checkpoint and fits the probe in a few minutes
of CPU.

---

## 4. Network changes

### 4.1 New file `contrastive/decomposed_networks.py`

A wrapper around the existing `make_networks(...)` that:

- builds `b_shared` (a `ResidualMLP` body) consuming `[state; action]`;
- adds two output heads on top of `b_shared`:
  - `h_phi`: linear or two-layer MLP to `repr_dim`
  - `h_dyn`: linear to `d_M`
- builds a separate `phi_task` network, same shape pattern as the
  shared body + contrastive head, but with smaller width/depth
  (default: width 256, depth 2) and no dynamics head;
- exposes a `_repr_fn` whose `sa_repr` is `h_phi(b_shared([s; a])) +
  phi_task([s; a])`;
- exposes a separate apply function
  `dyn_apply(b_shared_params, h_dyn_params, s, a)` used by the
  dynamics auxiliary;
- builds `g_encoder` exactly as the current `make_networks` does.

Existing `networks.py` is **not touched**. The new module is gated
behind the `critic_mode='decomposed'` config path.

### 4.2 Replay buffer

No changes. `next_observation` is already produced by
`continual_builder.py:155`. The first 11 indices are the next state.
The dynamics target is `next_observation[:, 0:11] * M`.

The same buffer is used by the InfoNCE loss and the dynamics
auxiliary in the first iteration (option A, see §6).

---

## 5. Learner changes

### 5.1 `contrastive/continual_learning.py`

Add a new code path activated by `critic_mode == 'decomposed'`. The
existing `'reset'`, `'persistent'`, `'cka'` paths are preserved
verbatim. New fields appear on `ContinualTrainingState` as
`Optional[...] = None` so existing checkpoints stay loadable.

New training-state fields (all optional):

```
b_shared_params, b_shared_opt_state
h_phi_params, h_phi_opt_state
h_dyn_params, h_dyn_opt_state
phi_task_params, phi_task_opt_state
```

`psi_params` continues to live where it currently does (the
goal-encoder portion of `q_params`).

Inner JIT step structure (decomposed path only):

```
for sgd_step in range(num_sgd_steps_per_step):
    # 1. compose
    sa_repr_shared = h_phi(b_shared(state, action))
    sa_repr_task   = phi_task(state, action)
    sa_repr        = sa_repr_shared + sa_repr_task
    g_repr         = psi(goal)
    s_next_pred    = h_dyn(b_shared(state, action))

    # 2. losses
    L_InfoNCE = infonce(sa_repr, g_repr)                          # current sgcrl InfoNCE
    L_dyn     = mean(((s_next_pred - mask * next_state)) ** 2)    # masked MSE

    # 3. component-wise grad
    grad_b_shared = grad_b_shared(L_InfoNCE) + mu * grad_b_shared(L_dyn)
    grad_h_phi    = grad_h_phi(L_InfoNCE)
    grad_h_dyn    = mu * grad_h_dyn(L_dyn)
    grad_phi_task = grad_phi_task(L_InfoNCE)
    grad_psi      = grad_psi(L_InfoNCE)

    # 4. optimiser apply for each group
```

Implementation note: routing per-component gradients is straightforward
with `jax.value_and_grad` taking a dict of params and unpacking; same
pattern we use in the CKA bundle update.

### 5.2 `__init__` boundary handling

When `critic_mode == 'decomposed'`:

- task 0: build `b_shared, h_phi, h_dyn, psi, phi_task` from scratch.
- task `k > 0`: carry over `b_shared, h_phi, h_dyn, psi` (their
  params and optimiser states) from the previous task. Reset
  `phi_task` (fresh init) and reset its optimiser state. Actor is
  reset per `actor_mode='reset'`.

The legacy `q_params` slot maps to the goal-encoder side; the
state-action side is replaced by the `(b_shared, h_phi, phi_task)`
trio. Accessors (`get_variables`, `q_params`, etc.) need updates so
the orchestrator and evaluator pull the composed
`phi(s,a) = h_phi(b_shared(s,a)) + phi_task(s,a)` correctly.

### 5.3 Logging

Per-step metrics added:

- `decomp/L_dyn` (scalar)
- `decomp/L_dyn_per_dim` (4-vec, masked dims only)
- `decomp/sa_shared_norm` and `decomp/sa_task_norm` (compose
  contributions)
- (CKA cells) `cka/pool_cos_mean_offdiag`, `cka/pool_cos_max_offdiag`,
  `cka/mixture_norm` (per §3.1, §3.2)

---

## 6. Configuration

In `contrastive/continual_config.py`, add:

```python
# Existing fields stay untouched.

# new option in the existing critic_mode flag
# critic_mode: 'reset' | 'persistent' | 'cka' | 'decomposed'

# decomposed-critic-only flags (only read when critic_mode='decomposed')
dyn_aux_weight: float = 1.0       # mu; sweep in {0, 0.1, 1.0}
dyn_mask_indices: Tuple[int, ...] = (0, 1, 2, 3)
phi_task_depth: int = 2           # smaller than the shared body
phi_task_width: int = 256         # smaller than the shared body
mixed_task_dyn_buffer: bool = False

# CKA-diagnostic flags (default off so existing runs are unaffected)
log_pool_cosine: bool = False
log_mixture_norm: bool = False
```

Defaults preserve all existing run configurations. Cosine-sim and
mixture-norm logging are off unless explicitly enabled per-run, so no
existing experiment changes behaviour.

---

## 7. Mixed-task dynamics buffer

The dynamics auxiliary should be task-agnostic by construction. Two
options for sourcing dynamics batches:

- **A. Same-task buffer (default).** The dyn batch is the current
  task's hindsight batch. Cheapest. Risk: `b_shared` learns task-`k`
  dynamics specifically and forgets earlier tasks' dynamics.
- **B. Mixed-task buffer.** Maintain a small dyn-only buffer that
  pulls transitions proportionally from `D_1, ..., D_k`. Costs one
  extra reverb table or one extra dataset iterator.

Start with A. If the linear probe (§3.4) shows `b_shared` drifting
toward task-specific dynamics across tasks, flip
`mixed_task_dyn_buffer=True` for the next round.

To answer the user's earlier question: **`phi_shared` and `phi_task`
share the same replay buffer for the InfoNCE loss.** The dynamics
auxiliary uses the same buffer under option A, and a separate
mixed-task buffer under option B. `phi_task` itself never sees
dynamics data.

---

## 8. Acceptance criteria for the first cell

Single cell first: `actor_mode='reset', critic_mode='decomposed', dyn_aux_weight=1.0`,
all other defaults intact (adapt_heads_only, K_max=5, target_entropy=-2,
ResidualMLP 1024x4, 8M steps/task, three seeds for the smoke run).

Compare to: `actor_mode='reset', critic_mode='persistent'` (existing
baseline; already in the 9-cell grid).

Pass criteria:

1. `L_dyn` decreases monotonically across tasks (Sawyer dynamics
   are consistent, so the body should learn them once).
2. Average task-`k` success rate is at least as good as the
   persistent baseline for `k >= 2`, on all three seeds.
3. Linear-probe task-classifier accuracy on `b_shared` output is
   below `1/N + 5%` (i.e., body is not absorbing task identity).

If all three pass, run the full ablation grid with five seeds:

| dyn_aux_weight | actor_mode | critic_mode  | comment             |
|----------------|------------|--------------|---------------------|
| -              | reset      | persistent   | existing baseline   |
| 0.0            | reset      | decomposed   | dyn-aux off         |
| 0.1            | reset      | decomposed   | dyn-aux weak        |
| 1.0            | reset      | decomposed   | dyn-aux full        |
| 1.0            | reset      | reset        | dyn-aux + reset crit|

The original 9 cells (actor x critic in `{reset, persistent, cka}^2`)
remain unchanged — they continue to populate the 9-cell grid that the
paper's main table is built around. The decomposed column is an added
column.

---

## 9. CKA diagnostic experiments (no algorithm change)

Independently of the decomposed-critic work, run the following on the
existing `actor_mode='cka', critic_mode='cka'` cell with three seeds,
turning on `log_pool_cosine=True` and `log_mixture_norm=True`:

- task-by-task pool cosine matrix (saved per task)
- task-by-task `mixture_norm`
- existing `actor_alpha_max`, `actor_alpha_scale`, `actor_alpha_entropy`

Expected pattern under the audit hypothesis:

- task 1: pool empty, all alpha metrics flat
- task 2: pool size 1, alpha metrics flat (single-slot degeneracy)
- task 3: pool size 2, alpha metrics finally trainable, but
  `pool_cos_mean_offdiag > 0.9` and `mixture_norm < 0.1`
- task 4+: pool grows, cosine remains > 0.9 (vectors stay aligned),
  merge rule starts collapsing, `mixture_norm` stays low

If observed, this is the empirical content of "CKA fails because the
per-task knowledge vectors do not span useful directions in this
setting", which the paper claims qualitatively. With this data the
claim becomes quantitative.

**REVISED 2026-05-12 (after the C0 diagnostic ran):** the
mixture-norm half of the expected pattern held; the cosine
alignment half did NOT (vectors stayed roughly orthogonal, mean
actor cosine ~0.17, critic ~0). The empirical claim was
reframed as "the mixture term lives in the gradient null space of
contrastive losses".

**FURTHER REVISED 2026-05-13 (after pulling 3-seed W&B data):**
the "gradient null space" reframing is also wrong. Critic
mixture-norm decays to ~0.5 (not zero) and critic `α_scale`
actively **grows** on later tasks. The mixture term is alive on
the critic side. The actor mixture does die, so the failure mode
is **asymmetric across actor vs critic**, which neither of the
prior hypotheses predicted. See
`docs/2026-05-13_wandb_findings.md` for numbers and
`docs/2026-05-13_revised_hypothesis_plan.md` for the fresh
hypothesis-test plan.

This run uses the existing CKA code; no algorithmic changes are
required, only logging.

---

## 10. Diagnostics to log every run

Independent of which cell:

- stratified InfoNCE accuracy
  - within-task hindsight (hard, baseline)
  - same-task off-trajectory (medium)
  - cross-task (easy if separable; flag if much higher than baseline)
- linear-probe task classifier on `b_shared(s, a)` (decomposed cells)
  or on the existing critic body (other cells)
- per-index masking sensitivity for the contrastive head (eval-time)
- gradient-norm vs loss curve on the body / encoder

The single most informative one, given cost, is the linear probe.
Implement it first in `eval_linear_probe.py`.

---

## 11. Order of work

1. Add `state_mask.py` with `STABLE_INDICES = (0, 1, 2, 3)`.
2. Add `pool_cosine_matrix(...)` and `pool_cos_summary(...)` to
   `knowledge_pool.py`. Unit test against a hand-built pool with
   known vectors.
3. Wire `log_pool_cosine` and `log_mixture_norm` into the existing
   CKA path of `continual_learning.py`. Default off. Push.
4. Launch the **CKA diagnostic run** (§9) on three seeds. This runs
   in parallel with the next steps and produces the paper figures
   for the negative-result section.
5. Write `decomposed_networks.py` with the `b_shared / h_phi / h_dyn /
   phi_task` split. Smoke-test in a Python REPL with random batches:
   verify shapes, gradient flow, and that `dyn_apply` consumes only
   `b_shared_params` and `h_dyn_params`.
6. Add the `critic_mode='decomposed'` code path to
   `continual_learning.py` with the additional state slots and the
   three-loss inner step. Default off via flag.
7. Smoke-test the learner with `critic_mode='decomposed'` and
   `dyn_aux_weight=0.0` on a one-task run. Should match the
   persistent baseline within seed noise (this is the regression
   check).
8. Run the **single-cell sanity experiment** from §8 with
   `dyn_aux_weight=1.0`. Validate the three acceptance criteria.
9. Implement `eval_linear_probe.py` in parallel with step 8.
10. If the sanity criteria pass, run the §8 ablation grid (5 cells x
    5 seeds = 25 runs) for the full 10-task sequence.
11. If the linear probe shows `b_shared` drift, switch on
    `mixed_task_dyn_buffer` and re-run the failing cells.
12. Update `docs/implementation_tracking.md` after each step. Push
    code + docs to `section3_done`; cherry-pick code-only commits to
    `clean`.

---

## 12. What this plan deliberately does not do

- Does not modify the existing `actor_mode='reset' | 'persistent' |
  'cka'` paths.
- Does not modify the existing `critic_mode='reset' | 'persistent' |
  'cka'` paths.
- Does not change defaults for any existing run config.
- Does not introduce a new actor mechanism. `actor_mode='reset'` for
  the new cells; the existing 9-cell grid keeps its CKA actor
  variants untouched.
- Does not introduce transformer / channel-attention or
  retrieval-augmented critic. Out of scope for this contribution.
- Does not split the goal encoder. `psi` stays as a single shared
  encoder.

The new `critic_mode='decomposed'` is a strict addition, gated by
flag. Reverting the column to nothing is one config flip away; existing
runs reproduce bit-for-bit.

---

## 13. Risk register

| risk                                  | likelihood | mitigation |
|---------------------------------------|------------|------------|
| `phi_shared` collapses to constant    | low        | dynamics aux gives it a real job |
| `phi_task` absorbs everything         | medium     | `phi_task` smaller (256x2) than shared body; sweep `mu` |
| `b_shared` drifts to task-specific    | medium     | mixed-task dyn buffer (option B); linear probe to detect |
| dyn easier than InfoNCE               | medium     | tune `mu`; consider weight schedule |
| 4 stable dims is too few              | low        | 7-DoF Sawyer arm summarised by 4-dim end-effector + gripper is enough to anchor a useful representation |
| code regression from refactor         | medium     | gate behind `critic_mode='decomposed'`; `dyn_aux_weight=0` cell must reproduce persistent baseline |
| cosine-sim logging slows existing runs| very low   | pure-JAX, host-only at task boundaries (rare); off by default flag |

---

## 14. Status & tracking

A new `docs/implementation_tracking.md` will be created in this push
(and updated after every code change), per the workflow preference.
This plan is the entry for "proposal 1: decomposed embedding +
dynamics auxiliary" and "CKA diagnostic instrumentation".
