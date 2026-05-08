# Plan: dynamics-aux + decomposed (s,a) embedding on sgcrl

This plan turns proposal 1 into concrete code changes against the
existing sgcrl tree. It is scoped tight, avoids touching anything not
needed, and lists the experiments that will validate (or invalidate)
the approach.

---

## 1. What we are building

The actor / critic forward pass becomes:

```
phi(s, a) = phi_shared(s, a) + phi_task(s, a)             # contrastive embedding
psi(g)    = psi(g)                                         # goal encoder, unchanged
score     = - || phi(s, a) - psi(g) ||_2                   # contrastive score (current sgcrl form)
```

Two encoders sit inside the body of `phi`. `phi_shared` is one network
with two heads: a contrastive head `h_phi` (output dim `repr_dim`) and
a dynamics head `h_dyn` (output dim equal to the number of stable
indices). `phi_task` is a separate network whose output also has
dim `repr_dim`. Both feed into the same elementwise sum.

The dynamics head is trained against `M . s'`, where `M` is a fixed
mask over state indices with stable cross-task semantics.

The actor stays as `actor_mode='reset'` (per the user's preference; CKA
on the actor side did not help). Critic CKA is dropped from this run;
it will reappear only if the cell-by-cell ablation later asks for it.

---

## 2. The mask M (no engineering required)

The codebase already contains
`docs/STATE_AND_GOAL_INDEX_SEMANTICS.md`, which lists per-task index
semantics. Combined with `env_utils.py` (`STATE_DIM_UNIFIED = 11`,
`GOAL_DIM_UNIFIED = 11`, all observations padded to `FULL_OBS_DIM = 22`),
the stable-index mask is:

| Index | Quantity                   | Stable across all 10 tasks? |
|-------|----------------------------|-----------------------------|
| 0     | end-effector x             | yes                         |
| 1     | end-effector y             | yes                         |
| 2     | end-effector z             | yes                         |
| 3     | gripper distance apart     | yes                         |
| 4     | object 1 x (or padding)    | no                          |
| 5     | object 1 y (or padding)    | no                          |
| 6     | object 1 z (or padding)    | no                          |
| 7     | quat / extra (or padding)  | no                          |
| 8     | quat / extra (or padding)  | no                          |
| 9     | quat / extra (or padding)  | no                          |
| 10    | quat / extra (or padding)  | no                          |

So `M = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]` over the 11 state indices,
and `d_M = 4`. The mask is task-agnostic by construction: every task's
end-effector is the same Sawyer arm with the same kinematics.

This is the cleanest form of the proposal. We keep it as a constant in
`contrastive/networks.py` (or a new `contrastive/state_mask.py`) so it
is reproducible and easy to ablate.

---

## 3. Network changes

### 3.1 New file `contrastive/decomposed_networks.py`

Adds a wrapper around the existing `make_networks(...)` that:

- builds `b_shared` (a `ResidualMLP` body) consuming
  `[state; action]`;
- adds two output heads on top of `b_shared`: a contrastive head
  `h_phi` (linear or 2-layer MLP to `repr_dim`) and a dynamics head
  `h_dyn` (linear to `d_M`);
- builds a separate `phi_task` network, same shape as the shared body
  + contrastive head, but with no dynamics output;
- exposes a `_repr_fn` whose `sa_repr` is `h_phi(b_shared([s; a])) +
  phi_task([s; a])`;
- exposes a separate apply function `dyn_apply(b_shared_params, h_dyn_params, s, a)`
  used by the dynamics auxiliary;
- builds `g_encoder` exactly as the current `make_networks` does.

We do **not** touch the existing `networks.py`. The new file imports
`ResidualMLP` and reuses every other helper. The branch should default
to the existing critic when a flag (`decomposed_critic=False`) is off,
so we can sanity-check that the refactor preserves baseline numbers.

### 3.2 Replay buffer

The replay buffer already produces `next_observation` in
`continual_builder.py` (line 155). The first 11 indices of
`next_observation` are the next state. The dynamics target is
`next_observation[:, 0:11] * M` for the masked dimensions only. No
buffer changes are needed.

The same buffer is used by the InfoNCE loss and the dynamics
auxiliary. They differ only in which fields of the same transition
they consume.

---

## 4. Learner changes

### 4.1 `contrastive/continual_learning.py`

Add a new path that activates when
`continual_config.decomposed_critic == True`:

- new fields on `ContinualTrainingState`:
  - `b_shared_params`, `b_shared_opt_state`
  - `h_phi_params`, `h_phi_opt_state`
  - `h_dyn_params`, `h_dyn_opt_state`
  - `phi_task_params`, `phi_task_opt_state`
  - `psi_params`, `psi_opt_state` (already exists conceptually as the
    goal-encoder portion of `q_params`)
- `__init__` builds these from `make_decomposed_networks`. At task `k`
  start:
  - `b_shared`, `h_phi`, `h_dyn`, `psi` carry over from task `k-1`
    (transfer; not reset)
  - `phi_task` is fresh (reset every task)
  - the actor is fresh (reset every task)
- `update_step` runs three losses inside the JIT body:
  - `L_InfoNCE` on the sum embedding (current task only)
  - `L_dyn` on `h_dyn(b_shared(s, a))` against `M . s'`
  - `L_actor` (unchanged from current code)
- gradient routing inside `update_step`:
  - `b_shared, h_phi, psi` see `L_InfoNCE` and a `mu`-scaled
    `L_dyn` (h_phi and psi do not receive dyn grads; b_shared
    receives both)
  - `h_dyn` sees only `L_dyn`
  - `phi_task` sees only `L_InfoNCE`
  - actor sees only `L_actor` (unchanged)

A single optax optimiser per parameter group (we already have this
pattern from the CKA refactor).

### 4.2 Mixed-task dynamics batch

The dynamics auxiliary should be **task-agnostic by construction**, so
its target distribution must not drift with the current task. Two
options:

- **A. Same-task dynamics batch.** The dyn batch is just the current
  task's hindsight batch. Cheapest. Risk: `b_shared` learns task-`k`
  dynamics specifically and forgets earlier tasks' dynamics, which
  defeats the point.
- **B. Mixed-task dynamics batch.** Maintain a small dyn-only
  buffer that pulls transitions proportionally from
  `D_1, ..., D_k`. Costs one extra reverb table OR one extra dataset
  iterator. The Sawyer dynamics is shared, so even a small buffer
  (a few thousand transitions per past task) is sufficient.

I propose starting with A for the first run (zero infra cost, fastest
turnaround), then upgrading to B if the linear probe shows
`b_shared` drifting toward task-specific dynamics.

The replay-buffer point in your question — "phi_shared and phi_task
use the same replay buffer" — is yes for InfoNCE. For the dynamics
auxiliary, option A reuses the same buffer; option B uses a separate
mixed-task buffer for the dyn loss only. Either way `phi_task` is
unaffected: it only consumes the InfoNCE batch.

---

## 5. Configuration

In `contrastive/continual_config.py`, add:

```python
decomposed_critic: bool = False
dyn_aux_weight: float = 1.0       # mu in the algorithm; sweep {0, 0.1, 1.0}
dyn_mask_indices: Tuple[int, ...] = (0, 1, 2, 3)
phi_task_depth: int = 2           # smaller than the shared body
phi_task_width: int = 256         # smaller than the shared body
mixed_task_dyn_buffer: bool = False  # turn on when option B is in
```

`decomposed_critic=False` reproduces the current sgcrl baseline
(persistent critic, reset actor) for a sanity check.

---

## 6. Acceptance criteria for the first run

Single cell of the ablation: `actor_mode='reset', critic_mode='persistent', decomposed_critic=True, dyn_aux_weight=1.0`.

Compare to the same cell with `decomposed_critic=False`.

- the dynamics loss `L_dyn` decreases monotonically across tasks (it
  should — Sawyer dynamics is consistent across tasks)
- average task-`k` success rate at end of `k` is at least as good as
  the persistent baseline for `k >= 2`
- `phi_shared` linear-probe task-classifier accuracy on the body output
  is below 1/N task accuracy + 5% (i.e., body is not absorbing task
  identity)

If all three pass, run the full ablation grid:

| dyn_aux_weight | actor_mode | critic_mode |
|----------------|------------|-------------|
| 0.0            | reset      | persistent  | (baseline equivalent)
| 0.1            | reset      | persistent  |
| 1.0            | reset      | persistent  |
| 1.0            | reset      | reset       | (does dyn-aux save reset critic?)

`actor_mode=cka` and `critic_mode=cka` are dropped — we have evidence
they do not help in this setting and will not include them in the
paper's primary comparison.

---

## 7. Diagnostics to log every task

Independent of the ablation, every run should log:

- stratified InfoNCE accuracy: within-task hindsight, same-task
  off-trajectory, cross-task (cross-task only if past data ever
  enters a softmax — for proposal 1 this never happens, but it is
  cheap to add and useful to confirm)
- linear-probe task classifier on `b_shared(s, a)`: target is
  near-chance accuracy; rising probe accuracy = `b_shared` is
  absorbing task identity (failure mode)
- per-index masking sensitivity for the contrastive head: drop in
  within-task accuracy when state index `i` is masked at evaluation
  time. Object-slot indices should be the dominant features for
  `phi_task` and roughly invariant for `phi_shared`. If `b_shared`
  leans heavily on object indices, the auxiliary is not biting.
- `L_dyn` value per task, per index. The dyn loss should be similar
  magnitude across tasks (because masked state indices have shared
  semantics).

The linear probe is a few minutes of post-hoc work per checkpoint and
is the single most informative diagnostic — invest in it first.

---

## 8. Risk register

| risk | likelihood | mitigation |
|------|------------|------------|
| `phi_shared` collapses to constant | low | dynamics aux gives it a real job |
| `phi_task` absorbs everything | medium | start with `phi_task` smaller than shared body; sweep `mu` upward |
| `b_shared` drifts to task-specific | medium | mixed-task dyn buffer (option B); linear probe to detect |
| dyn easier than InfoNCE | medium | tune `mu`; consider a small loss weight schedule |
| Sawyer 4 stable dims is too few | low | the arm is a 7-DoF manipulator; 4 dims is enough to anchor a useful representation |
| code regression from refactor | medium | gate the new path behind `decomposed_critic` flag; baseline cell with the flag off must reproduce existing numbers |

---

## 9. Concrete order of work

1. Add `state_mask.py` with `STABLE_INDICES = (0, 1, 2, 3)` constant.
2. Write `decomposed_networks.py` with the b_shared / h_phi / h_dyn /
   phi_task split, leaving the existing `networks.py` untouched.
3. Smoke-test the new module in a Python REPL with random `(s, a, s')`
   batches. Confirm shapes, gradient flow through all four components,
   and that `dyn_apply` consumes only `b_shared_params` and `h_dyn_params`.
4. Add the new `decomposed_critic` code path to `continual_learning.py`
   with the additional state slots and the three-loss inner step.
   Gate it on the flag.
5. Smoke-test the learner on a one-task run with `decomposed_critic=True`,
   `dyn_aux_weight=0.0`. Should match baseline within seed noise.
6. Run the single-cell sanity experiment from §6 with
   `dyn_aux_weight=1.0`. Validate the three acceptance criteria.
7. If criteria pass, run the §6 ablation grid for the full 10-task
   sequence with five seeds.
8. Add the linear-probe and per-index-masking diagnostics to the
   evaluation pipeline (after step 4, in parallel with step 6).
9. If the linear probe shows `b_shared` drift, switch on
   `mixed_task_dyn_buffer` (option B) and re-run the failing cells.
10. Once results stabilise, update the paper's method section to
    reflect the actual algorithm and write the related-work
    repositioning of CKA-RL as a comparison rather than a foundation.

---

## 10. What this plan deliberately does not do

- No new actor mechanism. `actor_mode='reset'` everywhere.
- No critic CKA. Dropped from the comparison.
- No transformer / channel-attention encoder.
- No memory bank / retrieval-augmented critic.
- No goal-encoder split. `psi` stays as a single shared encoder.
- No knowledge pool, no merge rule, no `alpha_logits`.

Everything we drop is something we have evidence does not help in
this setting, or that would add complexity without a clear
empirical reason. We can reintroduce any of these as a follow-up
once the dyn-aux + decomposed-encoder result is established.
