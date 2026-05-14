# 2026-05-14 — Mechanism Q&A: why decomposed-critic helps, and what C2 task-8 is doing

This note answers two questions from the 2026-05-14 review:

1. **Why does `b_shared + phi_task` help over the CKA-mixture critic?**
2. **Why does C2 climb on task k=8 while C0 (CKA) drops?**

It is meant as a record of the current mechanistic understanding and the
read of the W&B numbers as of this timestamp. Sources for numerical
claims are the CSVs under `docs/wandb_analysis/csv/` and the
corresponding PNGs under `docs/wandb_analysis/png/`.

---

## Q1 — Why `b_shared + phi_task` over a CKA mixture

The decomposed critic forms the (s,a) representation as

```
sa_repr(s,a) = h_phi(b_shared(s,a)) + phi_task(s,a)
```

The CKA-mixture critic forms it (schematically) as

```
sa_repr_mix(s,a) = sa_repr_base(s,a) + sum_j alpha_j * v_j
```

where the `{v_j}` are frozen per-task pool vectors and `alpha` is a
CKA-derived mixture over the pool. The mixture term is **state- and
action-independent**: once a task is locked, `sum_j alpha_j v_j` is a
constant vector in the representation space.

The two downstream consumers of `sa_repr` both quotient out additive
constants of this form:

- **Actor (argmax over actions).** The actor picks
  `argmax_a sa_repr(s,a) . g_repr(g)`. Adding a constant vector `c` to
  every `sa_repr(s,a)` shifts every score by `c . g_repr(g)`, which is
  the same scalar for every candidate action. The argmax is invariant.
  The gradient through the mixture term is therefore in the null space
  of the actor's induced behaviour on a single (s,g) pair.
- **InfoNCE softmax over goals.** The contrastive loss is a softmax
  over goals. Adding the same constant to every (s,a) representation
  again shifts every logit by the same scalar; softmax is invariant to
  uniform shifts.

This is the geometric statement of the H1/H2 result we already have on
W&B: across 24 (task, seed) pairs the critic `alpha_scale` is *alive*
(non-zero), but **anti-correlated** with end-of-task success
(Pearson r = -0.54, p = 0.007; see
`docs/wandb_analysis/csv/h1_h2_alpha_vs_success.csv`). A live but
anti-predictive mixture is exactly what you get when the mixture rides
on top of a representation that the downstream losses cannot distinguish
from "no mixture at all" — when the mixture term grows it indicates the
*base* representation has drifted, not that the mixture is contributing
useful knowledge.

`phi_task(s,a)` does not suffer from this. It is a small per-task
ResidualMLP that takes (s,a) as input. Its contribution varies with
both s and a, so:

- Different actions at the same state get different `phi_task`
  contributions ⇒ the argmax is genuinely modulated.
- Different (s,a) inputs get different shifts ⇒ the InfoNCE logits are
  not all shifted by the same constant ⇒ the softmax is not invariant.

Two further mechanistic points:

- **`b_shared` is the only continual constraint.** Section 3 of the
  paper carries `b_shared`, `h_phi`, `h_dyn`, `psi` forward between
  tasks and re-initialises only `phi_task`. So the cross-task transfer
  channel is the *body*, not a learned linear combination of frozen
  vectors. This is the structural difference that the CKA mixture
  cannot express.
- **`L_dyn` shapes `b_shared` to be predictive of next-state.** The
  masked-dynamics auxiliary
  `h_dyn(b_shared(s,a)) -> s'[STABLE_INDICES]` forces the body to
  encode the four EE/gripper coordinates that *are* invariant across
  the 10 Sawyer tasks. The mixture critic has no analogous mechanism
  binding the shared subspace to physically transferable features.

This is also consistent with the H3 finding (`docs/2026-05-14_h3_logsumexp_test.md`):
the mixture critic's `alpha_scale` only fires when `logsumexp` spikes
(tasks k=5 and k=8). Those are the runs where the *base* representation
is blowing up — the mixture is a numerical-stability response, not a
knowledge channel.

**Net.** The CKA mixture lives in the null space of both downstream
losses, so its gradient is approximately zero modulo a numerical-stability
signal that anti-correlates with success. The decomposed critic
escapes this null space by construction (state-and-action-dependent
perturbation) and adds an explicit physically-grounded transfer channel
(`b_shared` shaped by `L_dyn`).

---

## Q2 — Why does C2 climb on task k=8 while C0 (CKA) drops?

Short version: **too early to claim**. Inspect
`docs/wandb_analysis/png/task8_success_c0_vs_c2.png`.

What the W&B numbers actually say as of this timestamp:

- C0 (CKA, 3 seeds) on task k=8: 159 evaluator steps logged. Median
  success ~0.10 with intermittent spikes to 0.30–0.50.
- C2 (decomposed, 3 seeds) on task k=8: **25 evaluator steps logged**
  so far (resubmit after the gl017 infrastructure crash; see
  `docs/2026-05-14_c1_crash_and_ckpt_collision.md` for the crash
  template). Median ~0.10 in roughly the same band as C0 at the same
  step count.

So at matched evaluator-step counts the two runs are sitting on top of
each other. The "C2 climbs while C0 drops" reading was an artefact of
looking at C0 at step 159 vs C2 at step 25 and not normalising for the
horizon difference. Until C2 finishes the same 159 evaluator steps the
comparison is not interpretable.

### Steps vs parallelism vs intrinsic difficulty — opinion

The user asked which of the three is the binding constraint for k=8
(`sawyer_window_close`). My read:

- **Intrinsic difficulty (binding constraint).** `sawyer_window_close`
  is one of the harder Sawyer manipulation tasks because the success
  criterion is a contact-driven displacement of an actuated handle —
  the goal vector lives in the 4-d `STABLE_INDICES` subspace
  (EE xyz + gripper) only via a coupled handle dynamics that the agent
  has to *find*. The other 7 prior tasks in the curriculum do not
  exercise that contact mode. So the policy enters k=8 with a body
  (`b_shared`) that has been shaped by 8 tasks of point-goal reaching
  but no handle interaction. That is an intrinsic curriculum mismatch,
  not a sample-efficiency issue.
- **Steps (helpful, but not the bottleneck).** At 8M env steps per
  task we are inside the regime where the C0 success on k=8 has
  *already* plateaued near 0.10 with sporadic spikes. Doubling steps
  to 16M would let us read whether the spikes consolidate into a real
  policy, but my prior is "no, because the body is locked." This is
  testable cheaply via the C2b ablation cell (see below): if C2b
  (dyn-aux off after task 0) and C2 give the same task-8 curve, the
  bottleneck is the body, not the per-task auxiliary.
- **Parallelism (not relevant here).** More parallel envs reduce
  estimator variance and accelerate wall-clock, but cannot change the
  asymptote on an intrinsically hard task with a frozen-ish body.

So the recommended read is: **wait for C2's k=8 to reach the same
evaluator-step count as C0, then re-plot.** If C2 still plateaus near
0.10, k=8 is body-bound and we should add a small-data fine-tune
phase on `phi_task` *only* before re-evaluating, not increase the
horizon. If C2 climbs past 0.30 sustained, we have a real signal that
the decomposed critic is opening up the contact-dynamics direction
that the CKA mixture closed.

---

## Implications for what to launch next

- The ckpt-path fix (commit pending this turn) is now in. Decomposed
  checkpoints from different `(dyn_aux_weight, phi_task_width,
  phi_task_depth)` sweeps no longer collide.
- The **C2b** ablation cell (`dyn_aux_after_task0=0.0`) is staged in
  `experiment_configs.py` but commented out. Uncomment when ready to
  test the "task-0 initialiser only" hypothesis from
  `docs/2026-05-14_c2_ldyn_interpretation.md`. The new flag has been
  wired through the runner and all four DRAFT scripts.
- Do **not** redraw the C2-vs-C0 task-8 comparison until C2's k=8 has
  at least ~100 evaluator-step rows on W&B. The current 25-step
  snapshot is in `task8_success_c0_vs_c2.png` for the record only.

---

**Files referenced**
- `docs/wandb_analysis/csv/h1_h2_alpha_vs_success.csv`
- `docs/wandb_analysis/png/h1_h2_alpha_vs_success.png`
- `docs/wandb_analysis/png/h3_trajectories.png`
- `docs/wandb_analysis/png/c0_trajectories.png`
- `docs/wandb_analysis/png/c2_ldyn_trajectories.png`
- `docs/wandb_analysis/png/task8_success_c0_vs_c2.png`
- `docs/2026-05-14_c1_crash_and_ckpt_collision.md`
- `docs/2026-05-14_c2_ldyn_interpretation.md`
- `docs/2026-05-14_h1_h2_alpha_predicts_failure.md`
- `docs/2026-05-14_h3_logsumexp_test.md`
- `contrastive/decomposed_networks.py` (sa_repr definition)
