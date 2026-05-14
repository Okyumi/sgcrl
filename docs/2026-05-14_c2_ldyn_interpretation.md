# Is the dynamics auxiliary actually doing work? L_dyn analysis (C2)

Date: 2026-05-14

User question: "L_dyn is very low the whole time. Do you think it is
truly beneficial?"

**Short answer: the dynamics auxiliary is doing real work only at
task 0. After that it provides essentially zero gradient. It is
acting as a task-0 initializer for `b_shared`, not as a continual-
learning regularizer.** Whether that is good or bad for the paper
is discussed at the end.

---

## What the data shows

Pulled `learner/decomp/L_dyn` for each (task, seed) C2 run.

| Task | env                      | L_dyn initial (mean) | L_dyn final (mean) | Drop ratio |
|------|--------------------------|----------------------|--------------------|------------|
| k=0  | sawyer_hammer            | **4.0e-4**           | 1.3e-4             | **3.14×**  |
| k=1  | sawyer_push_wall         | 9.4e-5               | 1.1e-4             | 0.77×      |
| k=2  | sawyer_faucet_close      | 9.1e-5               | 1.4e-4             | 0.62×      |
| k=3  | sawyer_push_back         | 8.6e-5               | 7.7e-5             | 1.12×      |
| k=4  | sawyer_stick_pull        | 9.0e-5               | 1.1e-4             | 0.83×      |
| k=5  | sawyer_handle_press_side | 8.6e-5               | 7.5e-5             | 1.22×      |
| k=6  | sawyer_push              | 1.6e-4               | 1.0e-4             | 1.57×      |
| k=7  | sawyer_shelf_place       | 8.4e-5               | 1.1e-4             | 0.73×      |

See `docs/wandb_analysis/png/c2_ldyn_trajectories.png` for the
log-scale trajectory plot. Task 0 starts at 8e-4 and falls to 1e-4
within ~100 SGD steps. Every subsequent task starts at and stays
around 1e-4 the entire 500 SGD steps — flat lines.

CSV: `docs/wandb_analysis/csv/c2_ldyn_per_task.csv`.

## Why this happens (mechanically)

`STABLE_INDICES = (0, 1, 2, 3)` selects end-effector xyz + gripper
from the unified 11-dim state. These four scalars are determined
by **robot kinematics**, which is invariant across the 10
Meta-World Sawyer tasks (same robot, same kinematics, only the
manipulation target changes). The dynamics auxiliary therefore
trains `h_dyn ∘ b_shared` to predict
`next_state[:4] = f(state[:4], action)` for the **shared embodiment
dynamics**.

Once this function is learned, the dynamics target for any
subsequent Sawyer task is **the same**. Training on a new task does
not produce a new training signal for L_dyn. The body's
representation of EE+gripper dynamics is task-agnostic by
construction, which is exactly the property the plan section 6
wanted — but it means the auxiliary's gradient effectively goes to
zero after task 0.

## What this means for the algorithm

The plan's original framing was:

> The dynamics auxiliary keeps `b_shared` task-agnostic across
> tasks by constantly pulling its features toward the masked
> stable-state regression target.

The data says this framing is partially wrong. The auxiliary
**pulls features toward dynamics during task 0**, then **does
essentially nothing** during tasks 1-9 (because the loss is
already near its minimum). The body is **not being continually
pulled** by the auxiliary; it is being **left alone** with
whatever it learned at task 0.

This means:

1. The body's continual-time-step behavior on tasks 1-9 is shaped
   **only by the InfoNCE gradient**, not by the dynamics aux.
2. The auxiliary's contribution is to **set the initial
   conditions** for `b_shared` at the end of task 0. If those
   initial conditions are good enough, the body remains usable
   for all 10 tasks even though InfoNCE-only training would, in
   principle, be free to overwrite them.
3. Whether or not the body **actually drifts away** from
   dynamics-aware features during tasks 1-9 is an empirical
   question — the auxiliary is not preventing such drift after
   task 0, it's just providing a good starting point.

## Is this benefit, or just regularizer-as-init?

Three possibilities:

**A. The aux is genuinely useful** because it produces a `b_shared`
that, once initialised, is robust to InfoNCE-only training. In
this case, the C2 algorithm works because of a one-shot
representation that happens to be self-stabilizing under
contrastive gradients. The aux is real but it is a **one-shot
regularizer**, not a continual one.

**B. The aux is incidental** to the C2-vs-C0 performance gap. The
body would behave identically if the aux were turned off after task
0 (because L_dyn is at its floor and provides no signal anyway).
The C2 gain over C0 comes from the **architecture** (separate
`b_shared` + `phi_task` + `psi` heads), not from the auxiliary.

**C. The aux is silently harmful.** The aux's near-zero gradient
imposes a small but non-zero drag on `b_shared`'s ability to fit
the InfoNCE loss. C2 would be *better* if the aux were turned off
after task 0.

The C2 vs C0 data (best-during-training success: C2 matches or
beats C0 on every task with data) is consistent with A and with
B, but does not distinguish them. To distinguish, we need an
ablation: **run C2 with `dyn_aux_weight=0` on tasks 1-9 only
(keep it on at task 0)**.

This is a single new cluster cell. Cost: 1 run × 3 seeds × 8M
steps/task = same as C2. Worth running if you want a clean answer.

## Refined plan-section-6 language for the paper

Original: "the dynamics auxiliary keeps b_shared task-agnostic
across the task sequence."

Suggested: "the dynamics auxiliary constrains b_shared to encode
shared-embodiment dynamics during the base task. The learned
representation generalises across subsequent tasks because the
masked stable-state dynamics are invariant across the 10
Sawyer tasks (same robot, same EE+gripper kinematics)."

This is more accurate to the data and avoids the "continual
regularizer" claim that the trajectory does not support.

## Test predictions

If interpretation A is right (aux is useful as a one-shot
regularizer):

- Running C2 with `dyn_aux_weight=1.0` only on task 0 and 0.0
  afterward should produce essentially the same C2 result
  (same success rates, same forgetting curves).
- Running C2 with `dyn_aux_weight=0.0` throughout (no aux at all)
  should produce **worse** results, especially on later tasks
  where b_shared has drifted.

If interpretation B is right (aux is incidental):

- Both of the above should produce the same result as C2.

If interpretation C is right (aux is harmful):

- Turning the aux off after task 0 should produce slightly
  **better** results than C2.

## A separate signal worth checking now: linear probe across tasks

We have D6's linear-probe diagnostic (`docs/2026-05-08_d6_linear_probe.md`).
Original purpose: check whether `b_shared` absorbs task identity. We
should run it on a task-0 checkpoint AND a task-7 checkpoint and
compare. If task-0 `b_shared` has near-chance probe accuracy and
task-7 `b_shared` has high probe accuracy, then the body IS
drifting away from dynamics-aware features as InfoNCE-only training
proceeds — confirming that the aux is not providing a continual
constraint. **This is a free analysis once C2 task-8/9 reruns are
in and we have all 10 checkpoints.**

## Files produced

- `docs/wandb_analysis/csv/c2_ldyn_per_task.csv` — per-(task,
  seed) L_dyn initial / final / max / min.
- `docs/wandb_analysis/png/c2_ldyn_trajectories.png` — log-scale
  per-task L_dyn over 500 SGD steps, 3 seeds median ± IQR.
- `docs/wandb_analysis/c2_ldyn_analysis.log` — raw extraction log.
