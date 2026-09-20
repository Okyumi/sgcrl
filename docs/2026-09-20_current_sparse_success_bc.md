# Per-step Success-BC on Task 4 / Task 7

Date: 2026-09-20  
Status: 8M paper-budget probe launched after cancelling the 2M start.

## Motivation

Paper Success-BC (`episode_sparse_reward`) copies **every** step of an
episode that ever saw a sparse 1. That matches evaluation (any in-episode
hit still counts as success) but stalls Tasks 4/7 by cloning prefixes and
post-success undo. Tasks 5/8 want the opposite: clone the steps that are
**currently** in the success band (the latch / hold).

## Evaluation vs buffer

Eval is unchanged: `SuccessObserver` marks the episode successful if any
sparse reward is ≥ 1, even if the arm later leaves the goal. The new
label only changes **which transitions enter \(\mathcal{D}_{\mathrm{succ}}\)**.

## Label

Still the env 0/1 sparse bit. No extra sites or phase labels.

```
current_sparse_reward:  outcome_task_success_t = 1[r_t > 0]
episode_sparse_reward:  outcome_task_success_t = 1[max_k r_k > 0]   # old
```

λ stays 0.1. On 5/8, dwell steps keep `r=1` and still fill the buffer.
On 4/7, approach (`r=0`) and knock-away (`r=0`) are not cloned.

## Probe

Do not rerun the 10-task curriculum. Resume paper Success-BC seed-6
`task_3.pkl` → train Task 4 for **8M**, and `task_6.pkl` → train Task 7
for **8M**, matching the paper task budget so the curves can go on the
paper. Compare against the paper plain-DCC and episode-wide Success-BC
cells on the same tasks. The 2M start (`18032983`) was cancelled before
any real training.

```bash
sbatch DRAFT_jubail_current_sparse_bc_task47.sh
```

W&B `PAPER-DCC-CURRENT-SPARSE-BC-TASK47`.
