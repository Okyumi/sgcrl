# Per-step Success-BC on Tasks 4/7 and 5/8

Date: 2026-09-22  
Status: submitted on Jubail (`DRAFT_jubail_current_sparse_47_58.sh`).

## Motivation

Whole-episode Success-BC retains latch tasks 5/8 and stalls 4/7.
The candidate unified rule uses only the env sparse 0/1 bit:

\[
\mathcal{D}_{\mathrm{succ}} = \{(s_t,a_t): \text{sparse bit}_t = 1\}.
\]

Same \(\lambda=0.1\), same 4096 ring, same NLL. Eval is still any
in-episode hit.

On 5/8 the **first** 1-bit of an episode is the press/close. Later 1-bits
are hold after the mechanism is already down; they do not need to “do”
anything, but they also should not erase the press \((s,a)\) that is
already in the ring. On 4/7, 1-bits are only in-goal frames, so the
path is left to DCC.

If 4/7 stay near plain DCC and 5/8 retain, the paper method is this
rule, not whole-episode cloning. If 5/8 collapse, linger mass drowned
the press and whole-episode BC stays a 5/8-only ablation.

## Launch

```bash
sbatch DRAFT_jubail_current_sparse_47_58.sh
```

W&B `nyuad_mmvc/continual_gcrl_paper` / `PAPER-DCC-CURRENT-SPARSE-4758`.
Seed 6. Array 0–3: Task 4 8M, Task 7 8M, Task 5 1M, Task 8 1M.
Paper stack (`native_info`, dyn=0, 1024×4). Task 7 uses the pinned shelf.

## Validation

```bash
python tests/test_current_sparse_47_58.py
```

## Paper if it holds

DCC is the learner. The regularizer copies actions at sparse-success
states, not successful episodes. On a latch, the first such state is
the press the critic cannot rank. On a place/pull, those states are
only the goal slice.

## Limitations

Seed 6 only. Task 4/7 resume the paper Success-BC critic, not plain DCC.
The earlier T4/T7-only current-sparse job (`18033003`) died before it
could answer this.
