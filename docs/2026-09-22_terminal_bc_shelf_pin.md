# Task 7 terminal Success-BC with the shelf pinned

Date: 2026-09-22  
Status: 10-seed grid submitted on Jubail
(`DRAFT_jubail_terminal_bc_shelf_pin.sh`). Seed 6 is `18071431`; the
other nine are the follow-up array.

## Motivation

The earlier `PAPER-DCC-TERMINAL-BC-TASK47` Task-7 cell cloned last-bit
successes while the shelf mesh was still MetaWorld-randomized. Success
could fire at the commanded \(g=(0.02,0.89,0.30)\) off the physical
deck. `SawyerShelfPlace.reset` now pins the shelf body to that \(g\).
This grid asks whether terminal Success-BC still collapses Task 7 once
the object has to stay on the shelf.

## Method

Same paper DCC + actor regularizer as the Task-47 terminal cell:

- \(\lambda=0.1\), 4096 ring, `terminal_episode` (\(Y(\tau)=1[r_{T-1}>0]\))
- clone the whole 150-step episode under commanded \(g\)
- eval: any in-episode sparse hit
- no \(\lambda\) warmup, no window cut

Resume paper Success-BC `task_6.pkl` for seeds 5–14, train only Task 7
for 8M (`start_task=7`, `num_tasks=8`). Two actors, eval every 100k.

## Launch

```bash
sbatch DRAFT_jubail_terminal_bc_shelf_pin.sh
```

W&B `nyuad_mmvc/continual_gcrl_paper` / `PAPER-DCC-TERMINAL-BC-SHELF-PIN`.
Array 0–9 is seeds 5–14.

Compare Task 7 against:

- plain DCC (no BC)
- old terminal BC (unpinned shelf)
- warmup / first-success-window (also unpinned)

## Logged metrics (22 Sep, ~2–3.5M)

Eight seeds are training; 13 and 14 died on H200 `libEGL`. Matched-n
eval success at 2.0M: pin **20%** (n=8) vs paper plain DCC **39%**.
Partial AUC to 2.0M: **0.09** vs **0.19**. At 2.5M the pin mean is
**34%** (n=7) vs plain **33%**, but three seeds are still at 0% (6, 9,
10) while 5/11/12 have latched. Unpinned terminal BC was ~11% at 2.5M.

## Validation

```bash
python tests/test_shelf_place_fixed_goal_pin.py
python tests/test_terminal_bc_shelf_pin.py
```

## Limitations

The predecessor critic is still the paper Success-BC Task-6 critic, not
plain DCC. Seed 6 was started as a one-cell job before the array was
widened; do not submit index 1 again while that job is running. Seeds
13–14 need a non-H200 rerun. Numbers above are mid-run, not 8M paper
AUC.
