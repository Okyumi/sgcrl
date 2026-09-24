# One Success-BC rule for Tasks 4/7 and 5/8

Date: 2026-09-20  
Status: superseded. Task 4/7 redo uses `terminal_episode` at 8M
(`DRAFT_jubail_terminal_bc_task47.sh`). The 1M mixed grid was cancelled.

## Motivation

Plain DCC fails to retain Tasks 5 and 8. Episode-wide Success-BC
(`episode_sparse_reward`, λ=0.1) fixes those latch tasks but stalls
Tasks 4 and 7 by cloning prefixes and post-success undo. Tuning λ only
trades one failure for the other. The goal is **one** regularizer, using
only the env 0/1 sparse bit, that helps 5/8 and does not hurt 4/7.

## Do not start from λ

λ=0.1 is a weight on tanh-normal NLL. Lowering it can reduce 4/7 damage
and 5/8 retention together. The label is the thing that is task-property
dependent; λ is not.

## Two candidates (same λ=0.1, same ring buffer)

`terminal_episode`: clone the **whole** episode iff the **last** sparse
bit is 1. Matches the draft's written \(Y(\tau)\). Already retains 5/8
(handle/window video jobs, seed-6 finals 100%). Should drop 4/7 episodes
that succeed then undo. Still clones prefixes of episodes that finish
successful.

`current_sparse_reward`: clone only steps with \(r_t=1\). Drops 4/7
approach and undo. On 5/8 it keeps contact/hold, not the approach. The
8M Task 4/7 jobs (`18033003`) test whether this protects 4/7; cells 2–3
below test whether it still retains 5/8.

## Probe

1M, seed 6, no extra task information.

| cell | task | label | stack |
|---|---|---|---|
| 0 | 4 stick_pull | terminal_episode | paper, resume BC `task_3.pkl` |
| 1 | 7 shelf_place | terminal_episode | paper, resume BC `task_6.pkl` |
| 2 | 5 handle_press_side | current_sparse_reward | diagnosis, from scratch |
| 3 | 8 window_close | current_sparse_reward | diagnosis, from scratch |

1M is the window where episode-wide BC already collapsed Task 4
(~0.07 vs plain ~0.30). Tasks 5/8 diagnosis matches the existing
terminal-episode Success-BC cells.

```bash
sbatch DRAFT_jubail_universal_success_bc.sh
```

W&B `PAPER-DCC-UNIVERSAL-SUCCESS-BC`.

## Decision rule

- If terminal 4/7 stays near plain DCC **and** we already know terminal
  5/8 works: **use `terminal_episode` as the paper method** (and fix
  Section 2 to per-step env reward, terminal \(Y(\tau)\) for \(\mathcal{D}_{\mathrm{succ}}\)).
- If terminal still hurts 4/7, but current-sparse 5/8 retains: **use
  `current_sparse_reward`**.
- If neither works on both pairs, λ-only is still just a tradeoff; next
  would be cloning only the first success transition, still from the
  same 0/1 bit.

The 8M current-sparse Task 4/7 jobs remain the paper-budget check for
that candidate.
