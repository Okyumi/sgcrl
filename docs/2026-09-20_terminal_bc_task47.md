# Terminal-episode Success-BC on Task 4 / Task 7

Date: 2026-09-20  
Status: 10-seed 8M paper cells launched (seeds 5–14).

## Motivation

The 10-seed paper cell used `episode_sparse_reward`: if **any** sparse 1
appeared, the **whole** episode was cloned. That retains Tasks 5/8 but
stalls 4/7, because a mid-episode hit that later undoes still fills
\(\mathcal{D}_{\mathrm{succ}}\).

The intended rule is `terminal_episode`: clone the whole episode **only
if the last transition is still successful**. Tasks 5/8 already work
under this label (seed-6 handle/window video jobs, final 100%). These
runs are the matched 10-seed Task 4 / Task 7 curves for paper AUC and
best success.

Per-step cloning (`current_sparse_reward`) is not used here. On 5/8 the
useful behaviour is approach plus a brief press; most \(r_t=1\) frames
are linger whose actions need not cause success.

## Label

Still the env 0/1 sparse bit.

```
terminal_episode:       Y(τ) = 1[r_{T-1} > 0]; clone every (s_t, a_t) in τ
episode_sparse_reward:  Y(τ) = 1[max_t r_t > 0]; clone every (s_t, a_t) in τ
```

`flatten_fn` reads `sample.data.reward[seq_len - 2]` (last real
transition, the last entry of `reward[:-1]`). If that bit is 1, every
step of the episode is marked for the ring buffer. λ stays 0.1.

Eval is unchanged: any in-episode sparse hit still counts as success.
AUC and best success use the same recipe as Table 1: 10-episode eval
every 100k, 8M budget, seeds 5–14.

## Probe

Resume each paper Success-BC seed's `task_3.pkl` → Task 4 for 8M, and
`task_6.pkl` → Task 7 for 8M. Same paper stack (`native_info`, dyn=0,
1024×4, λ=0.1). Array 0–9 is Task 4, 10–19 is Task 7.

The seed-6-only start (`18033957`) was cancelled in favour of this
matched 20-cell grid.

```bash
sbatch DRAFT_jubail_terminal_bc_task47.sh
```

W&B `PAPER-DCC-TERMINAL-BC-TASK47`.
