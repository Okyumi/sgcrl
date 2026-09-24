# Task 7 policy GIFs and elite success buffer

Date: 2026-09-21  
Status: GIF export submitted as SLURM `18057318`
(`DRAFT_jubail_task7_policy_gifs.sh`).

## Prefix vs Tasks 5/8

Dropping **the whole prefix** (only clone \(r_t=1\) / high-\(Q\) in-goal
steps) is the thing that would hurt handle-press and window-close. Those
tasks need the approach that *causes* the brief contact. A critic-score
gate that throws away low-\(Q\) steps is the same family.

The lock-in on shelf-place is cloning a **bad full episode**, not
keeping an approach. The 5/8-safe version is: keep a full trajectory,
but only of the **best** successes so far.

## Elite buffer (proposed)

Sparse 0/1 has no denser return, so “most successful” means **shortest
time-to-first-success** (optionally still requiring the last bit to be
1). Periodically rebuild \(\mathcal{D}_{\mathrm{succ}}\) from that
elite set instead of FIFO:

1. Score each finished successful episode by \(t^\star=\min\{t:r_t>0\}\).
2. Keep the shortest \(N\) (or replace a stored episode iff the new one
   is strictly faster).
3. Clone **every** step of those elite episodes, including the approach.

On 5/8, a clean press that arrives earlier replaces a slow lucky press;
the approach is still cloned. On 7, a 40-step place replaces the first
140-step wander. This is self-imitation with a raising bar, not a
MetaWorld-specific window \(K\).

## GIFs

Seed 6, `native_info`, 320px, ≤80 frames.

| cell | policy | source | GIFs |
|---|---|---|---|
| 0 `shelf_plain` | plain DCC | paper `task_7.pkl` | 10 eval rollouts + first-success hunt |
| 1 `shelf_bc` | episode-wide Success-BC | paper `task_7.pkl` | same |
| 2 `shelf_terminal` | terminal-episode Success-BC | 8M mid-ckpts | 10 paced steps (0.8M…8M) + hunt at first 10% eval (~1.5M) |

Paper plain / episode-wide runs did not save mid-task snapshots, so
cells 0–1 are **final** policies. “Throughout training” is only cell 2.
The first-success GIF is a successful **eval** episode at that
checkpoint, not the original training frames.

```bash
sbatch DRAFT_jubail_task7_policy_gifs.sh
```

Output: `logs/jubail_task7_policy_gifs/gifs/{shelf_plain,shelf_bc,shelf_terminal}/`
