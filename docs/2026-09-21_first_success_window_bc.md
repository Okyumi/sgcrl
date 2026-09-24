# First-success window Success-BC

Date: 2026-09-21  
Status: submitted on Jubail SLURM `18046525` (`DRAFT_jubail_first_success_window_bc.sh`)

## Motivation

Episode-wide and terminal Success-BC keep Tasks 5/8 but freeze Tasks 4/7
on the first lucky success. The actor clones every step of that episode
under \(g_{\mathrm{task}}\), so \(\lambda=0.1\) NLL can match the DCC
update and the policy never improves past that first trajectory.

The failure is prefix cloning, not the 0/1 sparse bit. Tasks 5/8 still
need the approach before contact. Tasks 4/7 need DCC to keep shaping the
long reach / grasp, and must not imitate 150 steps of a sloppy place or
an undone pull.

## Rule

Same \(\lambda=0.1\), same 4096 ring, still only the env sparse bit.
An episode enters \(\mathcal{D}_{\mathrm{succ}}\) iff the **last**
transition has \(r>0\). The inserted steps are only

\[
\{(s_t,a_t)\}_{t=t^\star-K+1}^{t^\star},
\quad t^\star=\min\{t:r_t>0\}, \quad K=64.
\]

- Stick-pull undo: last bit is 0, so the episode is dropped.
- Shelf-place linger: object stays in the 0.07 ball, but steps after
  \(t^\star\) are not cloned.
- Handle / window: \(t^\star\) is the press / latch; the window is the
  approach plus that contact, not the later hold.

Eval is unchanged (any in-episode sparse hit still counts).

## Code

- `contrastive/success_bc_labels.py`: numpy + TF mask.
- `run_continual_contrastive.py` flatten / in-trajectory flatten.
- `--success_bc_label_mode=first_success_window --success_bc_window=64`.

## Probe

Seed 6, two actors, W&B `continual_gcrl_paper` /
`PAPER-DCC-FIRST-SUCCESS-WINDOW`.

| cell | task | budget | start |
|---|---|---|---|
| 0 | 4 stick_pull | 8M | paper BC `task_3.pkl` |
| 1 | 7 shelf_place | 8M | paper BC `task_6.pkl` |
| 2 | 5 handle_press_side | 1M | from scratch, diagnosis stack |
| 3 | 8 window_close | 1M | from scratch, diagnosis stack |

```bash
sbatch DRAFT_jubail_first_success_window_bc.sh
```

Decision: keep this as the paper method only if Task 5/8 still retain
and Task 4/7 stay near plain DCC (or recover past terminal BC). If 5/8
miss the approach, raise \(K\). If 7 still locks, the next knob is a
self-imitation bar on time-to-first-success, not a smaller \(\lambda\).

## Limitations

- \(K=64\) is a guess (\(\approx 40\%\) of a 150-step episode).
- Tasks 4/7 still resume from the episode-wide BC predecessors, so the
  critic is not the plain-DCC critic.
- One seed. 10-seed paper numbers wait on this probe.
