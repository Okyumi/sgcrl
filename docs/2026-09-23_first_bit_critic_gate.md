# First-sparse-bit vs critic-flatness Success-BC

Date: 2026-09-23  
Status: first-bit cells `18106553_[0-2]` still running. Cell 3 died on H200
`libEGL` and was retried with the replacement gate cells as SLURM
`18107080` (`--array=3-7`). Invalid `a_succ`-vs-π gate cells
`18106553_[4-7]` were cancelled.

## Motivation

Whole-episode Success-BC retains latch Tasks 5/8 and stalls sequential
Tasks 4/7. Current-sparse (`r_t=1`) still clones linger after the first
hit. We need one paper-legal regularizer that copies successful
**actions**, not successful episodes, using only information available
while interacting: observations, the agent's actions, the sparse 0/1
bit, and queries of the current critic / policy.

## Fix 1: first sparse bit

Only the env 0/1 bit. For a replay episode, insert the first transition
with \(r_t=1\). Later 1-bits stay out of \(\mathcal{D}_{\mathrm{succ}}\).
That stored \((s,a)\) is past experience we actually took, not an oracle
ranking.

## Why the `a_succ` vs \(\pi\) gate was invalid

DCC never observes a privileged "correct action" at a state. After a
sparse hit we may **store** that \((s,a)\) and clone it (replay of our
own behaviour). Comparing

\[
\mathrm{score}(s,a_{\mathrm{succ}},g)
\quad\text{vs}\quad
\mathrm{score}(s,a_\pi,g)
\]

uses \(a_{\mathrm{succ}}\) as a ranking label: "has the critic already
figured out the hindsight-correct action?" That is extra information
we would not have from interaction plus critic queries alone. Cells
4–7 of job `18106553` ran that rule and were cancelled.

## Fix 2: clone only where the critic cannot rank actions

Available at an observed success state \((s,g)\) from \(\mathcal{D}_{\mathrm{succ}}\):

- the current critic \(\mathrm{score}(s,a,g)=\varphi(s,a)^\top\psi(g)\)
- any action we choose to query (here 8 i.i.d. draws from the action box)

Not used for the gate: \(a_{\mathrm{succ}}\). It remains the NLL target
only.

Let \(\sigma_a(s)\) be the std of those 8 scores (action variation at
fixed \(s,g\)) and \(\sigma_s\) the std across the BC batch of the
action-averaged score (state variation, no spatial hover). Then

\[
\mathrm{gate}(s)
=\frac{\sigma_s}{\sigma_s+\sigma_a(s)}
\quad
(\text{both near 0} \Rightarrow 1).
\]

Stop-gradient on the gate. NLL is \(\mathrm{mean}(\mathrm{gate}\cdot(-\log\pi(a_{\mathrm{succ}}\mid s,g)))\).

- Critic flat in \(a\) (latch / confused ranking): \(\sigma_a\) small → clone.
- Critic already ranks actions (place/pull): \(\sigma_a\) large → skip, DCC trains the actor.
- No 0.09 hover, no task list, no stored action as Q-label.

This is the same scientific criterion as the old \(\mathrm{std}_a/\mathrm{std}_s\) diagnostic, measured only from critic queries at observed success states, not from mechanism geometry.

Logged: `retention/bc_critic_gate_mean`, `retention/bc_critic_sigma_a`,
`retention/bc_critic_sigma_s`.

## Launch

First-bit (still `18106553`):

| cell | method | task | budget |
|---|---|---|---|
| 0 | first bit | 4 stick_pull | 8M from paper BC `task_3.pkl` |
| 1 | first bit | 7 shelf_place | 8M from paper BC `task_6.pkl` |
| 2 | first bit | 5 handle_press_side | 1M from scratch |
| 3 | first bit | 8 window_close | 1M from scratch |

Flatness gate (resubmitted `--array=4-7`):

| cell | method | task | budget |
|---|---|---|---|
| 4 | critic flatness | 4 stick_pull | 8M from paper BC `task_3.pkl` |
| 5 | critic flatness | 7 shelf_place | 8M from paper BC `task_6.pkl` |
| 6 | critic flatness | 5 handle_press_side | 1M from scratch |
| 7 | critic flatness | 8 window_close | 1M from scratch |

W&B `nyuad_mmvc/continual_gcrl_paper` /
`PAPER-DCC-FIRSTBIT-CRITICGATE-4758`. Seed 6. Paper stack. Task 7 uses
the pinned shelf.

```bash
sbatch --array=3-7 DRAFT_jubail_first_bit_critic_gate.sh
```

Replacement job `18107080`.

## Validation

```bash
python tests/test_first_bit_critic_gate.py
```

## Limitations

Seed 6 only. Tasks 4/7 resume the paper Success-BC critic, not plain
DCC. Fix 2 still inserts linger; a linger state that is also
action-flat still clones. The two methods are not crossed.
The action box is `Unif[-1,1]^d` (Sawyer). \(\sigma_s\) is batch-relative,
not a calibrated physical unit.
