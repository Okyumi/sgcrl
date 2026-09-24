# Task 7: do first Success-BC paths have lower critic score than DCC?

Date: 2026-09-21  
Status: measured. Job `18057214`. JSON
`logs/task7_critic_score_gate/scores.json`.

## Question

A \(K=64\) first-success window is tied to MetaWorld's 150-step horizon.
A more general rule is: insert \((s,a)\) into \(\mathcal{D}_{\mathrm{succ}}\)
when the DCC score \(\phi(s,a)^\top\psi(g_{\mathrm{task}})\) is high
enough. That only filters the first lucky shelf-place if those paths
really score lower than the successes plain DCC finds.

## What was scored

Terminal-BC mid-ckpts store the actual ring. Seed 6 at the first 10%
eval (1.50M) and seed 10 at its first 10% eval (1.10M) each contain
**750 transitions = five full 150-step episodes**. Those are the first
cloned successes, not a later policy.

Paper DCC seed 6 `task_7.pkl` was rolled out for 30 eval episodes
(20 successes). Inner product and cosine are reported only under a
**shared** critic (raw \(\phi^\top\psi\) is not comparable across
checkpoints).

- `seed10_late_critic`: terminal-BC seed 10 at 7.90M (best eval 90%).
  Stand-in for a critic that saw good Task-7 successes. Paper DCC
  finals do not save \(\phi_{\mathrm{task}}\).
- `seed6_early_critic`: the online critic at lock-in (seed 6, 1.50M).

In-goal means object-to-shelf \(\le 0.07\). Prefix-far means \(>0.15\).

## Result

Under the late critic, **DCC in-goal actions score higher than the
first cloned in-goal actions**:

| \((s,a)\) set | in-goal \(\phi^\top\psi\) | all steps |
|---|---:|---:|
| plain DCC seed-6 success rollouts | **−10.8** | −14.3 |
| seed-6 first 5 cloned episodes | −17.1 | −14.5 |
| seed-10 first 5 cloned episodes | −18.5 | −12.7 |

The whole-episode means look similar because 80% of a cloned episode
is prefix. The gap is on the placing states. Cosine matches: DCC
in-goal −0.11 vs first-5 −0.24.

Under the **online** critic at first success, those lucky in-goal
steps are already the **highest** scores in the buffer (+0.16 vs
prefix −7.3). A “Q high enough” gate using the current critic would
still insert the first sloppy place. It would drop the wandering
prefix. That is Q-filtered / in-goal BC, not a bar that rejects the
first completion.

DCC in-goal is already +2.44 under that same early critic, so the
critic *can* rank a better place higher — it just has not seen one
yet when the first luck arrives.

## Implication

A critic-score gate is less env-specific than \(K=64\), and on Task 7
it would likely stop prefix cloning. It would **not** by itself stop
lock-in on the first completion, and on Tasks 5/8 the critic does not
rank the press, so the same gate may insert nothing useful there.
Raising the bar with advantage vs a running baseline (SIL / AWAC) is
the next generalisation if prefix-only filtering is not enough.
