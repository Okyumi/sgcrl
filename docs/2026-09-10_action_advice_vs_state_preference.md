# Same-state action advice vs success-state preference

Date: 2026-09-10
Status: Torch probe blocked by QOS; Jubail 1-seed train+probe submitted
(`DRAFT_jubail_task5_action_advice.sh`, see
`docs/2026-09-10_jubail_task5_action_advice.md`). Analysis below
uses already-finished dwell JSONs + retention v3.

## Motivation

W&B / phase probes showed the critic ranks **finished success states** above
**hover states** under \(g_{\text{task}}\) (`probe/gap_success_minus_hover > 0`).
That does **not** imply the critic gives useful **action** advice at a fixed
hover state, nor that policy gradient under HER goals will reinforce press.

## Why the implication fails (math)

Critic score:

\[
s(s,a,g)=\phi(s,a)^\top\psi(g).
\]

State-ranking evidence is about **different** \((s,a)\) pairs:

\[
\mathbb{E}\big[s(s_{\text{succ}},a_\pi(s_{\text{succ}}),g_{\text{task}})\big]
\;>\;
\mathbb{E}\big[s(s_{\text{hov}},a_\pi(s_{\text{hov}}),g_{\text{task}})\big].
\]

Actor PG (schematic) at a hover training state maximizes
\(s(s_{\text{hov}},a_\theta,g_{\text{train}})\) with \(g_{\text{train}}\)
usually a **HER future**, not \(g_{\text{task}}\). Needed for press:

\[
s(s_{\text{hov}},a_{\text{press}},g)
\;>\;
s(s_{\text{hov}},a_\pi,g)
\quad\text{and}\quad
\nabla_a s\big|_{a_\pi}\ \text{points toward press}.
\]

Neither follows from ranking finished success *states* above hover *states*:

1. **State vs action.** High score on success can be carried by \(\phi\)’s
   dependence on \(s\) (handle already down), not by preferring press at hover.
2. **Action-insensitive critic.** If \(\mathrm{std}_a\,s(s,a,g)\ll
   \mathrm{std}_s\,s(s,a_\pi,g)\), the landscape is mostly scene recognition.
3. **Wrong \(g\) in the actor loss.** Even if press wins under \(g_{\text{task}}\),
   under \(g\approx\) hover/far HER futures the ranking can flip.
4. **Local gradient ≠ global argmax.** Argmax over discrete candidates ≠
   reparameterized PG follows that direction from \(a_\pi\).

## Reconciliation with earlier conclusions (not a sudden flip)

| Earlier claim | Status | How it fits |
|---|---|---|
| Hover ≈ success under \(g_{\text{task}}\) (fake goal) | **Falsified** as *state* ranking | Keep falsified |
| Actor PG / success-score climb is a weak fix; Success-BC works | **Still true** (retention v3) | Compatible with bad *local* action advice or wrong \(g\) |
| OOD HER goals after collapse | **Still true** as the *loop after* failure | Explains lock-in, not first discovery |
| “Success contact too short” | **Overstated for successful eps** | Peak handle success eps ≈ 75/150 geometric success steps; problem is *stopping producing* them |
| No progress band on handle | **Still true** | Push has ~33 progress steps/ep; handle has 0 → HER composition asymmetry |
| Tiny D4 inject falsifies scarcity | **No** — inject was ~0.15% mass | Mass-matched inject still needed |

So the narrative is layered, not replaced: healthy **state** preference under
\(g_{\text{task}}\) can coexist with weak **action** signal at hover and/or
actor updates under HER hover goals; after success disappears, OOD HER fills
the buffer and the actor–data loop dies.

## What existing logs already say (no new run)

From `logs/task58_success_propagation/dwell_*.json` (job `17270305`):

| Checkpoint | hover `score(press)−score(π)` | success steps / 150 (success eps) |
|---|---|---|
| handle peak 100k | **+0.48** | ~75 |
| handle late | +0.79 (few success eps) | ~45 |
| push peak 250k | **+2.65** | ~78 + ~33 progress |

Retention v3: `actor_goal=task|mix` and `actor_success_score` collapse like
control; `success_bc_terminal` retains. That already suggests “maximize critic
on success *states*” ≠ “learn press *actions*.”

## New offline probe (this note)

`scripts/measure_hover_action_advice.py` on handle peak / late + push peak:

- same-state gap under \(g_{\text{task}}\) and under hover-goal \(g=s\)
- press rank among π + press + 32 random actions
- \(\mathrm{std}_a\) vs \(\mathrm{std}_s\) sensitivity ratio
- \(\nabla_a s\) cosine with press direction + local step toward press

Launch: `sbatch DRAFT_hover_action_advice.sh` (1 GPU, ~1h, offline).

## Mass-inject (still separate)

Scarcity vs action-advice are **both** open. Mass-matched 10%/20% inject
(`17270536`) tests scarcity; this offline probe tests action advice without
touching the 90-seed paper queue beyond holding pending jobs so the short
probe can start when the next GPU frees.
