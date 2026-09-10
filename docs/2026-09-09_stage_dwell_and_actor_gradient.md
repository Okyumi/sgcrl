# Specifics: stage geometry, inject mass, and actor update path

Date: 2026-09-09
Status: analysis + measurement job `17270305` (dwell JSON pending)

## 1. Horizon and stage definitions (exact)

Training horizon **T = 150** steps for every Sawyer task in this repo
(`env_utils.py` sets `max_episode_steps = 150`).

### Handle press side (`sawyer_handle_press_side`)

Env sparse success (corrected):

\[
r_t = \mathbf{1}\big[|z_{\text{handle}} - 0.07| \le 0.02\big]
\]

No hand-near requirement for the **reward**. Handle joint is a vertical
slide `range=[-0.1, 0]`, `damping=30`, no spring in the XML — once pressed
down it tends to **stay** down.

Geometric stages we use for paper numbers (same thresholds as HER logger):

| Stage | Predicate | Meaning |
|---|---|---|
| `far` | \(\|hand-handle\| > 0.15\) | approach from afar |
| `near_approach` | \(0.09 < \|hand-handle\| \le 0.15\) | closing in |
| `hover_contact` | \(\|hand-handle\| \le 0.09\) and axis **not** ok | touching but not pressed |
| `success` (geometric) | \(\|hand-handle\| \le 0.09\) **and** axis ok | pressing while still near |
| `object_progress` | **does not exist** | no continuous band |

**Critical asymmetry vs push:** HER classifies a future as `success` only if
**hand is still within 0.09 of the pressed handle**. A state where the handle
is down but the hand has left is **not** HER-success (it is `far` / `other`).

### Push (`sawyer_push`)

Env sparse success:

\[
r_t = \mathbf{1}\big[\|obj - (0.02,0.89,0.02)\|_2 \le 0.05\big]
\]

| Stage | Predicate |
|---|---|
| `far` | \(\|hand-obj\| > 0.15\) and obj not in progress/success |
| `near_approach` | \(0.09 < \|hand-obj\| \le 0.15\) |
| `hover_contact` | \(\|hand-obj\| \le 0.09\), obj farther than 0.15 from target |
| `object_progress` | \(0.05 < \|obj-target\| \le 0.15\) |
| `success` | \(\|obj-target\| \le 0.05\) (**hand location irrelevant**) |

So on push, once the cube arrives, **every remaining step of that episode**
is an HER-success future. On handle, only the **near + pressed** window is.

This is the concrete meaning of “no progress ladder” on Task 5:
not a vague metaphor — push has a 10 cm radial band (`0.05–0.15`) of
`object_progress` futures; handle has **zero** such band.

Exact per-stage step counts on successful policies: measurement job
`17270305` writing
`logs/task58_success_propagation/dwell_{handle,push}_*.json`.

## 2. What D4 actually injected (and why “shouldn’t inject help?”)

### What was written
- Full **eval** episodes rolled with the **deterministic eval actor**
- Written raw into Reverb via `EpisodeAdder` (obs = `state‖desired_goal`,
  env sparse reward, no Success-BC, no goal rewrite)
- Handle inject seed6 @ 200 400 steps: **n=300 transitions = 2 episodes**
  (150×2), after 6 attempts
- Handle inject seed5 @ 751 500: **n=150 = 1 episode** (40 attempts; hard to
  re-find success)

### Buffer mass math (why inject looked like a no-op)

At inject time ≈ 200 000 env steps:

| Quantity | Value |
|---|---|
| Transitions in replay (approx.) | ≈ **200 000** |
| Injected | **300** |
| Injected fraction | **300/200 000 = 0.15%** |
| Push’s healthy HER success mass | **~20–40%** ≈ 40 000–80 000 transitions |
| Shortfall vs push operating point | **~100–250× too little mass** |

Then the next 50 k env steps add ~50 k mostly-fail transitions and dilute
further. HER-phase EMA uses decay **0.99**, so 300 inserts barely move the
logged fraction (observed: HER succ stayed **0.07 → 0.03**).

So: **your intuition is correct** — if the only problem were “actor stopped,
buffer empty,” a *mass-matched* refill to ~20–40% success should change
behavior. Our D4 did **not** test that. It tested “add 0.15% success mass,”
which cannot match push’s training distribution.

**Next experiment (required):** on Task 5 at first ≥20% spike, inject until
HER geometric success mass reaches **0.20** (or inject **N ≈ 0.25 × buffer
size**, repeating successful episodes / cloning), keep `success_bc_weight=0`,
compare to measure twin.

## 3. Causal story, stated carefully

What we **can** say:

1. Critic under \(g_{\text{task}}\) usually ranks success states/episodes
   above hover (probe gaps large positive when both exist).
2. Actor **stops regenerating** successful rollouts after brief discovery
   (eval collapses; later probes often `n_success=0`).
3. Therefore HER success mass never rises on Task 5 (~2–5%).
4. (2)+(3) are consistent, but (3) is also structurally harder on handle
   because HER-success labels require sustained near-contact, while push
   HER-success persists after the object arrives.

What we **cannot** yet say: “actor stops *only* because buffer is empty.”
D4 was too weak to test that. Actor-side optimization under HER goals is
still an open failure mode.

## 4. Actor network / gradient path (what to track)

Actor loss (`continual_learning_decomposed.actor_loss_fn`):

\[
a \sim \pi_\theta(\cdot \mid s, g),\qquad
L_{\text{actor}} = -\phi(s,a)^\top\psi(g) - \alpha\,H[\pi]
\]

- Default **`actor_goal_mode=her`**: \(g\) is a **HER future**, not \(g_{\text{task}}\).
- Sampling is `distribution.sample` (reparameterized Normal–Tanh) → gradients
  flow through \(a\) into \(\phi(s,a)^\top\psi(g)\).
- Critic params are **not** updated by this loss (`argnums=0` = policy only).
  Critic can be “right” on a probe while the actor is optimized for a
  different goal distribution.

Implication for Task 5:

- With ~97% hover/far HER futures, almost every actor step is
  “match \(\psi(g_{\text{hover/far}})\)”.
- Probe statement “critic prefers success under \(g_{\text{task}}\)” is about a
  **different conditioning** than the actor’s training goal.
- We already tried `actor_goal_mode=task` (v3) and it still collapsed — so
  wrong goal alone is not the full story. Remaining suspects:
  1. Batch **states** are still mostly approach/hover (no success states to
     reinforce press from).
  2. At hover states, the local score landscape under \(g_{\text{task}}\) does
     not make press actions a clear ascent direction for \(\pi\).
  3. HER geometric success on handle is a thin contact set, so even rare
     successes contribute few HER-success futures per episode.

### Actor diagnostic to run next (on peak vs late ckpts)

At hover states from rollouts, log:

| Metric | Meaning |
|---|---|
| \(\phi(s,a_\pi)^\top\psi(g_{\text{task}})\) | what π scores under task goal |
| \(\phi(s,a_{\text{press}})^\top\psi(g_{\text{task}})\) | press-biased action under task goal |
| same two under \(g_{\text{hover}}\) | training-like goal |
| \(\|\nabla_a (\phi^\top\psi)\|_{a=a_\pi}\) direction vs press | does critic landscape point to press? |
| \(\log\pi(a_{\text{press}}\mid s,g)\) | does π put mass on press? |

Script `scripts/measure_success_stage_dwell.py` already logs
`hover_score_gap_biased_minus_pi` under \(g_{\text{task}}\) (pending job).

## 5. Paper-ready one-paragraph claim (after dwell JSON lands)

Fill in `N_*` from `dwell_*.json`:

> Under a 150-step horizon, successful push episodes spend on average
> \(N_{\text{prog}}\) steps in `object_progress` and \(N_{\text{succ}}\) steps
> in geometric success (object within 5 cm of target; hand free). Successful
> handle-press episodes have **no** progress band and only \(N_{\text{hov}}\)
> hover-contact plus \(N_{\text{hs}}\) near+pressed steps that HER counts as
> success; handle-down with hand away is **not** an HER-success future.
> After discovery, Task 5 HER success mass stays 2–5% while push
> success|progress rises to 60–80%. A 300-transition inject (0.15% of a
> 200 k buffer) does not change this; a mass-matched inject and actor
> landscape probes are required to separate data scarcity from actor
> optimization failure.
