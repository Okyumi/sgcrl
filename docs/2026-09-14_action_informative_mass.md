# Action-informative mass diagnostic (ICLR DCC figure)

Date: 2026-09-14
Status: job `17954891` COMPLETED (1h10m). Figure populated from matched
seed-6 DCC checkpoints. **Paper text not edited.**

## Motivation

Discussion / Appendix H hypothesis:

> Contrastive RL has difficulty learning useful action rankings when
> states at which changing the action meaningfully changes the future
> occupy little mass under the replay distribution.

Theoretical object:

\[
\mathcal{S}_{\mathrm{act}}^{\varepsilon}
=
\{s:\sup_{a,a'}D_{\mathrm{TV}}[p(F\mid s,a),p(F\mid s,a')]>\varepsilon\},
\qquad
\delta_{\varepsilon}=\mu(\mathcal{S}_{\mathrm{act}}^{\varepsilon}).
\]

We **cannot** claim to measure \(D_{\mathrm{TV}}\). This note reports
(i) a geometric occupancy proxy (hand within 9 cm of the object),
(ii) a **scale-normalized** critic-score proxy
\(\Delta_f/\sigma_s\), and (iii) a short-horizon environment-branch
proxy on a task coordinate. None of these is \(D_{\mathrm{TV}}\).

## Tasks and checkpoints

From `contrastive.continual_config.CONTINUAL_TASK_SEQUENCE`:

| k | env | role | checkpoint | eval |
|---|---|---|---|---|
| 5 | `sawyer_handle_press_side` | CRL fails (latch) | `task_0_step_100200.pkl` | 0% at 1M (job 17901349) |
| 8 | `sawyer_window_close` | CRL fails (latch) | `task_0_step_250500.pkl` | **0.0%** at 1M (job 17954891) |
| 6 | `sawyer_push` | well-solved continuous object motion | `task_0_step_250500.pkl` | ~90% in the matched handle/push cell |

Matched algorithm: DCC decomposed, reset actor, `adapt_heads_only=false`,
width 1024×4, `sawyer_success_mode=corrected`, `dyn_aux_weight=1`,
no Success-BC, seed **6**, 1M steps, `goal_conditioning_mode=full_state`.
Task 8 was trained in this job with
`experiment_configs_jubail_window_measure.py` into the same checkpoint
root as handle/push.

Handle is probed at ~100k (policy still reaches the fixture) rather than
250k. That is the more conservative comparison: later handle checkpoints
leave hover, which would **shrink** contact mass further. Push and window
are probed at ~250k. Only seed 6 is matched; no seed-level CI. Survival
bands are a 200-draw bootstrap over the 6000 on-policy states.

Reverb buffers are **not** saved. Occupancy is **on-policy stochastic
rollouts** of the trained actor (40 episodes × 150 steps), a late-training
replay proxy, not \(\mu\) of the serialized buffer.

Paths:

- ckpts: `logs/jubail_task5_action_advice/checkpoints/actor_reset_critic_decomposed_tid_False_heads_False_success_corrected_dyn1.000_pt256x4_env_{env}/seed_6/`
- probes: `logs/jubail_action_informative_mass/runs/action_mass_*.json|.npz`

## What is computed

At each collected state \(s\), score
\(f(s,a,g_{\mathrm{task}})=\phi(s,a)^\top\psi(g_{\mathrm{task}})\)
on: \(\pi\)-mode, zero, axis \(\pm 0.5\), Gaussian
\(\sigma\in\{0.10,0.25\}\) around \(\pi\), shuffled replay actions,
uniform \([-1,1]\). Sawyer actions are all in \([-1,1]^4\) (same scale).

\[
\Delta_f(s,g)=\max_a f(s,a,g)-\min_a f(s,a,g).
\]

**Scale correction.** Raw \(\Delta_f\) is not comparable across tasks:
on-policy \(f(s,a_\pi,g)\) has \(\sigma_s=52.8\) (handle), \(26.8\)
(window), \(4.79\) (push). The plotted critic proxy is

\[
\Delta_f(s,g)/\sigma_s,
\qquad
\sigma_s=\mathrm{std}\{f(s,a_\pi,g)\}
\text{ on the same rollouts}.
\]

Panel B uses one shared \(\varepsilon\) grid in units of \(\sigma_s\),
not tuned per task. Dotted curves repeat the survival using only
\(\{\pi\}\cup\) Gaussian-\(\sigma=0.25\) (local perturbation).

Contact occupancy: `her_future_phase` hand–object/mechanism distance
\(<9\) cm (`PHASE_HOVER`, `PHASE_PROGRESS`, or `PHASE_SUCCESS`). This is
a geometric stand-in for “states at which an action *can* move the
object,” not a simulator contact flag.

Environment branch: MuJoCo snapshot restore from 64 states; first
actions \(\{\pi,0,\text{task-bias},3\times\mathrm{Unif}\}\), then 8-step
\(\pi\) continuation. Reported quantity is the range of the task
coordinate (handle \(z\), window \(y\), push object–target distance).
This is a stronger *physical* proxy than \(\Delta_f\), still not
\(D_{\mathrm{TV}}\).

InfoNCE categorical accuracy is 256-way HER retrieval
(\(\mathrm{chance}=1/256\approx0.4\%\)) on the same rollouts, with and
without action shuffle.

## Figure

- `results/img/paper/fig_action_informative_mass.{pdf,png}`
- `results/data/action_informative_mass/fig_action_informative_mass.{csv,npz}`
- scripts: `contrastive/action_informative_mass.py`,
  `scripts/measure_action_informative_mass.py`,
  `scripts/plot_action_informative_mass.py`

Panel A, top: 9 cm contact occupancy vs episode time.
Panel A, bottom: \(\Delta_f/\sigma_s\) vs episode time (dashed line at 1).
Panel B: \(P(\Delta_f/\sigma_s>\varepsilon)\) on a shared grid; dotted =
local Gaussian \(\sigma=0.25\).

The plotted values are unchanged from job `17954891`. The figure was
restyled for the paper: Okabe–Ito colors, a shared legend above the
axes, and panel titles outside the data.

Caption (for later paper use, not pasted yet):

> Action-informative occupancy under on-policy rollouts of matched DCC
> critics (seed 6). **A.** Latch tasks spend little time with the hand
> within 9 cm of the object; push occupies that region for most of the
> episode. Scale-normalized critic action-range \(\Delta_f/\sigma_s\) is
> correspondingly small on Tasks 5/8 and stays above 1 throughout push.
> **B.** Survival of that ratio. Handle-press has essentially no mass
> with \(\Delta_f\) on the order of the state-score scale; push retains
> most mass past \(\varepsilon=1\). Dotted curves use a local Gaussian
> action perturbation (\(\sigma=0.25\)). Neither panel is \(D_{\mathrm{TV}}\)
> of futures.

## Results

All numbers below are seed 6, 6000 states/task, unless noted.

### Where along each task does action sensitivity appear?

**Geometric (9 cm contact).**

| task | contact mass | time profile |
|---|---:|---|
| Handle press | **0.17** | early bump ~0.23–0.31, then ~0.13 |
| Window close | **0.046** | brief bump to ~0.17 at \(t\in[0.2,0.4]\), then **0** |
| Push | **0.86** | rises to ~1 by \(t\approx0.2\) and stays there |

Push: 42% of states are `object_progress`, 28% `success`.
Handle: 60% `far_reach`, 15% `hover_unsolved`, 2.0% `success`.
Window: 82% `far_reach`, 4.6% hover, 0% success in this 40-episode probe
(training log at 250k–1M is `far≈0.83`, `hover≈0.04–0.05`,
`succ≈0.00–0.01`; final eval **0.0%**).

**Critic \(\Delta_f/\sigma_s\) vs time.** Handle is flat at ~0.16 for the
whole episode — **not** a hover/press spike. Window is 0.5–0.8 and
*rises after the policy leaves the window* (contact occupancy → 0).
Push stays ~2 throughout contact, slightly higher during the initial
reach. High critic action-sensitivity is therefore **not** concentrated
on the latch contact slice for Tasks 5/8.

**Environment branch (64 snapshots).** Range of the task coordinate
after a 1-step fork, then 8-step \(\pi\) continuation:

| task | 1-step median | 8-step median (p90) |
|---|---:|---:|
| Handle \(z\) | 0.05 mm | 3.2 mm (4.6 mm) |
| Window \(y\) | **0** | **0** |
| Push object–target | 0.43 mm | 7.7 mm (12 mm) |

Restore succeeded (`env_branch_error` empty). Window never moves.
Handle \(z\) rest is ~0.16 vs target 0.07; 3–5 mm is not a press.
Push moves the cube about a centimetre — small, but an order of
magnitude above the latch tasks.

### How much replay mass lies in those regions?

Contact masses are the table above (17% / 4.6% / 86%).

Scale-normalized critic survival, **shared** \(\varepsilon\), all
candidate actions:

| \(\varepsilon\) | handle | window | push |
|---:|---:|---:|---:|
| 0.46 | 0.002 | 0.655 | 1.000 |
| 1.00 | ~0 | 0.19 | 0.97 |
| 1.16 | 0 | 0.080 | 0.945 |
| 2.09 | 0 | 0.001 | 0.535 |

Local Gaussian \(\sigma=0.25\) (dotted in Panel B) preserves the same
order. Gaussian \(\sigma=0.10\) is even more extreme: at
\(\Delta_f/\sigma_s>0.10\), handle 0.002, window 0.43, push 0.98.

Raw (unnormalized) \(\Delta_f\) **does not** give this ranking — window
mean \(\Delta_f=17.6\) exceeds push \(10.3\) and handle \(8.5\). That
is a score-scale artifact (\(\sigma_s\) of \(f\) is 11× larger on
handle than on push). The figure therefore plots \(\Delta_f/\sigma_s\).
The ratio \(\mathrm{std}_a f/\sigma_s\) is 0.035 / 0.175 / 0.511.

### Are Tasks 5 and 8 qualitatively different from continuous pushing?

**Yes, on occupancy and on scale-normalized critic sensitivity.**
Push lives in contact and has \(\Delta_f>\sigma_s\) on almost every
state. Tasks 5 and 8 do not.

They are **not** interchangeable with each other. Handle is the
near-action-invariant extreme (flat \(\Delta_f/\sigma_s\approx0.16\),
shuffle does not hurt retrieval). Window has an intermediate critic
ratio (~0.66) whose action-sensitivity is *not* at the window, and
exactly-zero physical window motion under branching. Claiming “Tasks
5 and 8 look the same” would be too coarse; claiming “both are unlike
push in occupancy and in \(\Delta_f/\sigma_s\)” is supported.

### Does high contrastive accuracy coexist with weak action sensitivity?

256-way HER categorical accuracy vs action-shuffled accuracy:

| task | accuracy | shuffled | \(\Delta_f/\sigma_s\) mean |
|---|---:|---:|---:|
| Handle | 0.148 | **0.164** | 0.16 |
| Window | 0.168 | 0.117 | 0.66 |
| Push | 0.156 | 0.109 | 2.14 |

Chance is 0.39%. All three retrieve well above chance. Handle retrieval
is **unchanged** (slightly higher) after shuffling actions, which is the
key prediction: a usable matching critic that does not use \(a\).
Window and push lose a few points to shuffle, consistent with their
larger relative \(\Delta_f\). This probe does **not** show a unique
“high accuracy, low \(\Delta_f\)” pattern on window relative to push
(accuracies are similar); the dissociation is handle vs everyone, and
occupancy/relative-\(\Delta_f\) vs push for both latch tasks.

### Which claims are directly supported, and which remain hypotheses?

**Supported (this seed, this proxy, on-policy rollouts):**

1. 9 cm interaction occupancy is much smaller on Tasks 5 and 8 than on
   push (17% and 4.6% vs 86%), and is a brief bump rather than a
   plateau.
2. After dividing by the on-policy scale of \(f\), critic action-range
   occupancy ranks handle \(\ll\) window \(\ll\) push. The ranking is
   stable under Gaussian \(\sigma\in\{0.10,0.25\}\).
3. Raw \(\Delta_f\) is *not* a valid cross-task comparison (window would
   look most action-sensitive).
4. Handle retrieval can stay high after action shuffle.
5. Critic \(\Delta_f/\sigma_s\) on Tasks 5/8 is **not** a narrow spike
   at press/close; handle is flat, window rises away from the fixture.
6. Snapshot branching moves the window by 0, the handle by millimetres,
   the cube by ~1 cm. Critic \(\Delta_f\) is therefore not a TV estimate.
7. Task 8 DCC with the matched config fails (0% eval).

**Not supported / still hypotheses:**

- That \(\Delta_f/\sigma_s\) or 9 cm occupancy equals
  \(\mu(\mathcal{S}_{\mathrm{act}}^\varepsilon)\).
- Causality: we did not intervene on occupancy and measure learning.
- Seed-level uncertainty (n=1 matched seed).
- True replay-buffer occupancy (buffers were not serialized).
- That Task 8's critic is action-*invariant* (it is not; it is
  weakly action-sensitive relative to push, and in the wrong region).
- That 8-step coordinate range is a tight TV approximation.

## Code and validation

```bash
python tests/test_action_informative_mass.py
python scripts/plot_action_informative_mass.py --npz \
  logs/jubail_action_informative_mass/runs/action_mass_handle_press_side_s6_step100200.npz \
  logs/jubail_action_informative_mass/runs/action_mass_window_close_s6_step250500.npz \
  logs/jubail_action_informative_mass/runs/action_mass_push_s6_step250500.npz
```

Launch of the (already completed) job: `sbatch DRAFT_jubail_action_informative_mass.sh`.
