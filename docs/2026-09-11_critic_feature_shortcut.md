# Critic retrieval feature: handle z vs object xy

Date: 2026-09-11
Status: job `17901867` COMPLETED (1m43s). Seed-6 offline probe on
existing handle/push checkpoints. The predicted “handle \(z\)” retrieval
feature is **wrong**. Measured retrieval feature on Task 5 is handle
**\(xy\)** (episode identity). Handle \(z\) is the task bit, but it is too
rare and too action-insensitive for InfoNCE to use.

## Motivation

Task 5's critic is almost scene recognition: scores move with state, almost
not with action. This note records (i) when a sparse binary button makes
InfoNCE action-invariant, (ii) the specific coordinate we expect the critic
to use, and (iii) an offline probe that isolates that coordinate on the
already-trained seed-6 handle and push checkpoints.

No retraining. Checkpoints:

`logs/jubail_task5_action_advice/checkpoints/` (job 17901349).

## Mathematical objective

InfoNCE with in-batch negatives trains \(\phi(s,a)\) and \(\psi(g)\) so that
the matching future \(g^+\) ranks above other batch goals. An optimal
inner-product critic is a sufficient statistic for \(p(g\mid s,a)\).

**Proposition (action invariance).** If \(p(g\mid s,a)=p(g\mid s)\) for
\(\mu\)-almost every training pair, then there exists an InfoNCE-optimal
\(\phi\) that ignores \(a\). Then \(\nabla_a\,\phi(s,a)^\top\psi(g)=0\), and
the actor policy gradient through the critic is zero.

Equivalently, the critic only has to use \(a\) when the conditional mutual
information is large:

\[
I(G;A\mid S)=H(G\mid S)-H(G\mid S,A).
\]

This is small when (i) \(G\) is already nearly determined by \(S\)
(low \(H(G\mid S)\)), or (ii) different actions induce overlapping future
distributions (high \(H(G\mid S,A)\) relative to \(H(G\mid S)\)).

**Density form.** Let \(\mathcal{A}_\varepsilon=\{s:\mathrm{TV}(p(g\mid s,a),p(g\mid s,a'))>\varepsilon\text{ for some }a,a'\}\)
be the action-informative set. SGD on InfoNCE only sees \(\partial\phi/\partial a\)
on mass \(\mu(\mathcal{A}_\varepsilon)\). A binary latch has
\(\mu(\mathcal{A}_\varepsilon)\) concentrated on a thin contact slice at the
press instant. A continuous push has \(\mathcal{A}_\varepsilon\) equal to the
whole contact manifold, and \(G\) (object xy) still depends on \(A\) there.

This is **not** “every sparse binary button fails.” It fails when all of
the following hold:

1. The success coordinate is already in \(s\) and in \(g\) (easy retrieval).
2. The coordinate is a **latch** (absorbing, unimodal-or-bimodal, not a path).
3. Most actions at typical states do not change it, so
   \(I(G;A\mid S)\approx 0\) under the replay occupancy \(\mu\).
4. There is no dense progress band of intermediate \(g\) values that
   different contact actions would produce.

A hold-down button, a button whose state is omitted from \(s\), or
one-step InfoNCE on \(s'\) at the critical contact with diverse negatives
can still force \(\phi\) to use \(a\). Q-learning with the env reward can
back up through the rare transition; InfoNCE does not.

Handle-press-side is the worst case of that list. Push is the opposite:
object xy is a **continuous, action-dependent** coordinate with a progress
band.

## Predicted vs measured features

Unified Sawyer indices: hand `0:3`, gripper `3`, mechanism/object `4:7`.
Task 5 success is handle \(z\) (index 6) within 2 cm of 0.07. Push success
is cube xyz within 5 cm of `(0.02, 0.89, 0.02)`, which is almost xy.

Pre-job guess: Task 5 retrieval = handle \(z\). **Measured: handle \(xy\)**.
\(z\) is the env-success bit but is too rare for InfoNCE.

| | Task 5 (measured) | Push (measured) |
|---|---|---|
| Retrieval feature | handle **\(xy\)** (spawn / episode ID) | object **\(xy\)** + hand |
| Unused for retrieval | handle \(z\), **action** | object \(z\), action (batch-level) |
| Control feature at hover | none (press transplant \(\approx 0\)) | push-biased action (\(+2\) score) |
| Density of task bit | 3.3% then 0.15% in \(z\)-band | 27–52% success + progress band |

## Probe

`scripts/measure_critic_feature_shortcut.py` on handle ~100k + late and
push ~250k + late:

1. **Synthetic transplant** at hover under \(g_{\text{task}}\): write only
   handle \(z\leftarrow 0.07\) (or cube \(\leftarrow\) target). Recovery
   fraction vs copying a full success state. Action-only transplant is the
   control.
2. **HER retrieval shuffle**: drop in categorical accuracy after shuffling
   `mech_z`, `mech_xy`, `hand`, `gripper`, or `action`.
3. **One-step \(\Delta\)** from a frozen hover: how much press vs \(\pi\)
   vs random move handle \(z\) vs cube xy. Empirical \(I(G;A\mid S)\) proxy.
4. Pearson(score, coordinate) and residual correlation of score with the
   press/push axis after regressing out the candidate shortcut.

Launch:

```bash
sbatch DRAFT_jubail_task5_feature_shortcut.sh
```

Logs: `logs/jubail_task5_feature_shortcut/runs/feature_shortcut_*.json`

## Validation

```bash
python tests/test_critic_feature_shortcut.py
```

## How to read the JSON

Handle confirmation of the shortcut:

- `transplant_recovery.synthetic_progress.mean` and `donor_mech_z.mean`
  near 1, `press_action_only` near 0
- `retrieval.goal_mech_z_drop` \(\gg\) `retrieval.action_drop`
- `one_step_mechanism_delta.press.d_mech_z` not much larger than `pi`
- `correlation.score_vs_mech_z` high,
  `score_vs_press_axis_after_shortcut` near 0
- `density.all_states.frac_in_band` small, `z_std` small vs push `xy_std`

Push confirmation of the useful feature:

- `donor_mech_xy` / `synthetic_progress` recover the state gap
- `retrieval.action_drop` or `one_step ... d_mech_xy` under press is large
- residual correlation with the push axis remains after removing object pose

## Results (job `17901867`)

Four JSONs in `logs/jubail_task5_feature_shortcut/runs/`.
`recovery_fraction` vs copying a success *state* is a bad summary on
push (that copy scores *worse* than hover). Read raw score means and
shuffle drops.

### 1. Retrieval: what InfoNCE actually uses

Categorical accuracy drop after shuffling one block of a 256-pair HER batch.
Chance is \(1/256\approx 0.4\%\); baseline accuracy is only \(\sim 0.19\)–\(0.26\)
(hard batch), so drops are relative to that.

| Shuffle | handle 100k drop | handle late drop | push 250k drop | push late drop |
|---|---:|---:|---:|---:|
| **mech_xy** (state) | **0.172** | **0.258** | **0.199** | **0.184** |
| mech_xy (goal) | 0.160 | 0.250 | 0.152 | 0.180 |
| mech_z (state) | 0.027 | 0.008 | 0.023 | 0.039 |
| mech_z (goal) | 0.043 | **0.000** | 0.004 | 0.051 |
| hand (state) | 0.051 | 0.098 | 0.188 | 0.172 |
| **action** | **0.016** | **0.031** | 0.031 | 0.039 |
| zero action | 0.027 | 0.070 | 0.023 | 0.012 |

Both tasks retrieve on **mechanism \(xy\)**. Action is unused for
batch classification. Handle \(z\) is unused. Push also uses **hand**
(\(xy\) of the cube *and* of the gripper).

Handle \(xy\) never moves: one-step \(\Delta xy=0\) for \(\pi\), press, and
random. It is spawn identity, constant inside an episode, randomized
across episodes. HER positives share that identity; in-batch negatives
mostly do not. That is sufficient for retrieval and independent of \(a\).

### 2. Transplant at hover under \(g_{\text{task}}\) (raw scores; higher = better)

Handle 100k (86 hovers). Base \(\pi=-32.7\). Copying a success state: \(-26.9\).

| Intervention | score | vs \(\pi\) |
|---|---:|---:|
| donor handle **xy** | \(-30.1\) (std 1.5 vs 23.9) | recovers the scene cluster |
| donor full handle xyz | \(-17.2\) | overshoots |
| synthetic \(z\leftarrow 0.07\) | \(-23.4\) | off-manifold \(z\) can move the score |
| donor \(z\) only | \(-24.4\) (std still 18.7) | does **not** collapse scene variance |
| **press action only** | \(-32.8\) | **0** |
| donor hand / gripper | \(-36.9\) / \(-32.3\) | none / none |

Late handle: press still 0; \(z\)-only median recovery \(0.01\); **xy**
still the intervention that moves scores.

Push 250k (240 hovers). Base \(\pi=-11.74\). **Push-biased action:
\(-9.69\)** (\(+2.05\), matches the earlier press−π gap of \(+2.04\)).
Teleporting the cube to the target *without* the hand **hurts**
(\(-34.6\)). Copying a finished success state also hurts (\(-15.6\)).
So on push the hover-local control feature is the **action**; object \(xy\)
is the retrieval feature, not a hover-state score cheat.

### 3. One-step occupancy / \(I(G;A\mid S)\) proxy at frozen hover

| | handle \(\Delta z\) | handle \(\Delta xy\) | push \(\Delta xy\) |
|---|---:|---:|---:|
| \(\pi\) | 2.6 mm | **0** | 4.7 mm |
| press / push-bias | 2.6 mm | **0** | 4.7 mm |
| random | 2.6 mm | **0** | 4.5 mm |

At hover, press does not lower the handle more than waving. Handle \(xy\)
is a kinematic fixture. Push does move the cube, but one-step \(|\Delta xy|\)
is similar across actions; the critic still *ranks* the \(+y\) action
(score gap \(+2\)), which this magnitude probe does not see.

### 4. Density of the task bit

| | handle 100k | handle late | push 250k | push late |
|---|---:|---:|---:|---:|
| frac in success band | 3.3% | **0.15%** | 27% | 52% |
| frac in progress band | 0 | 0 | **44%** | 22% |
| Bernoulli entropy of band | 0.15 | 0.01 | 0.58 | 0.69 |
| \(z\) std | 0.029 | 0.023 | 0.002 (table height) | 0.002 |

Handle \(z\) is a rare bit, so shuffling it cannot break retrieval.
Push object-to-target distance is a spread-out coordinate (progress +
success mass).

### 5. Score correlations (handle 100k)

Under \(g_{\text{task}}\): `hand_y` \(+0.78\), `hand_obj_dist` \(-0.71\),
`mech_x` \(+0.53\), `mech_z` \(-0.29\), **press axis \(-0.09\)**.
At hover only: `mech_y` \(+0.74\), `mech_z` \(-0.07\), press axis
\(-0.02\). Scene / instance, not press.

Push 250k: `obj_target_dist` \(+0.45\); after residualizing that
shortcut, press-axis correlation is \(0.07\). Retrieval is still pose;
the *local* action ranking showed up in the action transplant, not in
this global Pearson.

## Interpretation (revises the pre-job prediction)

The critic on Task 5 did **not** latch onto handle \(z\) as the InfoNCE
feature. It latched onto **handle \(xy\)**: an action-invariant episode
ID. Handle \(z\) is the bit the env reward cares about, but \(\mu\) almost
never puts it in the success band, and at hover every action yields the
same 2.6 mm \(\Delta z\). InfoNCE therefore has no reason to use \(z\) or
\(a\).

Push uses **object \(xy\)** for the same retrieval reason (plus hand),
but that coordinate *is* the task and *does* accumulate the push. On top
of that, at a fixed hover state the push-biased action still raises
\(\phi(s,a,g_{\text{task}})^\top\psi(g_{\text{task}})\) by \(\approx 2\).
Handle’s press-biased action raises it by \(\approx 0\).

Same retrieval *type* (mechanism \(xy\)). Different **geometry**: fixture
ID vs displaced object. That is the \(I(G;A\mid S)\) / occupancy claim.
