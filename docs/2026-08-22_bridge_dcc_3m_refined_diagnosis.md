# Refined Task-5/Task-8 diagnosis after the Bridge-DCC pilots

Date: 2026-08-22
Project: `nyuad_mmvc/continual_gcrl_paper`
W&B snapshot: approximately 2026-08-22 09:57 UTC

## Scope and status

This report analyzes the three proposed fixes at their current horizons:

1. **IWR-DCC**: interaction-weighted future-goal relabeling.
2. **Advantage-DCC**: a task-local forward action-effect head.
3. **Bridge-DCC**: IWR-DCC and Advantage-DCC together.

All cells use in-trajectory negatives with repetition factor $r=12$ and seeds
$5,6,7$. The runs are still partial. The three Task-5 IWR-DCC runs crashed
together at approximately $1.62$M steps; all other cells were running between
approximately $2.8$M and $3.9$M steps when the data were retrieved. Therefore,
the present data are sufficient to reject the mechanisms **as implemented**,
but not to estimate final 8M-step performance precisely.

## Intended hypotheses and acceptance tests

| Variant | Intended repair | Main acceptance test |
|---|---|---|
| IWR-DCC | Increase representation of rare contact/interaction bridge states | More near-bridge samples and higher matched-horizon success AUC |
| Advantage-DCC | Supply local goal-conditioned action credit missing from DCC | Positive same-state score versus 100-step mechanism-progress correlation |
| Bridge-DCC | Fix both sparse interaction coverage and action credit | Both mechanisms activate and jointly improve success AUC |

The pre-registered interpretation was: IWR-only success would support a
coverage bottleneck; Advantage-only success would support missing local action
comparison; Bridge-only success would imply that both were jointly necessary.

## Performance: no material improvement over the CRTR control

The table uses the three-seed mean evaluation curve. `Peak` is a five-point
smoothed peak, and `late` is the mean of the last 20% of that smoothed curve.
The CRTR column is evaluated at the same horizon as each new variant.

### Task 5: Sawyer handle press side

| Variant | Horizon | Success AUC | CRTR AUC at same horizon | Peak | Late | Final | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| IWR-DCC | 1.60M | 6.5% | 6.9% | 10.0% | 3.9% | 0.0% | Crashed |
| Advantage-DCC | 3.70M | 5.1% | 5.3% | 10.7% | 4.3% | 0.0% | Running |
| Bridge-DCC | 2.80M | 5.1% | 5.8% | 8.7% | 5.7% | 13.3% | Running |

### Task 8: Sawyer window close

| Variant | Horizon | Success AUC | CRTR AUC at same horizon | Peak | Late | Final | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| IWR-DCC | 3.80M | 13.6% | 13.8% | 20.7% | 12.3% | 13.3% | Running |
| Advantage-DCC | 3.85M | 12.4% | approximately 13.8% | 24.0% | 9.4% | 6.7% | Running |
| Bridge-DCC | 3.35M | 14.3% | 13.9% | 27.3% | 12.1% | 6.7% | Running |

Bridge-DCC's Task-8 prefix AUC is only $0.4$ percentage points above CRTR at
the matched horizon, while the other comparisons are equal or worse. This is
not evidence that any variant repaired the task. More importantly, all Task-8
curves peak near $0.25$--$0.35$M steps and then lose roughly $8$--$15$
percentage points between their smoothed peak and late performance.

## Diagnostic results

Dense learner metrics below are averaged over the late 20% of their histories.
Sparse causal-probe metrics are averaged over all available probe events so a
single 16-action probe does not determine the conclusion.

| Variant | Task | IWR near-bridge fraction | Effect cosine | Effect advantage std. | Near-contact anchors | $\rho$(score, 100-step mechanism progress) | Mechanism-outcome std. |
|---|---:|---:|---:|---:|---:|---:|---:|
| IWR-DCC | 5 | 0.212 | — | — | 0.556 | -0.010 | 0.00350 |
| Advantage-DCC | 5 | — | 0.858 | 0.00143 | 0.286 | 0.017 | 0.00113 |
| Bridge-DCC | 5 | 0.134 | 0.794 | 0.00150 | 0.400 | -0.095 | 0.00485 |
| IWR-DCC | 8 | 0.144 | — | — | 0.262 | 0.002 | 0.00045 |
| Advantage-DCC | 8 | — | 0.752 | 0.02264 | 0.143 | 0.002 | 0.00018 |
| Bridge-DCC | 8 | 0.114 | 0.767 | 0.01733 | 0.278 | 0.015 | 0.00127 |

### 1. The action-effect auxiliary task was learned, but it was the wrong target

The forward action-effect cosine is high: approximately $0.79$--$0.86$ on
Task 5 and $0.75$--$0.77$ on Task 8. It also continues to increase after
evaluation success has already peaked and begun to collapse. Nevertheless,
the combined actor score has approximately zero or negative same-state
correlation with measured 100-step mechanism progress.

This rejects the explanation that Advantage-DCC merely failed to optimize.
The head successfully predicts

$$
y_t = \operatorname{sg}\left[
\gamma\bar\psi(s_{t+1})-\bar\psi(s_t)
\right],
$$

but success on this auxiliary task does not imply correct action ordering in
the environment. The target inherits the same learned $\psi$ geometry whose
control calibration was already in question. It measures predictable
one-step movement in representation space, not the contact sequence and
mechanism displacement that determine success.

### 2. The action-effect correction is probably too weak to replace DCC's hill

The actor maximizes

$$
C(s,a,g)=
\frac{f_{\mathrm{DCC}}(s,a,g)}{m}
+A_{\mathrm{effect}}(s,a,g).
$$

The observed `action_effect/advantage_std` is only about $0.0014$--$0.0015$
on Task 5 and $0.017$--$0.023$ on Task 8. By comparison, the same-state
candidate standard deviation of the combined score is approximately
$0.40$--$0.49$ for the action-head variants. These are not perfectly matched
statistics, but they strongly suggest that the new term is a small correction
to the original DCC landscape rather than a replacement for it.

The current logs do not decompose actor gradients from the DCC and action-effect
terms. Therefore, numerical and directional gradient dominance remains a
conjecture, not a proven cause.

### 3. The head may have learned another state-only shortcut

The network $u_k(s,a)$ receives an action, but the loss never tests whether it
needs that action. In replay generated by a nearly deterministic policy,
$s_t$ alone can predict much of $\psi(s_{t+1})-\psi(s_t)$. A high cosine can
therefore coexist with an action-insensitive effect head.

The logged advantage standard deviation is computed across different states;
it does not establish same-state variation across alternative actions. No
action-shuffle, zero-action, or fixed-state intervention was recorded for the
head itself. This is a major untested shortcut hypothesis.

### 4. IWR reweighted future goals, not the transitions that teach contact

IWR's late selected near-bridge fraction is only $0.11$--$0.21$, and its mean
selected hand-to-mechanism distance is approximately $0.26$--$0.34$, well
above the nominal $0.09$ contact boundary. More fundamentally, IWR changes
which future state is used as the relabeled goal; it does not create more
current-state contact transitions, mechanism-moving actions, or successful
action sequences.

Thus, even if IWR made the contrastive classification problem harder, it did
not fix the action-data distribution. Its matched-horizon AUC is equal to or
worse than CRTR.

### 5. The relevant control unit is probably an action sequence, not one action

Candidate scores vary substantially, but candidate outcomes do not. Across
the causal probes, the standard deviation of 100-step mechanism progress is
only approximately $0.00018$--$0.00485$, while full-state progress varies much
more. The initial candidate action is followed by a common continuation
policy; one action rarely changes the contact mode or the final mechanism
position enough to produce an identifiable ranking signal.

This is not explained only by imperfect anchors. Restricting the analysis to
probe events where at least half of the anchors are near contact gives mean
score--mechanism Spearman correlations of approximately $-0.068$ on Task 5
and $0.012$ on Task 8. Even at interaction states, the current one-action score
does not rank long-horizon progress.

### 6. The original action-free contrastive shortcut persists

Across all six method-task cells, categorical accuracy retains approximately
$0.89$--$0.95$ of its value after shuffling actions and $0.96$--$0.98$ after
replacing actions with zero. The DCC retrieval component therefore remains
largely action-independent.

The coordinate intervention result also changes our earlier, more specific
story. Shuffling hand-goal coordinates lowers categorical accuracy by roughly
$0.13$--$0.20$, whereas shuffling mechanism-goal coordinates lowers it by only
roughly $0.01$--$0.06$. These off-manifold interventions do not prove causal
feature use, but they indicate that the shortcut feature block is not stable
across contrastive objectives. The robust diagnosis is **action-free goal
retrieval**, not permanent reliance on one specific coordinate block.

### 7. Transient discovery and retention are separate problems

Task-8 mean success briefly reaches approximately $0.21$--$0.27$ and then
falls to late values of approximately $0.09$--$0.12$. During this decline,
the effect cosine improves rather than collapses. The system can occasionally
discover useful behavior, but the ongoing contrastive/actor updates do not
preserve it.

This suggests a second failure mode after exploration: rare successful
transitions are diluted by replay, the goal geometry continues to move, and
the actor drifts away from its best policy. Representation drift and policy
drift were not directly logged, so their relative contributions remain open.

## Refined causal picture

The new results refine “the actor is climbing the wrong hill” into four linked
problems:

1. **The hill remains mostly DCC's hill.** The action-effect term is small and
   its actor-gradient contribution was not balanced or measured.
2. **The auxiliary target is self-consistent but not outcome-calibrated.** It
   predicts one-step change in the same learned geometry that caused the
   original mismatch.
3. **Contact control is sequence-dependent.** A single alternative action has
   almost no measurable long-horizon mechanism effect under a common
   continuation policy.
4. **Rare success is not retained.** Briefly useful policies disappear even
   while representation-prediction metrics continue improving.

## New conjectures and decisive tests

| Conjecture | Current evidence | Decisive test |
|---|---|---|
| $u_k(s,a)$ ignores $a$ | High cosine but no useful ranking; no head intervention was logged | Shuffle/zero actions for the head and measure fixed-$(s,g)$ head-score variance |
| DCC actor gradients dominate or oppose the head | Head advantage variance is tiny relative to combined-score variation | Log separate actor gradient norms, their cosine, and their ratio; run effect-only actor ablation |
| $\psi$ geometry is the wrong progress geometry | Effect cosine is high while task-progress Spearman is zero | Train the same head against raw achieved-goal/mechanism progress and compare ranking |
| One-step credit is insufficient | Mechanism-outcome variance across one-action candidates is nearly zero | Intervene with action chunks and train $H$-step targets for $H\in\{5,25,100\}$ |
| IWR changes goals but not useful action coverage | Low late near-bridge fraction and no AUC gain | Log and oversample actual contact/mechanism-moving transitions, not future goals |
| Moving representations destabilize the target | Plausible, not measured | Fixed-anchor $\psi$ drift, label drift, and frozen/EMA target-encoder ablation |
| Successful behavior is forgotten | Early peak followed by persistent decline | Compare current, EMA, and best-checkpoint actors; add success-buffer distillation |

## Recommended next proposal: Outcome-Calibrated Sequence DCC

The next algorithm should **decouple DCC representation learning from the
actor's control objective**. DCC can remain an auxiliary/shared representation
learner for transfer, but the actor should be trained by a task-local,
non-bootstrapped finite-horizon outcome head.

For a diagnostic Sawyer version, define the empirical $H$-step mechanism
progress target

$$
y_t^{(H)}(g)=
d_M(s_t,g)-\min_{1\le h\le H}d_M(s_{t+h},g),
$$

and the finite-horizon success target

$$
c_t^{(H)}(g)=
\max_{1\le h\le H}\mathbf{1}
\left[\operatorname{success}(s_{t+h},g)\right].
$$

A task-local head $Q_H(s_t,a_t,g)$ is trained by supervised Monte Carlo
regression/classification:

$$
\mathcal{L}_{\mathrm{outcome}}
=
\operatorname{Huber}\left(Q_H^{\Delta},y_t^{(H)}\right)
+\lambda_c\operatorname{BCE}\left(Q_H^{c},c_t^{(H)}\right).
$$

There is no TD bootstrap, avoiding the instability seen in RBC-DCC. The actor
initially maximizes the standardized outcome score rather than the DCC score:

$$
\mathcal{L}_{\pi}
=
\mathbb{E}\left[
\alpha\log\pi(a\mid s,g)-\widehat Q_H(s,a,g)
\right]
+\lambda_{\mathrm{BC}}
\mathbb{E}_{(s,a,g)\sim\mathcal{B}_{\mathrm{succ}}}
\left[-\log\pi(a\mid s,g)\right].
$$

$\mathcal{B}_{\mathrm{succ}}$ is a small persistent buffer of successful and
mechanism-moving segments. Its purpose is to preserve useful behavior after it
is first discovered. If DCC shaping is reintroduced later, its coefficient
should be adjusted by actor-gradient norms and must pass a non-conflict test.

For a general goal-conditioned benchmark, the oracle $d_M$ is replaced by the
environment's achieved-goal distance and sparse success predicate. The
mechanism-specific target should first be used as a causal Sawyer diagnostic:
if even an oracle-aligned finite-horizon head cannot solve Tasks 5 and 8, the
problem is action coverage or policy optimization rather than representation
geometry.

## Efficient experiment sequence

Do not launch another 18-cell full run immediately. Use a staged falsification
sequence on Tasks 5 and 8:

1. **Current head, effect-only actor:** removes DCC score dominance while
   retaining the one-step $\psi$ target.
2. **Raw outcome head, effect-only actor:** replaces the representation target
   with $H$-step task progress; test $H=25$ first.
3. **Raw outcome head plus success buffer:** tests whether the late collapse is
   a retention failure.
4. **Action-chunk version:** only if the $H$-step label is informative but the
   first-action head still cannot rank outcomes.

Run seeds 5 and 6 to $1$M steps for these mechanism tests, then expand only a
variant that passes the causal metrics to seeds 5, 6, and 7 and the full
horizon.

## Acceptance criteria for the next implementation

A new method should not advance to full continual experiments unless it shows:

- head-only fixed-state action variance that collapses under action shuffling;
- positive same-state score versus $H$-step mechanism-progress Spearman,
  preferably $\rho>0.3$ across seeds;
- non-degenerate candidate outcome variance at verified contact anchors;
- a controlled ratio and non-negative cosine between DCC and outcome-head
  actor gradients if both are used;
- matched-horizon success AUC at least 5 percentage points above CRTR; and
- late success at least 80% of the smoothed peak, indicating that discoveries
  are retained.

## Bottom line

The new variants did not fail because their auxiliary losses were unlearnable.
They failed because the supervision was still attached to the wrong object:
future-goal sampling and one-step change in DCC representation space do not
provide sequence-level, task-outcome-calibrated action credit. The next test
should remove DCC's score from the actor, supervise a finite-horizon outcome
head without TD bootstrapping, sample actual mechanism-moving transitions, and
retain successful segments explicitly.
