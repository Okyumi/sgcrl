# Counterfactual-Ranking DCC at 1M Steps: Diagnosis and Next Proposal

Date: 2026-08-23  
Project: `nyuad_mmvc/continual_gcrl_paper`  
Branch evaluated: `section3_done` at `48e38beb0252900966982f7c91ffe10b8ad88ff6`  
Experiment groups: `CFRDCC-taskgoal-H100-chunk5-task5` and `CFRDCC-taskgoal-H100-chunk5-task8`

## Executive conclusion

The task-goal counterfactual experiment fixed two earlier problems: it produced nontrivial outcome variation from identical simulator states, and the new head became strongly action-dependent. It did **not** produce a control-calibrated landscape on states visited by the live policy. The clearest signature is that the head can rank the just-collected scripted-contact candidates, especially on Task 5, while its score has negative or approximately zero correlation with outcomes in the independent same-state probe.

The result refines the earlier phrase “the actor is climbing the wrong hill.” The new hill is locally learnable, but it is learned on a narrow, privileged contact distribution and under five-step repeated-action semantics; the actor is then asked to use it globally while choosing a fresh action every step. Therefore the next test should not add another global actor loss. It should first align labels, held-out evaluation, and execution, then test a phase-gated, receding-horizon chunk controller.

## Snapshot and completion status

The W&B snapshot contains four intended cells. One run completed the 1M budget; W&B marks the other three as crashed after substantial but incomplete prefixes.

| Task | Seed | W&B run | State | Latest learner step | Latest evaluator step | Latest evaluation success |
|---|---:|---|---|---:|---:|---:|
| Handle press side (Task 5) | 5 | `368qol5c` | Finished | 990,000 | 951,900 | 0.00 |
| Handle press side (Task 5) | 6 | `azl7314d` | Crashed | 945,000 | 951,900 | 0.10 |
| Window close (Task 8) | 5 | `7406x25w` | Crashed | 894,450 | 851,700 | 0.00 |
| Window close (Task 8) | 6 | `2rnrkhnv` | Crashed | 847,500 | 851,700 | 0.00 |

The displayed evaluation curves peak transiently at approximately 0.25 on Task 5 and 0.10 on Task 8, then fall. Because three runs are not marked finished, the existing evaluator correctly blocks formal promotion and an exact final two-seed AUC should not be reported as a completed result. The prefixes are nevertheless sufficient to reject promotion: the success is not retained, and the independent causal-ranking gate fails by a large margin on both tasks.

The paired crash timing should be checked in the Torch `.out` and `.err` files before rerunning. This is an execution-status issue, but it does not explain the failed causal-ranking metrics recorded well before the endpoints.

## The hypothesis sequence so far

| Experiment | Intended hypothesis | Result |
|---|---|---|
| Plain contrastive RL / DCC | Contrastive future-goal retrieval supplies a useful actor landscape | Rejected on Tasks 5 and 8: high retrieval accuracy coexisted with poor control and action removal retained most retrieval performance |
| RBC-DCC | A Bellman residual can calibrate DCC as a value function | Rejected: the additive residual and moving compressed features produced severe numerical instability |
| DCC-SAC | A bounded twin-Q branch can safely supply value ranking | Rejected as implemented: stable HER Q statistics did not imply useful original-goal ranking, and the gate was often unused or uninformative |
| AC-DCC | Explicit action identification repairs action insensitivity | Rejected: 32-way action accuracy reached 77% on Task 5 and 68% on Task 8, but learned one-step compatibility/inverse dynamics rather than long-horizon action ordering |
| In-trajectory negatives | Harder negatives prevent easy future-goal retrieval | Partially supported: categorical accuracy became harder and AUC improved consistently, but absolute success and same-state action ranking remained poor |
| IWR-DCC | Reweighting interaction bridge goals fixes contact coverage | Rejected as implemented: it changed relabeled goals, not the supply of contact-producing current-state/action sequences |
| Advantage/Bridge DCC | A learned one-step change in $\psi$ supplies local action credit | Rejected: effect cosine was high, but score versus real mechanism progress stayed near zero |
| Stage-1 effect-only actor | DCC numerically overwhelms an otherwise useful effect head | Rejected: removing DCC from the actor did not repair same-state ordering |
| Stage-2 observational $H=25$ outcome head | Raw finite-horizon progress fixes the learned-$\psi$ target | Rejected: labels were nearly all positive and one behavior action per state allowed another state/trajectory shortcut |
| Task-goal same-state counterfactual ranker | Identical-state candidate rollouts remove the observational shortcut and directly teach action ordering | **Partly supported locally, rejected as a global actor critic:** labels vary and the head fits some collected batches, but held-out causal ordering and full-episode control fail |

The consistent historical conclusion remains that actor exploitation is not the primary cause. The policy-score versus outcome-percentile gap in the present independent probe averages only about 0.05 on both tasks, and the policy-to-replay action distance remains modest. The stronger failure is that the score does not generalize as an outcome ranking on the policy's actual state distribution.

## What the new experiment successfully fixed

### 1. The collector generated informative same-state alternatives

At the latest event, every Task-5 anchor and 75%--100% of Task-8 anchors were informative. Candidate progress standard deviation is far above the registered $0.002$ floor.

| Metric, latest event averaged over seeds | Task 5 | Task 8 | Gate |
|---|---:|---:|---:|
| Informative-anchor fraction | 1.000 | 0.875 | $\geq 0.25$ |
| Candidate progress standard deviation | 0.0157 | 0.0476 | $\geq 0.002$ |
| Task-success variation across candidates | 0.875 | 0.375 | Diagnostic |
| Candidate success fraction | 0.602 | 0.281 | Diagnostic |
| Scripted-anchor near-interaction fraction | 1.000 | 1.000 | Diagnostic |

This rejects the narrow conjecture that no candidate action can measurably change a 100-step outcome once a useful contact state is supplied.

### 2. The head did not ignore the action

The fixed-$(s,g)$ score standard deviation is large, and the current-batch rank metrics are substantially better than chance on Task 5.

| Metric, latest event averaged over seeds | Task 5 | Task 8 | Registered gate |
|---|---:|---:|---:|
| Fixed-state score standard deviation | 0.718 | 0.909 | $\geq 0.001$ |
| Current-batch rank Spearman | 0.479 | 0.246 | $\geq 0.30$ |
| Current-batch pairwise accuracy | 0.754 | 0.662 | $\geq 0.65$ |

Task 5 therefore supports the claim that exact-state counterfactual supervision can teach a local action ordering. Task 8 is seed-unstable: its final rank Spearman is $0.597$ for seed 5 and $-0.105$ for seed 6, while pairwise accuracy ranges from $0.864$ to $0.460$.

## What failed

### 1. The learned order did not transfer to the independent causal probe

The decisive held-out-like intervention is the independently collected action-landscape probe. It fixes the same state and goal, compares policy, local, replay, and uniform actions, and measures the resulting 100-step mechanism progress.

| Metric, latest event averaged over seeds | Task 5 | Task 8 | Gate |
|---|---:|---:|---:|
| Independent score/progress Spearman | **-0.150** | **-0.039** | $\geq 0.30$ |
| Independent candidate progress std. | 0.00889 | 0.00082 | Diagnostic |
| Independent candidate score std. | 0.722 | 0.596 | Diagnostic |
| Policy score-minus-outcome percentile gap | 0.047 | 0.052 | Near zero is not an exploitation signature |
| Policy-to-replay action distance | 0.069 | 0.056 | Diagnostic |

The score varies strongly across actions, but those variations point in the wrong direction. This is not the original action-insensitive DCC failure. It is **action-sensitive misranking outside the narrow data on which the ranker was fitted**.

### 2. The collector and the live policy occupy different state distributions

The rank collector uses a scripted hand-to-mechanism controller and reaches the nominal contact region for every latest anchor. The independent probe searches states reached by the current policy.

| Anchor source | Task 5 near interaction | Task 8 near interaction |
|---|---:|---:|
| Scripted rank collector | 1.00 | 1.00 |
| Current-policy independent probe | 0.75 | 0.00 |

This is especially decisive on Task 8: the ranker is trained where candidate actions often change the mechanism and sometimes satisfy the proxy success threshold, but the actual policy never reaches that region in the latest probe. The effect-only actor applies the contact-trained head to every replay state, so its gradients at approach states are extrapolations rather than supervised control signals.

Task 5 shows an additional complication. Some policy-derived anchors satisfy the coarse hand-to-mechanism distance threshold, but causal ranking is still negative. A single distance threshold does not identify contact mode, contact direction, gripper state, or whether the mechanism is mechanically engaged.

### 3. Training, diagnosis, and execution use different action semantics

For a candidate vector $a$, the collector labels the outcome of repeating it for $C=5$ steps and then following the deterministic policy:

$$
y_5(s,a,g)
=
\Delta d_M^{(H=100)}(s,\underbrace{a,\ldots,a}_{5\ \mathrm{steps}},\pi,g)
+\lambda z.
$$

The independent probe instead changes only the first action before following the policy, so it measures $y_1(s,a,g)$. The actor also samples a fresh action at every environment step; it does not commit to the five-step intervention whose result trained the head. Therefore a head that correctly approximates $y_5$ can still fail the registered $y_1$ probe and can give an inappropriate gradient to a one-step actor.

This mismatch means the present negative causal correlation cannot by itself distinguish poor ranker generalization from a correct five-step ranker evaluated and executed as a one-step score. Full-episode failure shows that the current use is ineffective, but the exact cause requires an aligned repeat-$5$ probe.

### 4. The current-batch ranking metrics are optimistic

At every collection event, the implementation:

1. adds the newly collected four anchors to the training buffer;
2. performs 25 rank updates from that buffer; and
3. scores the same newly collected batch for `score_vs_task_progress_spearman` and `pairwise_accuracy`.

The metrics are therefore resubstitution measurements, not clean held-out generalization. With 80 anchors in the full buffer, the four current anchors are expected to be sampled repeatedly across 25 batches of 16 anchors. The Task-5 gap between current-batch Spearman $0.479$ and independent Spearman $-0.150$ is consistent with overfitting, distribution shift, action-semantic mismatch, or a combination of all three.

### 5. One metric is mislabeled and should not be used as evidence

`learner/counterfactual_rank/action_shuffle_retention` does not re-score shuffled or zero actions. It circularly shifts the already computed score vector and measures its centered correlation with the original vector. Because candidates are ordered by action family, the value can reflect ordering and family structure rather than action dependence. Fixed-state score variance is valid evidence of action sensitivity; the current “shuffle retention” key should be replaced, not interpreted.

### 6. The success proxy must be checked against the benchmark predicate

The rank collector defines success as mechanism-goal distance at most $0.05$. The independent probe uses the environment's emitted reward/success signal, and the evaluator uses the benchmark episode success. The large gap between counterfactual candidate success and evaluation success may be caused mostly by state-distribution and control differences, but the three definitions have not been validated as equivalent.

The next diagnostic must log a confusion matrix between the distance proxy and the exact evaluator success predicate. Until then, `counterfactual_rank/task_success_fraction` should be described as **proxy success**, not benchmark success.

## Refined causal explanation

The new experiment rules out the claim that exact-state counterfactual supervision is inherently uninformative. It instead identifies a credit-support-execution mismatch:

1. useful labels are generated only after a scripted controller places the hand on a narrow contact manifold;
2. the head is trained and evaluated on overlapping contact anchors;
3. the live actor uses that head on all states, including unsupported approach states;
4. the head predicts the effect of a five-step repeated action, but the actor and independent probe use one-step decisions; and
5. repeated online actor updates do not preserve the transient successful policy.

The original DCC action-free retrieval shortcut still appears in the new runs, but it is not the direct cause of this cell because the actor uses only the counterfactual head. Actor exploitation also remains unsupported as the primary explanation. The present failure is more specific: **the ranker learned a local intervention model and was deployed as a global, one-step critic**.

## New proposal: Phase-Gated Counterfactual Chunk Control

The next proposal treats the counterfactual head as a local contact controller rather than a global value function. DCC can remain an auxiliary representation learner, but it should not define the manipulation actor objective until it passes causal calibration.

### Phase 1: reach the interaction support

For the diagnostic Sawyer version, define interaction distance

$$
d_I(s)=\lVert x_{\mathrm{hand}}(s)-x_{\mathrm{mechanism}}(s)\rVert_2.
$$

A reach policy is trained or distilled to reduce $d_I$ until a contact-support classifier $c(s)$ activates. The first falsification may use the existing scripted contact controller, but that must be reported as a diagnostic oracle rather than a general solution.

### Phase 2: rank and execute action chunks

Let a chunk be

$$
\mathbf a=(a_0,\ldots,a_{C-1}).
$$

From exactly restored state $s$, evaluate candidate chunks with the **same benchmark success predicate** and define

$$
y(s,\mathbf a,g)
=
\max_{1\leq t\leq H}
\left[d_M(s,g)-d_M(s_t,g)\right]
+\lambda\max_{1\leq t\leq H}\operatorname{success}_{\mathrm{env}}(s_t).
$$

Train $U_\eta(s,g,\mathbf a)$ only on within-state preferences, using disjoint training and validation anchors:

$$
\mathcal L_{\mathrm{chunk}}
=
\mathbb E_{j,k}
\log\left(1+
\exp\left[-\operatorname{sign}(y_j-y_k)
\frac{U_\eta(s,g,\mathbf a_j)-U_\eta(s,g,\mathbf a_k)}{\tau}
\right]\right).
$$

At contact, select a chunk from supported candidates, execute exactly the labeled chunk length, and replan:

$$
\mathbf a^*=\arg\max_{\mathbf a\in\mathcal C(s)}U_\eta(s,g,\mathbf a).
$$

This makes the supervised intervention, diagnostic, and deployed action unit identical.

### Phase 3: retain successful behavior

Only after held-out causal ranking passes, retain benchmark-success chunks in a task-local buffer and distill them into the policy. Use an EMA or best-checkpoint policy to measure and reduce drift. Retention cannot repair a wrong score, so it remains downstream of the causal-ranking gate.

## Staged falsification sequence

### Stage A: repair the measurement before another training run

Budget: dependency-light tests plus short simulator probes; no 1M learner run.

Implement:

- held-out anchor buffers that are never sampled for optimization;
- pre-update and post-update score metrics for each new batch;
- aligned probes for action repeat $C\in\{1,5\}$ using the same best-over-$H$ progress definition;
- exact environment/evaluator success plus the mechanism proxy and their confusion matrix;
- true action permutation degradation rather than circularly shifting scores;
- metrics binned by approach, pre-contact, contact, and mechanism-moving phases;
- policy reach rate and time-to-contact.

Proceed only if the repeated-action held-out probe has Spearman at least $0.30$ and pairwise accuracy at least $0.65$ on both tasks.

### Stage B: oracle controller decomposition

Budget: 100--200 evaluation episodes per condition, not 1M training steps.

Compare four conditions:

| Condition | What it tests |
|---|---|
| Current policy reaches; oracle selects one action | Whether reaching is already sufficient |
| Current policy reaches; oracle selects a five-step chunk | Whether temporal abstraction is missing |
| Scripted contact; oracle selects one action | Whether contact-state support is the main bottleneck |
| Scripted contact; oracle selects and replans five-step chunks | Whether the candidate family and horizon can solve manipulation at all |

Here “oracle” means selecting the candidate with the best measured counterfactual benchmark outcome, not using the learned score. If the best available candidate cannot solve the task, changing the ranker or actor is unjustified; the candidate sequence class or horizon must be expanded first.

Promotion gate: from scripted contact, the oracle chunk selector should reach at least 50% benchmark success and exceed random candidate selection by at least 20 percentage points. The live-policy condition must also report reach rate separately.

### Stage C: held-out local chunk ranker

Budget: at most 250k learner steps or an equivalent fixed offline dataset.

Train on phase-stratified anchors and evaluate only on disjoint simulator states before updating on them. Required gates on both tasks:

- held-out chunk-score/outcome Spearman $\geq 0.30$;
- held-out pairwise accuracy $\geq 0.65$;
- top-chunk outcome regret no worse than random selection;
- true action-permutation rank drop $\geq 0.20$;
- validation performance stable across at least two seeds.

If training rank metrics pass but held-out metrics fail, stop and improve data coverage or regularization. Do not connect the head to the actor.

### Stage D: 1M phase-gated closed-loop test

Use the reach controller outside contact support and the learned chunk selector inside it. Execute the full labeled chunk, replan, and distill only benchmark-success behavior. Compare Tasks 5 and 8, seeds 5 and 6, against the matched first-1M CRTR control.

Promotion requires on both tasks:

- success AUC gain of at least $0.05$;
- final-20%-to-peak retention at least $0.80$;
- live-policy contact reach rate at least $0.50$;
- held-out causal chunk Spearman at least $0.30$;
- no material gap between selected-chunk predicted rank and realized outcome rank.

Only a Stage-D winner should receive an 8M run and continual evaluation.

## What not to run next

- Do not extend the current counterfactual-ranker cells to 8M steps.
- Do not proceed directly to success-buffer retention; the causal ranking gate failed.
- Do not tune ranker width or loss temperature using the current resubstitution metrics.
- Do not interpret the current circular-shift metric as action-shuffle retention.
- Do not combine the original DCC score back into the actor before the local controller independently works.

## Immediate engineering checklist

1. Inspect the three crashed jobs' Torch logs and make the launcher exit state unambiguous.
2. Correct the success predicate and W&B metric definitions.
3. Add held-out and repeat-aligned probes.
4. Run the oracle decomposition before training another policy.
5. Implement the phase-gated chunk controller only if the oracle and held-out gates pass.

## Simple takeaway

The experiment showed that the counterfactual idea can create useful local supervision, so the idea itself was not disproved. It failed because the ranker was trained after scripted contact, evaluated partly on the same data, and then used as a global one-step critic even though its labels described five repeated action steps. Task 8 mainly exposes the contact-distribution gap; Task 5 additionally shows that a coarse near-contact state and a locally fitted rank are not enough for reliable manipulation. The next experiment must align the action chunk, success definition, validation split, and executed controller before testing another long run.

