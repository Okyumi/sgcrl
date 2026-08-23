# Phase-Gated Counterfactual Chunk Control: Staged Implementation

Date: 2026-08-23  
Status: implemented for Torch HPC; jobs not submitted

## Motivation

The first task-goal counterfactual ranker generated informative labels near scripted contact, but it did not improve Tasks 5 or 8. Its latest current-batch score/outcome correlation was positive on Task 5 while the independent same-state score/progress correlation remained negative. The implementation audit identified three confounds:

1. new anchors were added to the training buffer before the same anchors were scored;
2. training labels repeated the candidate action for five steps and used best-over-horizon progress, while the independent probe changed one action and measured final progress; and
3. the ranker was trained only after scripted contact but used as a global one-step actor critic.

The staged implementation isolates these mechanisms before another long run.

## Shared outcome definition

For state $s$, task goal $g$, and mechanism distance $d_M$, the aligned progress target is

$$
p_H(s,\mathbf a,g)
=
d_M(s,g)-\min_{1\leq t\leq H}d_M(s_t,g).
$$

The Sawyer experiments use the repository's sparse step-penalty reward convention:

$$
z_H(s,\mathbf a,g)
=
\max_{1\leq t\leq H}\mathbf 1[r_t=0],
$$

where failure has reward $-1$ and success has reward $0$. The mechanism-distance proxy remains logged, but it no longer silently substitutes for benchmark success in the staged cells. Every collector logs proxy/benchmark agreement, false positives, false negatives, and signal availability.

The ranked outcome is

$$
y_H=p_H+\lambda_s z_H.
$$

## Held-out pairwise ranker

The task-local head $U_\eta(s,g,a)$ represents the outcome of holding action $a$ for $C=5$ steps and then using a common deterministic continuation policy. For candidates $a_j,a_k$ restored from the same simulator state, it minimizes

$$
\mathcal L_{\mathrm{rank}}
=
\log\left(1+
\exp\left[-\operatorname{sign}(y_j-y_k)
\frac{U_\eta(s,g,a_j)-U_\eta(s,g,a_k)}{\tau}
\right]\right).
$$

Fresh validation anchors come from a separately seeded simulator. They are scored before and after each training event and are never inserted into the optimization buffer. The previous current-batch metrics remain for compatibility, but promotion uses only `heldout_post` and the independent causal probe.

## True permutation diagnostic

The old `counterfactual_rank/action_shuffle_retention` circularly shifted an already computed score vector. It did not re-pair action scores with their correct outcomes and could reflect action-family ordering. It is replaced by

$$
\Delta_{\mathrm{perm}}
=
\rho(U(a),y(a))
-\mathbb E_{\pi}\rho(U(a_{\pi}),y(a)),
$$

logged as `learner/counterfactual_rank/*/action_permutation_spearman_drop`.

## Phase-gated chunk controller

Let

$$
d_I(s)=\lVert x_{\mathrm{hand}}(s)-x_{\mathrm{mechanism}}(s)\rVert_2.
$$

Outside contact support, the controller uses either the ordinary policy or the explicitly diagnostic scripted contact controller. Inside contact support, it constructs policy, local, and uniform candidate actions, selects

$$
a^*=\arg\max_{a\in\mathcal C(s)}U_\eta(s,g,a),
$$

executes $a^*$ for exactly five steps, and replans. The ordinary policy does not receive gradients from the rank head in Stages A, C, or D; DCC remains its objective. This prevents unsupported contact-head gradients from changing the approach policy, while the phase-gated wrapper uses the ranker only where its data are collected.

The scripted-reach Stage-D cell is a diagnostic upper bound and is not a fair final baseline comparison. The policy-reach cell is the deployable comparison.

## Four stages

| Stage | Budget | Purpose | Actor use of rank head |
|---|---:|---|---|
| A | 100k | Validate sparse success, held-out logging, repeat-5/best-progress causal probe, and corrected metric keys | None |
| B | 100k | Four-condition oracle decomposition: policy/scripted anchors crossed with repeat 1/5 | None; no ranker |
| C | 250k | Train the chunk ranker and require held-out plus independent causal generalization | None |
| D | 1M | Closed-loop phase-gated rank selection with aligned five-step execution | Candidate selection only; no rank-head policy gradient |

Each stage has four runs: Tasks 5/8 crossed with seeds 5/6. Stage D is submitted twice, once for each reach mode.

## Stage gates

### Stage A

- benchmark success availability at least $0.99$;
- held-out metrics present;
- independent probe records action repeat $5$;
- aligned causal Spearman is present and finite.

### Stage B

From scripted contact with repeat $5$:

- oracle best-candidate benchmark success at least $0.50$;
- oracle gain over random candidate selection at least $0.20$.

Policy-anchor reach rate is reported separately. Failure means the candidate class or horizon must be changed before learning a ranker.

### Stage C

- held-out score/outcome Spearman at least $0.30$;
- held-out pairwise accuracy at least $0.65$;
- held-out action-permutation Spearman drop at least $0.20$;
- independent repeat-aligned score/progress Spearman at least $0.30$.

### Stage D

- first-1M success AUC gain over matched CRTR at least $0.05$;
- final-20%-to-peak retention at least $0.80$;
- episode-level contact reach rate at least $0.50$;
- held-out and independent aligned Spearman both at least $0.30$.

Only the policy-reach Stage-D cell is eligible for 8M promotion.

## Implementation map

| File | Change |
|---|---|
| `contrastive/counterfactual_outcomes.py` | Shared progress, interaction phase, sparse-reward and proxy success semantics |
| `contrastive/counterfactual_ranking.py` | Benchmark/proxy confusion metrics, corrected permutation metric, oracle summaries |
| `contrastive/action_ranking_diagnostics.py` | Configurable repeated intervention, best-over-horizon progress, sparse success, phase metrics |
| `contrastive/phase_gated_control.py` | Acme-compatible reach/contact wrapper and aligned chunk execution |
| `contrastive/continual_learning_decomposed.py` | Ranker can train without affecting the actor; direct counterfactual scoring API |
| `run_continual_contrastive.py` | Disjoint validation simulator, pre/post held-out metrics, four-condition oracle events, phase-gated train/eval actors |
| `contrastive/continual_config.py` | Backward-compatible stage controls |
| `experiment_configs_counterfactual_stages.py` | Stage A--D Tasks 5/8, seeds 5/6 grids |
| `scripts/evaluate_counterfactual_stages.py` | Stage-specific W&B promotion gates |
| `DRAFT_counterfactual_stages.sh` | Torch Slurm launcher |
| `tests/test_counterfactual_stages.py` | Sparse reward, permutation, actor chunk, and config checks |

All new switches default off or to the legacy semantics. Existing CFR-DCC, DCC, Bridge-DCC, RBC-DCC, DCC-SAC, and AC-DCC configurations retain their prior paths.

## W&B metrics

Important new keys include:

- `learner/counterfactual_rank/benchmark_success_fraction`
- `learner/counterfactual_rank/success_predicate_agreement`
- `learner/counterfactual_rank/train_pre/score_vs_outcome_spearman`
- `learner/counterfactual_rank/train_post/score_vs_outcome_spearman`
- `learner/counterfactual_rank/heldout_pre/score_vs_outcome_spearman`
- `learner/counterfactual_rank/heldout_post/score_vs_outcome_spearman`
- `learner/counterfactual_rank/heldout_post/action_permutation_spearman_drop`
- `learner/action_landscape/aligned_score_vs_progress_spearman`
- `learner/action_landscape/action_repeat`
- `learner/action_landscape/anchor_phase_contact`
- `learner/oracle/{policy|scripted_contact}/repeat{1|5}/best_success_fraction`
- `learner/oracle/{policy|scripted_contact}/repeat{1|5}/success_gain`
- `evaluator/phase_control/contact_episode_reach_rate`
- `evaluator/phase_control/contact_step_fraction`
- `evaluator/phase_control/first_contact_step_mean`
- `evaluator/phase_control/chunk_selections`

## Torch launch sequence

Run one stage at a time:

```bash
cd /scratch/yd2247/sgcrl
git pull --ff-only origin section3_done

COUNTERFACTUAL_STAGE=A sbatch DRAFT_counterfactual_stages.sh
python scripts/evaluate_counterfactual_stages.py --stage A

COUNTERFACTUAL_STAGE=B sbatch DRAFT_counterfactual_stages.sh
python scripts/evaluate_counterfactual_stages.py --stage B

COUNTERFACTUAL_STAGE=C sbatch DRAFT_counterfactual_stages.sh
python scripts/evaluate_counterfactual_stages.py --stage C
```

Only after Stage C passes:

```bash
COUNTERFACTUAL_STAGE=D COUNTERFACTUAL_REACH_MODE=scripted_contact \
  sbatch DRAFT_counterfactual_stages.sh
python scripts/evaluate_counterfactual_stages.py \
  --stage D --reach-mode scripted_contact

COUNTERFACTUAL_STAGE=D COUNTERFACTUAL_REACH_MODE=policy \
  sbatch DRAFT_counterfactual_stages.sh
python scripts/evaluate_counterfactual_stages.py \
  --stage D --reach-mode policy
```

The launcher uses one experiment per L40S GPU because exact-state simulator rollouts are CPU-heavy. It runs the simulator-restore and dependency-light preflights after activating the Torch `contrastive_rl` Conda environment.

## Validation

The implementation is validated with:

- Python compilation for every changed and new module;
- shell syntax checks for both launchers;
- exact four-cell enumeration for Stages A--C and both Stage-D reach modes;
- dependency-light sparse-reward, corrected permutation, grouped-buffer, oracle, and chunk-repeat checks;
- legacy counterfactual-ranking checks.

The full MuJoCo restore-step and environment success tests run on Torch during launcher preflight because Meta-World and the project environment wrappers are not installed in the lightweight local image.

## Limitations

- Exact simulator restoration and scripted reach are diagnostic privileges, not a general model-free algorithm.
- The first chunk class holds one action constant for five steps; Stage B explicitly tests whether that restricted class contains successful behavior.
- The contact gate uses known Sawyer observation coordinates. A general version needs a learned interaction-support classifier.
- The ranker still changes online. EMA/best-policy retention is intentionally deferred until causal held-out ranking passes.
- Stage A is a 100k instrumentation smoke test, not a performance comparison.
- No jobs are submitted by the implementation commit.
