# Paper Plan — Sparse-Reward Continual Goal-Conditioned RL

**Target venue.** NeurIPS 2026 (abstract May 4 AOE; full paper May 6 AOE).
**Working title (draft).** *Contrastive Goal-Conditioned Reinforcement Learning with Policy Decomposition and a Knowledge Pool for Sparse-Reward Continual Control.*
**Branch of record.** `section3_done` (SGCRL repository). BuilderBench results are appendix-only.

This document is the writing plan for the main paper. It is organised as an abstract draft followed by section-by-section narratives describing the argument, evidence, and structure we intend to produce in the manuscript. Citation keys follow `docs/citations.md`.

---

## 1. Abstract (draft)

Continual reinforcement learning requires an agent to acquire a growing set of skills without forgetting previously mastered ones, and most existing benchmarks and algorithms attack this problem with dense, hand-crafted reward functions that encode per-task expert knowledge. Such reward shaping is laborious, brittle across environments, and arguably sidesteps the central representational challenge: learning policies and value functions whose structure transfers across a task sequence. We study continual reinforcement learning in the *sparse-reward* regime, where the reward signal is the self-supervised event of reaching a commanded goal, and we instantiate this setting on a ten-task Meta-World Sawyer manipulation sequence with eight million environment steps per task and no dense shaping. Within this setting we propose a framework that combines three ingredients: (i) a contrastive goal-conditioned critic that represents reachability through a dual encoder trained with InfoNCE, (ii) a policy decomposition that expresses the per-task policy as a frozen base plus a linear combination of knowledge vectors drawn from a bounded knowledge pool, and (iii) an offline-to-online hard-negative mining scheme that reuses goals from prior tasks as additional, per-anchor-selected negatives in the contrastive objective. Across a nine-cell ablation over actor and critic evolution strategies, we find that the combination of a decomposed actor with a persistent contrastive critic delivers the strongest forward transfer, that actor-side representational quality is the primary bottleneck in long task sequences, and that naïve cross-task negatives degrade learning whereas principled per-anchor hard mining yields consistent improvements. The results position contrastive representation learning as a natural substrate for sparse-reward continual control and separate the respective contributions of critic persistence and actor decomposition.

---

## 2. Contributions

We structure the manuscript around the following contributions, each stated in a single sentence, most important first, with abbreviations defined on first use.

1. **Problem setting.** We introduce and empirically instantiate *sparse-reward continual goal-conditioned reinforcement learning* on a ten-task Meta-World Sawyer manipulation sequence, where the only reward is the self-supervised goal-reaching event `r_g(s_t,a_t) = (1-γ)·p(s_{t+1}=g | s_t, a_t)`, removing the hand-crafted dense rewards that previous continual reinforcement learning (CRL) benchmarks rely on.
2. **Framework.** We propose a framework that combines a dual-encoder contrastive goal-conditioned RL (GCRL) critic `f(s,a,g) = φ(s,a)^⊤ψ(g)` trained with the InfoNCE objective, a policy decomposition of the form `θ′ = θ_base + Σ_j α_j v_j + v_k` managed by a bounded cosine-similarity knowledge pool, and an offline-to-online hard-weighted negative bank that draws per-anchor hard negatives from the buffers of previously seen tasks.
3. **Empirical findings.** We isolate, through a 9-cell ablation over actor and critic evolution modes (`{reset, persistent, decomposed} × {reset, persistent, decomposed}`), that actor-side decomposition combined with a persistent contrastive critic is the strongest configuration, that critic representation retention dominates forward transfer in line with prior SAC-based evidence, and that actor plasticity loss becomes the primary bottleneck over long sequences.
4. **Negative-sampling analysis.** We show that naïve cross-task negatives are trivially separable and *hurt* the contrastive objective in multi-region workspaces, whereas per-anchor top-K hard mining with logit down-weighting (`w_bank = 0.3`) converts the offline task history into a useful signal.

---

## 3. Introduction

### 3.1 Motivation

Continual reinforcement learning (CRL) concerns an agent that faces a sequence of tasks `M^(1), …, M^(N)`, each a Markov decision process, and must learn the current task while preserving competence on earlier ones ([Khetarpal et al., 2022]). Existing evaluation pipelines, including Continual World ([Wołczyk et al., 2021]) and the CKA-RL benchmark ([Hu et al., 2025]), are built on the dense, hand-crafted reward functions that Meta-World ships with ([Yu et al., 2020]). These dense rewards entangle two very different questions: (a) whether an algorithm can preserve and reuse knowledge across tasks, and (b) whether it can cope with the engineering choices baked into each per-task reward. In practice, results reported under dense rewards frequently fail to transfer to settings where the reward is only the success event, which is the regime most faithful to deployment on real robots and to the goal-conditioned formulation of reinforcement learning ([Liu et al., 2025]).

We therefore study continual reinforcement learning in the *sparse-reward, goal-conditioned* regime. The agent receives reward only when the commanded goal is reached; it must discover its own dense learning signal through contrastive classification of future states ([Eysenbach et al., 2022]). This shift of setting is not cosmetic. It removes a substantial source of inter-task variance (differences in reward scaling), it eliminates per-task reward engineering, and it forces the critic to represent reachability structure rather than task-specific scalar returns. We argue that this representational shift is precisely what a continual-learning algorithm should exploit.

### 3.2 Framing of the contribution

We develop an algorithmic framework that instantiates three complementary ideas inside this setting. First, a contrastive goal-conditioned critic whose dual encoders are trained with the InfoNCE objective ([van den Oord et al., 2018]; [Eysenbach et al., 2022]). Second, a policy decomposition that expresses the per-task policy as a frozen base plus a linear combination of knowledge vectors drawn from a bounded pool. Policy decomposition and bounded knowledge pools have been previously studied in the continual RL literature, most recently by [Hu et al., 2025], and we adopt them as building blocks rather than as objects of study in their own right. Third, an offline-to-online negative-sampling scheme that converts the replay buffers of earlier tasks into additional contrastive negatives for the current task, after filtering them per-anchor to retain only those that exert meaningful gradient pressure on the critic.

We explicitly do not position this work as an extension of any single prior algorithm. The thesis is that the *combination* — contrastive GCRL, policy decomposition with a knowledge pool, and a principled cross-task negative bank — produces a coherent approach to sparse-reward CRL that existing components, applied in isolation, do not.

### 3.3 Summary of findings

We evaluate the framework on a ten-task Meta-World Sawyer manipulation sequence (`hammer, push_wall, faucet_close, push_back, stick_pull, handle_press_side, push, shelf_place, window_close, peg_unplug_side`) with sparse rewards, eight million environment steps per task, and five random seeds per configuration. A full 9-cell grid over `actor ∈ {reset, persistent, decomposed} × critic ∈ {reset, persistent, decomposed}` cleanly separates the contributions of the two mechanisms. The strongest configuration uses a decomposed actor with a persistent contrastive critic. The 1000-layer architectural backbone of [Wang et al., 2025] is retained throughout.

---

## 4. Related Work

### 4.1 Contrastive and goal-conditioned reinforcement learning

The reduction of goal-conditioned value learning to contrastive classification was formalised by [Eysenbach et al., 2022], who showed that an InfoNCE-trained dual-encoder `φ(s,a)^⊤ψ(g)` is consistent with the discounted visitation objective `r_g(s_t,a_t) = (1-γ)·p(s_{t+1}=g|s_t,a_t)`. [Liu et al., 2025] demonstrated that skills and exploration emerge from this formulation when only a single goal is commanded per episode, which is the operating point we adopt. [Wang et al., 2025] pushed contrastive reinforcement learning to very deep residual networks with layer normalisation and Swish activations, establishing the 1000-layer backbone on which our implementation builds. [Bortkiewicz et al., 2025] released the JaxGCRL codebase that makes such experiments tractable, and [Myers et al., 2025] analysed the exploration mechanisms that emerge in this regime. InfoNCE itself traces back to [van den Oord et al., 2018]; the broader practice of maintaining memory banks of negatives appears in the representation-learning literature in [He et al., 2020], and the importance of hard-negative selection has been studied by [Robinson et al., 2021]. Hindsight relabelling ([Andrychowicz et al., 2017]) provides the positive-pair mechanism we inherit.

### 4.2 Continual reinforcement learning

A broad survey of CRL is given by [Khetarpal et al., 2022]. On the benchmark side, [Wołczyk et al., 2021] introduced Continual World, a ten-task Meta-World sequence that has become a standard testbed, and [Wołczyk et al., 2022] showed that in soft actor-critic (SAC, [Haarnoja et al., 2018]) transferring the critic gives a larger forward-transfer benefit than transferring the actor. Their analysis is conducted under dense rewards and with a scalar Q-function; we revisit the same question under a contrastive critic and in the sparse-reward regime. Replay-based approaches such as CLEAR ([Rolnick et al., 2019]) and regularisation approaches such as EWC ([Kirkpatrick et al., 2017]) remain orthogonal baselines.

### 4.3 Policy decomposition and knowledge pools

Decomposing the per-task policy as a frozen base plus a set of additive knowledge vectors has recently been proposed in CKA-RL ([Hu et al., 2025]) for SAC-based continual control, where a bounded pool merges the most cosine-similar pair whenever it exceeds a maximum size. We adopt the same mathematical form, `θ′ = θ_base + Σ_j α_j v_j + v_k` with `α = softmax(β·α_scale)` and merge rule `|V| > K_max ⇒ merge arg-max-similarity pair`, because it is simple, parameter-efficient, and composable with the contrastive critic; the present work is not an extension of CKA-RL and we do not inherit its SAC back-end, its reliance on dense rewards, or its critic-reset protocol. The broader literature on modular and compositional policies — including progressive networks, parameter-efficient adapters, and mixture-of-experts policies — provides alternative realisations of the same underlying idea.

### 4.4 Plasticity loss and representation collapse

Neural networks trained on non-stationary data streams lose plasticity ([Dohare et al., 2024]). Relevant diagnostic tools include the primacy bias characterised by [Nikishin et al., 2022] and the dormant-neuron analysis with reset protocol of [Sokar et al., 2023], from which we adopt the `τ = 0.025` threshold used in our actor diagnostics. Representational collapse at the feature level is diagnosed via the neural-collapse statistics (NRC1, NRC2) of [Papyan et al., 2020]. These metrics are used in this paper as analysis instruments; our algorithm does not introduce new plasticity regularisers.

---

## 5. Problem Setting

### 5.1 Sparse-reward continual goal-conditioned RL

We consider a sequence of tasks `{M^(k)}_{k=0}^{N-1}` sharing a common state space `S`, action space `A`, discount factor `γ`, and robot embodiment. Each task specifies a transition kernel `p^(k)` and a goal distribution `p_g^(k)` over a task-specific goal manifold embedded in `S`. The reward is the self-supervised goal-reaching event

`r_g(s_t, a_t) = (1 − γ) · p(s_{t+1} = g | s_t, a_t)`,

which, up to a constant, is equivalent to the indicator `1[s_{t+1} = g]` discounted by `γ^t`. The agent observes `(s, g)` and selects `a`. No dense reward shaping is available at any point during training.

### 5.2 Learning objective

At task `k`, the agent optimises

`max_π E_{s_0 ~ p_0^{(k)}, g ~ p_g^{(k)}, π} Σ_t γ^t r_g(s_t, a_t)`,

subject to stability constraints on performance on tasks `0, …, k-1`. We measure success with two families of metrics: intra-task learning curves (success rate over environment steps) and cross-task forgetting (success rate on tasks `0, …, k-1` after training on task `k`). No dense-reward proxy is used for evaluation.

### 5.3 Benchmark instantiation

We instantiate this setting on ten Meta-World Sawyer manipulation tasks in the canonical Continual-World ordering (`hammer, push_wall, faucet_close, push_back, stick_pull, handle_press_side, push, shelf_place, window_close, peg_unplug_side`). Goals are extracted from the last three positional coordinates of the Meta-World observation and are interpreted per task (object position, handle position, or nail position). Each task is run for eight million environment steps; each configuration is run with five seeds. All experiments use sparse rewards; dense-reward baselines are run only for the purpose of comparison with previous work and are reported separately.

---

## 6. Method

We describe the three components of the framework in the order in which they appear in the training loop: the contrastive critic (Section 6.1), the policy decomposition and knowledge pool (Section 6.2), and the offline-to-online hard-weighted negative bank (Section 6.3). Algorithm 1 in the appendix gives the full pseudocode; `docs/algorithm_pseudocode.md` is the reference implementation specification.

### 6.1 Contrastive goal-conditioned critic

Following [Eysenbach et al., 2022] and [Liu et al., 2025], we train a dual-encoder critic

`f(s, a, g) = φ(s, a)^⊤ ψ(g)`,

where `φ : S × A → R^d` and `ψ : S → R^d` are realised by residual MLPs (width 1024, depth 4, LayerNorm, Swish activations) following the architecture of [Wang et al., 2025]. Positives are drawn by hindsight relabelling ([Andrychowicz et al., 2017]); negatives are in-batch goals. The critic loss is the InfoNCE objective

`L_InfoNCE = E_D [ −log ( exp f(s,a,g^+) / ( exp f(s,a,g^+) + Σ_j exp f(s,a,g^-_j) ) ) ]`,

implemented as the `contrastive_cpc` variant that also applies a logsumexp regulariser. The actor is trained by maximising the diagonal of the critic score matrix,

`max_θ E_{(s,g)~D} [ φ(s, π_θ(s, g))^⊤ ψ(g) ]`,

with an adaptive entropy coefficient tuned by the SAC dual gradient ([Haarnoja et al., 2018]) with `target_entropy = −2.0`. A `NormalTanh` head, preceded by an additional LayerNorm and Swish layer, outputs stochastic actions.

### 6.2 Policy decomposition and knowledge pool

For task `k > 0`, the policy parameters are expressed as

`θ′ = θ_base + Σ_{j=1}^{k−1} α_j v_j + v_k`,   `α = softmax(β · α_scale)`,

where `θ_base` is the base policy obtained at the end of task 0, `v_j` is the knowledge vector learned on task `j`, `v_k` is the current task's knowledge vector, and `β, α_scale` are learnable blending parameters. By default only the output head (mean and log-std) is decomposed (`adapt_heads_only = true`), while the encoder body is fed gradients (`encoder_from_base = false`) so that representation drift is possible without proliferating knowledge vectors. The pool `V = {v_1, …, v_{k-1}}` is bounded by `K_max = 5`; whenever `|V| > K_max`, the pair with the highest cosine similarity is merged by averaging, following the protocol of [Hu et al., 2025]. The same decomposition form can be applied to the critic encoders, and we expose this as an ablation cell.

### 6.3 Offline-to-online hard-weighted negative bank

During task `k`, the learner has access to the replay buffers of tasks `0, …, k-1`. We form a `NegativeBank` that stores, per task, up to `10,000` HER-relabelled goals, with a FIFO retention over at most `20` tasks. At each critic update, we sample a candidate pool of `1,024` goals from the bank and score them against each anchor `(s_i, a_i)` to form a score matrix `φ(s_i, a_i)^⊤ ψ(g_c)`. Per anchor, we retain the top `M = 256` candidates via `jax.lax.top_k`, and append the resulting logits to the in-batch contrast with a scalar down-weight `w_bank = 0.3`:

`extended_logits = concat( in_batch_logits, w_bank · bank_logits )`.

The categorical cross-entropy is taken over the extended logits, with the positive column unchanged. This design is motivated by a concrete failure mode of the vanilla variant, documented in `docs/negative_bank.md` and revisited in Section 7.4: because Meta-World tasks occupy disjoint workspace regions, unfiltered cross-task goals are trivially separable from current-task goals, so the critic achieves near-perfect categorical accuracy without learning a useful representation. Per-anchor top-K mining selects negatives that lie near the current decision boundary, and logit down-weighting limits damage from residual false negatives.

### 6.4 Training loop

Training proceeds in two phases: a base phase (task 0) that trains the full policy and critic with no decomposition, and a continual phase (tasks 1 to N−1) that applies the decomposition to the actor and one of three evolution modes to the critic. The actor auto-reset mechanism developed in earlier iterations of this work is retained as a diagnostic but is disabled by default. The learner and driver are implemented in JAX/Haiku in the `section3_done` branch; SLURM orchestration uses the frozen `draft_3.sh` environment recipe, and the batch launcher `draft_4.sh` together with `experiment_configs.py` manages the 9 × 5 = 45 full-scale runs.

---

## 7. Experiments

### 7.1 Experimental setup

All experiments are run on the ten-task Meta-World Sawyer sequence described in Section 5.3. Each configuration uses five random seeds and eight million environment steps per task. Reported metrics are mean and standard error across seeds. The full 9-cell ablation grid is

| Actor evolution \\ Critic evolution | reset | persistent | decomposed |
|---|---|---|---|
| reset | (a) | (b) | (c) |
| persistent | (d) | (e) | (f) |
| decomposed | (g) | (h) | (i) |

Cell (e) corresponds to a naïve full-network carryover without decomposition, and cell (i) corresponds to a fully decomposed actor and critic. Cell (h), decomposed actor with persistent contrastive critic, is our proposed default.

### 7.2 Core result: learning curves and forgetting

Figure 1 (to be produced) shows, for each cell in the 9-grid, the per-task success curves stacked along the task sequence; Figure 2 shows the post-sequence forgetting matrix. The predicted pattern is (a) a substantial gap between any configuration using a contrastive critic and a sparse-SAC baseline, (b) cell (h) as the dominant configuration, (c) decomposed-actor cells dominating reset-actor cells, and (d) persistent-critic cells dominating reset-critic cells, consistent with [Wołczyk et al., 2022] in the dense-reward SAC regime and extending their finding to the contrastive sparse-reward regime.

### 7.3 Forward and backward transfer

We report both forward transfer (learning speed on task `k` given the continual-learning state at the end of task `k-1`) and backward transfer (change in performance on tasks `0, …, k-1` after training on task `k`). Forward transfer is expected to be dominated by the critic (persistent > decomposed > reset), consistent with the critic carrying reachability information that is task-agnostic. Backward transfer is expected to be dominated by the actor decomposition, since the knowledge pool and the frozen base provide the only mechanism for protecting prior-task behaviour.

### 7.4 The negative bank ablation

We compare three settings of `neg_bank_mode`: `off`, `vanilla`, and `hard_weighted`. The vanilla variant is expected to slow or reverse learning in the early continual tasks because of the workspace-separability problem described in Section 6.3. The hard-weighted variant is expected to match or improve on `off` uniformly. Metrics reported are `categorical_accuracy` (in-batch), `bank/extended_categorical_accuracy`, `bank/logits_mean`, and `bank/logits_max`. These provide a diagnostic chain from raw negative quality (`bank/logits_mean`) to final effect on the softmax (`extended_categorical_accuracy`) to task-level success.

### 7.5 Plasticity diagnostics

We report four actor-side plasticity metrics across the task sequence: dormant-neuron ratio at threshold `τ = 0.025` ([Sokar et al., 2023]), neural-collapse statistics NRC1 and NRC2 ([Papyan et al., 2020]), feature rank, and entropy. The hypothesis is that the reset-actor cells exhibit the smallest dormancy accumulation but forfeit backward transfer, whereas the persistent-actor cells accumulate dormancy most rapidly; the decomposed-actor cells should occupy an intermediate point and should show the clearest correlation between delayed performance jumps on late tasks and shifts in actor-feature rank.

### 7.6 Robustness

We report sensitivity to `K_max ∈ {3, 5, 8}`, `w_bank ∈ {0.1, 0.3, 0.5}`, and `target_entropy ∈ {−1.5, −2.0, −2.5}`, around the default cell (h). Full details of configurations are in `docs/batch_experiments.md` and `experiment_configs.py`.

### 7.7 Comparisons to prior work

We additionally report a sparse-reward SAC baseline (full actor and critic reset each task), a dense-reward SAC baseline (for reference with published Continual-World and CKA-RL numbers), and a sparse-reward CKA-RL re-implementation (decomposed actor, SAC critic, critic reset each task) to isolate the contribution of the contrastive critic under matched decomposition machinery.

---

## 8. Discussion

### 8.1 Why contrastive representations help in continual settings

The contrastive critic learns a representation of reachability that is invariant to the particular scalar reward of a task. This invariance is what makes critic persistence work: where an SAC critic would have to be reset to avoid inheriting stale Q-values scaled to a previous task's reward, a contrastive critic can, in principle, be fine-tuned across tasks. Our results quantify the size of this effect in sparse-reward, goal-conditioned manipulation.

### 8.2 The role of decomposition

Decomposition, as deployed here, is a parameter-efficient mechanism for protecting past-task behaviour. It is not the reason contrastive continual learning works; rather, it is a compatible mechanism for turning the representation retained by the critic into retained behaviour in the actor. Prior work ([Hu et al., 2025]) has studied decomposition in SAC-based continual RL; our contribution here is not decomposition per se but the demonstration that the technique is useful *on top of* a contrastive critic in the sparse-reward regime.

### 8.3 Limitations

Our benchmark uses a single robot embodiment, a fixed task ordering, and ten tasks; longer sequences and task orderings are explored partially via the 20-task (double-pass) setting but not exhaustively. The framework inherits the dependence of contrastive GCRL on the availability of future states for relabelling, which restricts the method to episodic Markovian settings. We do not study language-conditioned or image-based observations.

---

## 9. Appendices (planned)

- **A. Algorithm pseudocode.** Full version of Algorithm 1, mirroring `docs/algorithm_pseudocode.md`.
- **B. Hyperparameters.** Complete tables for actor, critic, negative bank, decomposition.
- **C. Architectural details.** Residual MLP specification, `NormalTanh` head, LayerNorm placement.
- **D. Extended plasticity metrics.** Full curves for all 9 cells.
- **E. BuilderBench preliminaries.** Single-task and two-task validation runs; noted as preliminary because adaptive entropy, the negative bank, and actor auto-reset are not yet plumbed into the BuilderBench driver.
- **F. Compute and reproducibility.** SLURM scripts, seeds, W&B project, checkpoint paths.

---

## 10. Writing schedule

| Deliverable | Owner | Due |
|---|---|---|
| Full ablation run (9 × 5, 10 tasks × 8M) launched on NYUAD HPC | Yumi | Apr 22 |
| First-pass figures (learning curves, forgetting matrix, bank diagnostics) | Yumi | Apr 28 |
| Introduction + Problem Setting + Method drafts | Yumi | Apr 30 |
| Experiments + Discussion drafts | Yumi | May 3 |
| Full internal review with Prof. Ross | Yumi + Keith | May 4 AM |
| Abstract submission | — | May 4 AOE |
| Final polish, appendices, final submission | Yumi | May 6 AOE |

---

## 11. Open questions for Prof. Ross

1. Whether the 9-cell ablation should be the headline figure or whether cell (h) alone should carry the narrative with the other cells deferred to an appendix.
2. Whether the dense-reward SAC and dense-reward CKA-RL numbers should be shown alongside our sparse results in the main paper, or only as an appendix calibration.
3. Whether the 20-task stress test belongs in the main experiments or as an appendix stress test.
4. Whether the negative-bank analysis (Section 7.4) merits a standalone narrative slot given how cleanly it isolates a qualitative failure mode of naïve cross-task negatives.
