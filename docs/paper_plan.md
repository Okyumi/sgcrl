# NeurIPS 2026 Paper — Writing Plan & Bulleted Outline

**Working title:** *Continual Goal-Conditioned Contrastive RL with Knowledge Decomposition*
(Placeholder — a method name like **CGCL** for "Continual Goal-Conditioned Contrastive Learning," or **CoCReL**, or **PoolCRL** can be chosen later.)

**Deadline:** Abstract May 4 AOE — Full paper May 6 AOE
**Authors:** Yumi, Zixuan, Prof. Keith W. Ross (NYU Abu Dhabi)

---

## Framing (one-paragraph elevator pitch)

We introduce **continual goal-conditioned contrastive RL in a sparse-reward setting** — a new problem setup in which an agent sequentially faces MetaWorld manipulation tasks whose only supervision is a binary reach/no-reach signal. On top of contrastive goal-conditioned RL (GCRL), which turns goal-reaching into a classification problem and thereby sidesteps reward engineering, we layer CKA-RL-style **knowledge decomposition with a bounded pool**: after a base task, every subsequent task learns a lightweight delta on top of a frozen base; deltas are stored, blended, and merged in a bounded pool. A full 3 × 3 ablation of actor / critic transfer regimes (reset, persistent, knowledge-decomposed) plus a novel **offline-to-online negative bank** (previous tasks' HER-relabeled goals as hard negatives for InfoNCE) reveals what transfers across tasks and what does not. We complement this with a **representation-level analysis** of both actor and critic across training — feature rank, Neural Rank Collapse (NRC1/NRC2), Gini index, dormant-neuron ratio, and weight norms — showing that actor plasticity, not critic degradation, is the primary bottleneck in continual RL.

---

## Why this framing is defensible (differentiation from nearest neighbours)

| Nearest neighbour | What they do | What we do differently |
|---|---|---|
| Contrastive RL / SGCRL (Eysenbach 2022; Liu 2025) | Contrastive critic on a **single** goal-reaching task | Continual sequence; adds knowledge decomposition + pool; adds offline-to-online negative bank |
| CKA-RL (Kaplanis 2024) | Actor knowledge decomposition on continual **SAC with dense reward**; critic reset every task | Contrastive critic (no reward engineering); **critic transfer** ablation; both actor and critic decomposition |
| Continual World (Wolczyk 2021) / Disentangling Transfer (Wolczyk 2022) | Continual manipulation benchmark + transfer diagnostics with SAC | Sparse reward regime; contrastive critic; richer representation diagnostics |
| Scaling CRL (1000-layer paper) | Single-task contrastive RL at huge scale | Continual; decomposed; representation analysis over task boundaries |
| Primacy Bias (Nikishin 2022), ReDo (Sokar 2023), plasticity loss lit. | Periodic full resets or per-neuron resurrection on single-task | We study plasticity loss *in the continual regime specifically*, tie it to actor-side metrics, and propose a dormancy-triggered reset that preserves ablation integrity |

Our novelty is not a single new loss or a single new pool mechanism. It is the **composition**: sparse-reward continual setting × contrastive critic × knowledge decomposition × offline-to-online negative mining × representation-level analysis. Any one of these pieces alone is a minor contribution; together they give a clean story with clearly differentiable ablations.

---

## Writing plan (section by section, time budget)

| Section | Target length | Writer | Target deadline | Status |
|---|---|---|---|---|
| 0. Title + Abstract (150 words) | 0.5 pg | Yumi | May 3 | stub |
| 1. Introduction (contributions, setup, TL;DR figure) | 1.5 pg | Yumi | May 2 | outline below |
| 2. Related Work | 1 pg | Yumi | May 3 | outline below |
| 3. Preliminaries (MDP, GCRL, InfoNCE, CKA-RL) | 0.75 pg | Yumi | May 1 | skeleton exists in `algorithm_pseudocode.md` |
| 4. Problem Setup (sparse-reward continual GCRL) | 0.5 pg | Yumi | May 1 | outline below |
| 5. Method (9-config grid + pool + negative bank) | 2 pg | Yumi + Zixuan | May 3 | code + pseudocode exist |
| 6. Experimental Setup (benchmark, baselines, seeds, metrics) | 0.75 pg | Yumi | May 3 | `batch_experiments.md` + `draft_4.sh` exist |
| 7. Results: Main comparison (9-config grid) | 1.5 pg | Yumi | May 4 | batches running |
| 8. Results: Representation analysis | 1 pg | Yumi | May 5 | metrics in place |
| 9. Results: Negative-bank ablation | 0.5 pg | Yumi | May 5 | variant implemented |
| 10. Discussion + Limitations | 0.5 pg | Yumi + Keith | May 5 | — |
| 11. Conclusion | 0.25 pg | Yumi | May 5 | — |
| Appendix: hyperparams, extra curves, 20-task, BuilderBench | 3 pg | Yumi | May 6 | — |
| **TOTAL main body** | **10 pg** | | | |

The NeurIPS main-track page limit is 9 pages + references; appendix is unlimited. Above budget targets 10 pages assuming some trimming during editing.

---

## TL;DR figure (front-page headline result)

A single figure, on page 1 or 2, that conveys the whole paper at a glance. Proposed contents:

- **Panel A (main result).** 10-task Meta-World sparse-reward sequence. Four curves: (i) Contrastive GCRL + decomposed actor + persistent critic (ours, best); (ii) Contrastive GCRL with everything reset (ablation baseline); (iii) SAC with sparse reward (floor); (iv) SAC with dense reward (ceiling reference). y = mean success rate across tasks seen, x = env steps.
- **Panel B (representation).** Actor dormant ratio over time for a "healthy" seed vs a "stuck" seed of the vanilla (no-adaptive-entropy) variant, showing the plasticity-loss mechanism we discovered.
- **Panel C (forward transfer heatmap).** Evaluation matrix M[i,j] = success on task i after training through task j. Columns show how performance on earlier tasks behaves when later tasks are trained — and whether newer tasks learn faster given prior knowledge.

---

## Section-by-section bullets

### 1. Introduction (1.5 pg)

**Opening hook**
- Continual RL is the right abstraction for real-world deployed agents: tasks arrive sequentially, the agent cannot revisit old environments at will, and replay from past tasks is always available offline.
- Two pervasive obstacles make continual RL hard in practice: (i) **reward engineering** — every new task demands a shaped reward, making per-task design a bottleneck — and (ii) **catastrophic forgetting / plasticity loss** — later training destroys earlier skills or exhausts the network's capacity to learn new ones.
- Goal-conditioned RL (GCRL) partly addresses (i): specify a target state, not a reward function.
- Contrastive GCRL goes further — it reduces goal-reaching to a classification problem over InfoNCE and learns reachability representations (φ, ψ) that, unlike scalar Q functions, are intrinsically shared across tasks on the same robot.

**Gap we address**
- Despite its natural fit, contrastive GCRL has never been studied in a sparse-reward continual setting.
- The most relevant continual-learning framework (CKA-RL) decomposes the actor into base + knowledge pool but **resets the SAC critic every task** because scalar Q-values do not transfer.
- But a contrastive critic is not a scalar Q: it factors into two encoders that represent state-action reachability and goal identity, and these can — and should — transfer across tasks.

**What we do**
- Build **Continual Goal-Conditioned Contrastive RL with Knowledge Decomposition**: at each task, maintain a base policy θ_base, a frozen or slowly evolving base critic (φ_base, ψ_base), a shared knowledge pool of per-task deltas, and adaptive entropy.
- Run a full 3 × 3 ablation over actor transfer × critic transfer regimes (reset / persistent / knowledge-decomposed), giving nine configurations that each isolate a specific mechanism.
- Introduce an **offline-to-online negative bank**: use HER-relabeled goals from past replay buffers as extra hard negatives in the current task's InfoNCE contrast, with a principled hard-mining and down-weighting scheme to prevent trivial cross-task shortcuts.
- Instrument both actor and critic with **representation diagnostics** — feature rank, NRC1, NRC2, Gini, dormant ratio, weight norms — captured throughout training at both frequent and occasional sampling rates.

**Contributions** (as promised to the supervisor; 3–4 sentences, most important first, all abbreviations defined)
1. We formalise **continual goal-conditioned contrastive reinforcement learning (RL) under sparse rewards** as a new problem setup — a sequence of MetaWorld manipulation tasks in which the only supervision is a binary goal-reaching signal — and introduce an algorithm that combines contrastive goal-conditioned RL (GCRL) with knowledge decomposition inspired by Continual Knowledge Adaptation RL (CKA-RL), enabling sequential skill acquisition without per-task reward engineering.
2. Through a systematic ablation across nine actor–critic transfer configurations (reset, persistent, and knowledge-decomposed), we find that a knowledge-decomposed actor paired with a persistent critic achieves the best performance, and that retaining critic representations across tasks is the primary driver of forward transfer — more so than actor-side knowledge reuse.
3. We propose an **offline-to-online negative bank** that uses hindsight-relabeled goals from previous tasks' replay buffers as hard negatives for the current task's InfoNCE contrast, with a principled hard-mining and down-weighting scheme that prevents trivial cross-task shortcuts.
4. We provide a representation-level analysis of continual contrastive RL, tracking feature rank, Neural Rank Collapse (NRC), dormant neuron ratio, and weight norms throughout training, showing that successful task transitions are driven by rapid shifts in actor expressivity rather than gradual critic adaptation, and that actor plasticity loss is the primary bottleneck in longer task sequences.

### 2. Related Work (1 pg)

Organised into four short paragraphs:

**Contrastive goal-conditioned RL.** Eysenbach et al. (2022) showed that learning the discounted state occupancy measure as a critic reduces goal-reaching to a binary classification problem, training (φ, ψ) via InfoNCE. Liu et al. (2025, "Single Goal Is All You Need") extended this to the case of a single fixed goal and identified an "overexploitation" failure mode on sparse-reward manipulation. Scaling laws for contrastive RL have been studied at billion-parameter scale (Wang et al., "scaling CRL"). None of these works address sequential task learning.

**Continual reinforcement learning.** Continual World (Wolczyk et al., 2021) established Meta-World task sequences as the de-facto benchmark for continual manipulation. Wolczyk et al. (2022) disentangled forward and backward transfer and showed that critic transfer often matters more than actor transfer in SAC-based continual learning. Methods include knowledge pools with task-identification (PackNet, CLEAR), parameter-efficient adaptation (LoRA-like deltas), and experience replay (CLEAR, ER). Continual Knowledge Adaptation RL (CKA-RL; Kaplanis et al., 2024) is the closest methodological neighbour: it decomposes the actor head via learnable deltas with a softmax-blended pool, but resets the SAC critic because scalar Q does not transfer. We replace the SAC critic with a contrastive dual-encoder critic (φ, ψ) that transfers naturally and extend the decomposition to both actor and critic.

**Plasticity loss and network resets in RL.** Dohare et al. (2024, Nature) documented plasticity loss in deep RL. Nikishin et al. (2022, "Primacy Bias") showed that periodic full network resets can improve long-horizon training. Sokar et al. (2023, ReDo) proposed per-neuron resurrection based on a dormancy threshold. Lyle et al. (2024) and Abbas et al. (2026) tied plasticity loss to activation geometry. Our diagnostics adopt these metrics; we extend them to the actor side (most prior work focuses on critic) and tie plasticity loss specifically to seed-dependent variance in sparse-reward continual RL.

**Representation analysis in deep RL.** Neural Rank Collapse (Papyan et al., 2020; Zhu et al., 2021) was originally formulated for classification but has been imported into RL (Lyle et al.; He et al. on feature rank in DQN). We compute both NRC1 (subspace collapse toward action dimension) and NRC2 (alignment of hidden features with the final-layer weight's column space), separately for the actor trunk and critic encoders, across continual task boundaries.

### 3. Preliminaries (0.75 pg)

- **Notation.** Goal-reaching MDP (S, A, p, γ). Goal g ∈ G ⊆ S. Contrastive critic f(s, a, g) = φ(s,a)^⊤ ψ(g) trained with InfoNCE over batches of (s, a, g^+) tuples where g^+ is sampled from future states of the same trajectory (HER-style).
- **InfoNCE training objective** (equation).
- **Actor objective** (equation): π maximises E[φ(s, π(s, g))^⊤ ψ(g)] plus adaptive entropy (log α autotuned toward target entropy H_* = -½|A|).
- **CKA-RL actor decomposition** (equation): θ' = θ_base + Σ_j α_j v_j + v_k, α = softmax(β), pool {v_j} merged by cosine similarity when |pool| > K_max.
- **Critic-side analogue** (equation): (φ, ψ)' = (φ, ψ)_base + Σ_j α_j w_j + w_k.

### 4. Problem Setup (0.5 pg) — **the "new problem setup" Keith wants emphasised**

- **Sequence of N tasks** (τ_1, …, τ_N), each an MDP sharing state/action space but differing in goal distribution and physics.
- **Sparse reward.** r(s, a, g) = 1 iff ‖obs_to_goal(s) − g‖ < ε; 0 otherwise. No shaping. No demonstrations. Only replay.
- **Offline-to-online asymmetry** (key framing insight).
  - While learning task k (online phase), all earlier tasks 0, …, k−1 have already produced trajectories stored in replay buffers.
  - This is a natural offline-to-online structure that most continual-RL methods ignore: they either discard past replay (CKA-RL, progress-and-compress) or use it only as a regularisation / rehearsal term (CLEAR, ER).
  - Contrastive learning is a particularly good fit because its loss is naturally a function of *negatives* — more negatives, from more diverse distributions, can improve representation quality if they are used carefully.
- **Evaluation protocol.** After each task finishes, evaluate on all tasks 0..k. Two scalar summaries: (i) learning curve on task k (plasticity / forward transfer); (ii) retention on tasks 0..k−1 (stability / forgetting).

### 5. Method (2 pg) — core technical section

- **Section 5.1 Architecture.** ResidualMLP body (SGCRL / scaling-CRL default) for both actor and critic encoders φ, ψ. Policy head = NormalTanhDistribution. 1024-width, depth 4, LayerNorm + Swish. Bound actor trunk with extra LayerNorm + Swish so the head sees well-conditioned features (matches scaling-CRL).
- **Section 5.2 Base-phase training (task 0).** Standard contrastive GCRL. At task 0 the pool is empty and v_0 = 0, so the composed policy equals θ_base (the random init) exactly. Critic and actor both receive full gradients. This reduces *exactly* to SGCRL / scaling-CRL and we verify this empirically (single-task sanity check).
- **Section 5.3 Continual-phase training (tasks k ≥ 1).**
  - **Actor.** θ' = θ_base + Σ α_j v_j + v_k. Gradients flow through v_k only; v_j and θ_base are frozen additive constants. Head-only decomposition (adapt_heads_only=True, matching CKA-RL) stores only the `Normal/linear` delta in the pool; the body delta is folded into θ_base post-task so the encoder can evolve.
  - **Critic.** Three modes: (a) persistent — carry forward φ, ψ with optimizer state; (b) reset — fresh init each task; (c) CKA — (φ, ψ)' = (φ, ψ)_base + Σ α_j w_j + w_k.
  - **Adaptive entropy.** Per-task log α initialised at 0, autotuned toward target entropy H_* = -½|A| via SAC's dual gradient. Alpha is *not* part of the CKA state — each task has its own exploration schedule.
- **Section 5.4 Offline-to-online negative bank.**
  - **Motivation.** Natural offline-to-online structure: previous task's HER goals are free contrastive negatives.
  - **Vanilla variant.** Uniform random bank goals appended to the batch, labelled negative. We show empirically this *hurts*: MetaWorld workspaces are task-specific, so cross-task negatives are trivially separable from positives, categorical accuracy saturates near 1, gradients vanish, and representations stop improving.
  - **Principled variant (`hard_weighted`).** (i) Per-anchor hard-negative mining: score a candidate pool of C bank goals against each anchor's φ(s_i, a_i) and keep top-M (via `lax.top_k`). (ii) Down-weight bank logits by w_bank ∈ (0, 1]. Prevents cross-task shortcuts and false-negative leakage.
  - **Metrics.** bank/logits_mean, bank/logits_max, bank/extended_categorical_accuracy.
- **Section 5.5 Pool mechanics.** Fixed K_max. When |pool| > K_max, merge the two most cosine-similar vectors. Blending weights α = softmax(β · α_scale) with β_k ~ N(0, 0.01) learnable per-task and α_scale learnable scalar.
- **Section 5.6 Knowledge decomposition for both actor and critic.** When critic_mode='cka', we mirror the actor decomposition on (φ, ψ): per-task w_k, pool {w_j}, merge by cosine. The critic pool evolves with the actor pool.
- **Pseudocode box** (reuse `algorithm_pseudocode.md`).

### 6. Experimental Setup (0.75 pg)

- **Benchmark.** 10-task Meta-World manipulation sequence (hammer, push_wall, faucet_close, push_back, stick_pull, handle_press_side, push, shelf_place, window_close, peg_unplug_side). Sawyer robot, 4-DoF action space. Sparse reward: success defined by Meta-World's native `obj_to_target` threshold. 8M env steps per task. Optionally 20-task variant (two passes for forgetting/plasticity stress test).
- **Baselines.**
  - *Lower bounds:* **SAC (sparse reward)** — the canonical failure mode of sparse-reward RL. **Sparse SAC + dense SAC** contrast for reward-shaping reference.
  - *Continual baselines:* **CKA-RL** — actor-pool + reset SAC critic (their default). **Continual Contrastive (reset/reset)** — our method ablated to reset both networks each task (lower bound within our framework).
  - *Upper bounds / references:* **Single-task SGCRL** on each task independently (per-task ceiling for oracle transfer).
- **Seeds.** 5 seeds per configuration. Report mean ± std.
- **Evaluation.** Every 50K env steps, run 10 evaluation episodes with deterministic policy π(a|s,g) = argmax_a φ(s,a)^⊤ψ(g) via K-sample-argmax (K=10, optional) or the mean of the Normal distribution. After each task, evaluate on all previous tasks for retention.
- **Hyperparameters.** 1024-width ResidualMLP, depth 4, Swish+LayerNorm. target_entropy = -2.0. batch_size = 256. optimizer = Adam 3e-4. logsumexp penalty 0.01 (SGCRL) / 0.1 (scaling-CRL). random_goals = 0.5 for actor loss. Discount 0.99. See Appendix A for full table.
- **Infrastructure.** JAX/Haiku. NYU Abu Dhabi HPC, NVIDIA A100 GPUs. Batched experiments via SLURM job arrays with JAX memory fraction 0.45 (two runs per GPU). See `batch_experiments.md`.

### 7. Results — Main 9-config grid (1.5 pg)

**Figure 2 (the main plot).** 3 × 3 grid of learning curves, one cell per (actor_mode, critic_mode) combination. Each cell shows average success rate across all tasks seen, versus env steps across the whole 10-task sequence (80 M total). Columns = critic mode, rows = actor mode. Highlight the (CKA actor, persistent critic) cell as the best.

**Expected key findings** (aligned with Yumi's message to Keith):
- **F1.** Contrastive GCRL substantially outperforms SAC-sparse (which fails to learn on most tasks). This validates the contrastive GCRL choice for sparse-reward continual.
- **F2.** (CKA actor, persistent critic) > (CKA actor, CKA critic) > (reset actor, reset critic) > CKA-RL (which uses SAC+reset). Thus: (i) knowledge decomposition helps; (ii) persistent critic beats reset; (iii) critic-pool (CKA critic) is *not* necessary — persistent is simpler and works as well or better.
- **F3.** The critic's mode matters much more than the actor's mode. Compare same-row differences (actor fixed, critic varying) vs same-column differences (critic fixed, actor varying). This confirms Wolczyk et al.'s finding that critic transfer dominates — but in the new setting of contrastive critics.
- **F4.** Forward transfer, measured as the learning speed on task k vs a from-scratch baseline, is strongest for (CKA actor, persistent critic).

**Figure 3 (cross-task retention heatmap).** Matrix M[i, j] for each mode. Diagonal should be high (task-k performance at end of task k). Sub-diagonal shows forgetting. Compare reset-critic (expect strong forgetting) vs persistent-critic (expect mild forgetting).

**Table 1.** End-of-run summary per configuration: average success, forward transfer score, backward transfer score, wall-clock, parameter count.

### 8. Results — Representation analysis (1 pg)

**Figure 4.** Actor & critic diagnostics over task boundaries. Six small panels, one per metric (weight norm, NRC1, NRC2, Gini, feature rank, dormant ratio), x = env steps across all tasks with vertical lines at task boundaries, separate curves for actor and critic. Compare the best config (CKA+persistent) to the worst (reset+reset) and to CKA-RL.

**Expected findings** (aligned with Yumi's message):
- **F5.** Critic-side NRC1 and feature rank track success rate monotonically — expected and matches Lyle / Abbas results. Critic representation quality improves across tasks in persistent mode but *oscillates* in reset mode.
- **F6.** Actor-side metrics tell a different story: sudden drops in **actor feature rank** and spikes in **actor dormant ratio** at task boundaries correlate with performance collapse. The actor loses plasticity faster than the critic degrades.
- **F7.** In some runs (especially seeds with low initial weight norm) the actor gets trapped: weight norm stays ~50 instead of ~600, dormant ratio exceeds 10%, and success rate never recovers. This motivates our dormancy-triggered reset diagnostic (presented as a safety mechanism for diagnosis, not as a competing method — the user explicitly disabled it by default to preserve ablation integrity).
- **F8.** Adaptive entropy is essential — with fixed or zero α, the feedback loop between actor expressivity and critic contrast can collapse. α autotuning keeps α high (exploration) until the policy is entropy-rich enough to sustain learning.

### 9. Results — Negative-bank ablation (0.5 pg)

**Figure 5.** 3-way comparison on tasks 1–9 (bank is empty on task 0 by construction):
- Bank off (baseline)
- Bank = vanilla (uniform random cross-task goals, no weighting)
- Bank = hard_weighted (top-M by anchor-score, w_bank = 0.3)

**Expected findings:**
- **F9.** Vanilla bank *hurts* — categorical_accuracy saturates near 1.0 early, representations stop improving, success rate drops.
- **F10.** hard_weighted bank slightly improves critic representations on tasks 2+ (better feature rank, lower NRC2), and matches or modestly exceeds the off baseline on task-success rate.
- **F11.** This validates the offline-to-online intuition: previous replay *is* useful, but only with principled hard mining and down-weighting. A plain concat of old goals is counterproductive.

### 10. Discussion + Limitations (0.5 pg)

- The persistent-critic result generalises Wolczyk et al. (2022) from SAC to contrastive GCRL: in any continual RL, a *shared reachability representation* is the right cross-task object to preserve. Scalar Q-values are task-specific; two-encoder contrastive critics are not.
- The actor-plasticity bottleneck is novel. We suspect it stems from the combination of (a) contrastive critic's reliance on diverse exploration data and (b) the actor's dependency on critic gradients that themselves collapse when the actor's trajectories become uniform.
- **Limitations.**
  - Evaluated on MetaWorld Sawyer only; pixel-based (BuilderBench, robosuite) deferred to follow-up.
  - 5 seeds per config; a fuller seed budget would tighten confidence intervals.
  - Negative-bank variant is a first-attempt; further strategies (curriculum mining, diversity-aware sampling) are promising.
  - Pool merge heuristic is cosine-similarity; other clustering schemes unexplored.

### 11. Conclusion (0.25 pg)

Contrastive goal-conditioned RL is a natural fit for sparse-reward continual learning. When combined with knowledge decomposition and a persistent contrastive critic, it substantially outperforms standard SAC baselines and matches or beats CKA-RL's SAC-based actor decomposition. Cross-task representation sharing is driven by the critic; the actor's primary failure mode in continual RL is plasticity loss. We release code, pipelines, and representation diagnostics to make this setting reproducible.

### Appendices

- **A. Full hyperparameter table.**
- **B. Per-task learning curves** (Fig 2 expanded).
- **C. Additional representation analysis** (per-layer NRC decomposition, entropy-coefficient trajectories).
- **D. 20-task extension** (two passes of the sequence — stress test for forgetting).
- **E. Single-task sanity check** (verifying CGCRL reduces to SGCRL exactly at task 0 on single-task runs).
- **F. BuilderBench preliminary results** (different domain — language-conditioned block manipulation; demonstrates generality).
- **G. Ablations we did not have space for** (target_entropy sweep, K_max sweep, logsumexp_penalty sensitivity).
- **H. Reproducibility statement** (git hash, exact configs, WandB links).

---

## Experimental checklist (what absolutely needs to run before submission)

**Critical (must-have for main figures):**
1. 9-config grid × 5 seeds × 10-task sequence at 8M steps per task. This is the **main contribution**. Batch-runnable via `draft_4.sh` (23 A100 GPU-days × 2 tasks per GPU × 5 seeds ≈ 5 GPU-weeks; 45 configs / 2 per GPU / 5 seeds = 23 array tasks, each running ~80M env steps on 1 GPU).
2. SAC baselines (sparse + dense) × 5 seeds × 10-task. Needed for Fig 1 lower bound.
3. CKA-RL baseline × 5 seeds × 10-task (their actor pool + their SAC critic). Needed for Fig 2 ablation.
4. Representation metrics logging on the best (CKA, persistent) config — already running per-step in the training loop.

**Important (for secondary figures):**
5. Negative-bank ablation × 3 variants × 3 seeds × 10-task (Fig 5).
6. Single-task SGCRL reproduction on task 0 of each MetaWorld task × 3 seeds (sanity check — ideally matches task-0 of 9-config grid).

**Nice-to-have (for appendix):**
7. 20-task extended run on best config × 3 seeds.
8. Adaptive-entropy on/off ablation on best config × 3 seeds.

**Total compute budget estimate:**
- Primary grid: 45 runs × ~24 GPU-hours each = ~1100 GPU-hours.
- Baselines + ablations + replications: ~400 GPU-hours.
- Grand total: ~1500 GPU-hours (~60 GPU-days). Fits within NYU-AD HPC allocation if start today and use 2 tasks per GPU.

---

## Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Some seeds hit the plasticity-loss trap and produce flat curves | Appears as huge variance, weakens Fig 2 | (i) We now use adaptive entropy which mitigates; (ii) report median + IQR alongside mean + std |
| Reviewers ask "why not a single-buffer rehearsal baseline (CLEAR/ER)" | Reviewer might consider this a straightforward baseline we omitted | Add CLEAR as an additional baseline in an appendix if time permits; anyway our method does not use cross-task rehearsal in the standard sense |
| Reviewers ask "why MetaWorld and not something harder" | Weakens generality claim | Include BuilderBench preliminary results in appendix; discuss generality in Discussion |
| CKA-RL baseline underperforms badly — reviewers may call it a strawman | Unfair comparison concern | Reproduce CKA-RL's published numbers on a dense-reward setting first, then run them on our sparse setting; acknowledge the sparse setting is harder for SAC |
| Negative-bank section has weak signal if the hard_weighted gain is modest | Section 9 contribution looks thin | Be clear that the *analysis* (why vanilla fails) is as important as the positive result; reframe as "a principled approach is needed, we propose one." |
| BuilderBench port has known algorithmic gaps (no adaptive entropy, no neg bank) | Appendix F weaker than hoped | Plan: either skip BuilderBench for this submission or port the adaptive-entropy fix before running (see "BuilderBench audit" section below) |

---

## BuilderBench audit — current gaps and quick fixes

We reviewed the BuilderBench port (`rl/impls/continual_crl.py`) and identified gaps against the reference SGCRL code:

| Component | SGCRL (ours, fixed) | BuilderBench (current) | Fix effort |
|---|---|---|---|
| Entropy coefficient | **Adaptive** (log α autotuned toward target entropy -2.5 = -0.5 × 5) | **Fixed** `entropy_cost=0.1` (old pre-fix version) | 1 day — port the SAC dual-gradient from SGCRL |
| target_entropy | -0.5 × action_dim (= -2.5 for BuilderBench's 5-D actions) | Not defined | Trivial once adaptive entropy is in |
| Negative-replay bank | Implemented (vanilla + hard_weighted) | Not implemented | 1 day — port from `contrastive/negative_bank.py` |
| Automatic actor reset | Implemented, disabled by default | Not implemented | Optional — user explicitly doesn't want resets by default |
| CKA composition | Head-only + body folded into base | Matches SGCRL logic ✓ | — |
| Actor/critic decomposition | Actor head (Dense_4, Dense_5) vs body | Matches SGCRL logic ✓ | — |
| 12-task mixed-cube sequence | N/A (SGCRL uses MetaWorld) | Implemented with padding | — |

**Recommendation for the NeurIPS paper.** Treat BuilderBench as a **secondary / appendix-only** result in this paper. The architectural gap (no adaptive entropy) is the most important fix — without it, seed-dependent plasticity-loss traps will dominate BuilderBench as they did in SGCRL before our fix. If there is time after the main experiments run, do a one-day port of the adaptive-entropy code and rerun the two or three most important BuilderBench configs (CKA actor + persistent critic, reset + reset). If there is not time, the paper stands cleanly on MetaWorld alone.

---

## Post-submission (camera-ready) roadmap

- Additional seeds (10+) on the main grid for tighter error bars.
- BuilderBench full port with adaptive entropy + neg bank.
- Learnable α for pool blending (currently uniform-softmax over β_k).
- More principled negative-bank strategies (curriculum, diversity-aware sampling).
- Pixel-based observations (extend to Meta-World v3 / robosuite visual).
- Theoretical analysis of when persistent critic provably dominates reset critic (information-theoretic argument).

---

## Operational check-in — what Yumi should do next (week-by-week)

**Week of Apr 21 (NOW):**
- Kick off the full 9-config × 5-seed grid on NYU-AD HPC (`sbatch draft_4.sh`).
- Launch SAC-sparse and SAC-dense baselines in parallel.
- Start writing Introduction + Related Work (does not depend on results).

**Week of Apr 28:**
- First results come in — inspect learning curves, fix any launch issues, relaunch stragglers.
- Write Method + Problem Setup sections.
- Generate Fig 1 / Fig 2 placeholder layouts with synthetic data for layout.

**Week of May 5 (deadline week):**
- May 3: freeze main figures, polish Introduction.
- May 4: abstract submission. Polish Results.
- May 5: polish Discussion + Appendix.
- May 6: full paper submission.

---

## Proposed algorithm name

Candidates:
- **CG-CRL** — "Continual Goal-Conditioned Contrastive RL" (literal, clear, matches keywords).
- **CoCoRL** — "Continual Contrastive RL" (short, pronounceable, phonetically memorable).
- **KnowCRL** — "Knowledge-decomposed Contrastive RL" (emphasises decomposition).
- **C²KA-RL** — "Contrastive Continual Knowledge Adaptation RL" (emphasises CKA-RL lineage).

Recommendation: **CoCoRL** — simple, short, pronounceable, and does not commit to a particular architectural detail.
