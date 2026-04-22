# Paper Plan — NeurIPS 2026

## Setting

- We consider sparse-reward continual reinforcement learning.
- An agent faces a sequence of tasks on a shared embodiment, each with its own goal.
- The only reward is a terminal 0/1 signal on whether the goal is reached.
- The agent must learn each new task while retaining what it learned before.

## Gap in the literature

- Continual RL work has almost always used dense, hand-engineered per-task rewards.
- Sparse-reward RL has almost always been studied one task at a time.
- The combination — continual learning with only a per-task goal and a terminal success signal — is what an actually-deployed agent would face, and sits as a gap between these two lines of work.
- Dense rewards hide the representational question; when rewards are sparse, the algorithm must rely on structure that transfers across tasks.

## Solver

- Goal-conditioned RL with self-supervised objectives is a natural fit for this setting; we use contrastive goal-conditioned RL.
- The critic $f(s,a,g) = \phi(s,a)^\top \psi(g)$ is trained by classification on hindsight-relabeled positives and in-batch negatives.
- The critic encodes reachability structure rather than a task-specific scalar return, and reachability structure is exactly what should transfer across tasks that share a single robot.
- Contrastive GCRL scales cleanly to deep residual networks (Wang et al., NeurIPS 2025), which matters because a continual agent needs enough expressivity and compute to keep absorbing new skills (Myers et al., 2026).

## Framework

- Carry the contrastive critic across tasks so that reachability structure is not thrown away at every task boundary.
- Use knowledge decomposition on the actor: at task $k$, $\theta' = \theta_{\text{base}} + \sum_{j=1}^{k-1} \alpha_j v_j + v_k$ with $\alpha = \text{softmax}(\beta \cdot \alpha_{\text{scale}})$. By default we decompose only the output head.
- Maintain a bounded knowledge pool $\mathcal{V}$ capped at $K_{\max} = 5$; when it overflows, merge the two most cosine-similar vectors by averaging.
- Reuse past-task replay via a hard-weighted negative bank on the contrastive loss.
    - Past-task buffers are already full of hindsight-relabeled goals; discarding them at task boundaries is wasteful.
    - A naive bank fails: Meta-World tasks occupy disjoint workspace regions, so cross-task goals are trivially separable and the critic saturates without learning.
    - For each anchor we score a candidate pool of past-task goals and keep the top $M$ hardest; these are appended to the in-batch logits with weight $w_{\text{bank}} = 0.3$.

## Evaluation

- Ten-task Meta-World Sawyer sequence, 8M steps per task, 5 seeds per configuration.
- Main evaluation: 9-cell ablation over actor $\times$ critic evolution in $\{\text{reset}, \text{persistent}, \text{decomposed}\}$.
- Four empirical questions:
    - (i) When the critic is contrastive, does persisting it across tasks help?
    - (ii) Given a persistent contrastive critic, does decomposing the actor add value on top?
    - (iii) How does actor representation quality evolve over long sequences, and at what point does plasticity loss bottleneck performance?
    - (iv) Does reusing past-task data as contrastive negatives help, and under what filtering?
- Plasticity diagnostics on the actor: dormant ratio ($\tau = 0.025$), NRC1/NRC2, feature rank, entropy.
- Bank ablation: off, vanilla, hard-weighted.

## Expected story

- Persisting a contrastive critic helps; this extends Wołczyk et al. (2022) from dense-SAC to the sparse-contrastive regime.
- Knowledge decomposition on the actor translates critic-side retention into actor-side retention and drives backward transfer.
- The headline cell is a decomposed actor with a persistent contrastive critic.
- Actor plasticity loss eventually becomes the binding constraint; the critic does not.
- Vanilla cross-task negatives hurt; hard-weighted mining recovers useful signal.

## Discussion and future directions

- The setting is a probe for the representations a continually-learning agent forms, not just another benchmark.
- Our design choices are deliberately simple; better results are very plausible along every axis — objective, decomposition, pool, data reuse, plasticity management.
- The setting is algorithm-agnostic: imitation, self-supervised representation learning, world-modelling, and offline-to-online methods all apply.
- Open directions:
    - Representation drift and retention on a value function, parallel to supervised continual-representation work.
    - Scalability: how far deep contrastive scaling carries a continual agent.
    - Offline-to-online reuse of past-task buffers beyond the contrastive-negatives view.
    - Longer and harder task sequences, larger and cross-embodiment setups.
    - Variants of the setting: a fixed-goal variant with state-distribution drift only; a state-conditioned bandit variant with stochastic success routes that change across the sequence.

---

## Target and logistics

- Target venue: NeurIPS 2026. Abstract due May 4 AOE, full paper May 6 AOE.
- Branch of record: `section3_done` (sgcrl).
- Paper repo: `Okyumi/NeurIPS-2026---RL`, build root `main.tex`, writing committed directly to `main` as `Okyumi`.
- Experimental driver: `run_continual_contrastive.py`; launchers `draft_3.sh` (single-run) and `draft_4.sh` (batch).
- Ablation configs: `experiment_configs.py`, 9 cells × 5 seeds = 45 runs.
- Full annotated citation list: `docs/citations.md`.
- Algorithm pseudocode: `docs/algorithm_pseudocode.md`.
- Negative bank design note: `docs/negative_bank.md`.
- Progress tracker: `docs/paper_planning_tracking.md`.

## Writing schedule

- Apr 22 — launch the full 9 × 5 grid on NYUAD HPC.
- Apr 28 — first-pass figures: nine-cell curves, forgetting matrix, bank diagnostics, plasticity metrics.
- Apr 30 — draft introduction, problem setting, related work, method.
- May 3 — draft experiments and discussion.
- May 4 AM — internal review with Prof. Ross; submit abstract May 4 AOE.
- May 6 AOE — final polish, appendices, submission.

## Open questions for Prof. Ross

- Nine-cell grid as the headline figure, or decomposed-actor + persistent-critic cell alone with the grid in an appendix?
- Dense-reward SAC and dense-reward policy-decomposition baselines: main body (for calibration) or appendix only?
- Twenty-task stress test: main experiments or appendix robustness study?
- Negative-bank analysis: standalone narrative slot, or a subsection inside experiments?
