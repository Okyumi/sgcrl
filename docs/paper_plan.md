# Paper Plan — NeurIPS 2026

- We propose a sparse-reward continual goal-conditioned reinforcement learning setting.
- The agent faces a sequence of tasks $\mathcal{M}^{(1)},\ldots,\mathcal{M}^{(N)}$ that share a state space, action space, and robot embodiment; each task defines its own transition kernel and goal distribution.
- The only reward signal the agent sees on any task is the goal-reaching event $r_g(s_t,a_t)=(1-\gamma)\,p(s_{t+1}=g\mid s_t,a_t)$, and no dense reward shaping is available at any point during training or evaluation.
- The agent must learn the current task while retaining competence on earlier tasks, and is evaluated by intra-task success curves, cross-task forgetting, and forward transfer.
- To the best of our knowledge, no prior continual-RL benchmark operates in this regime: Continual World (Wołczyk et al., NeurIPS 2021) and the CKA-RL benchmark (Hu et al., NeurIPS 2025) both rely on the Meta-World V2 dense rewards, sparse-reward Meta-World variants exist only in the single-task model-based RL literature, and hindsight-relabelling work for sparse reward targets meta-RL rather than continual RL.
- In most real-world and simulator settings we ultimately only care whether the agent reaches a goal or completes a task.
- Reward engineering relies heavily on human effort and per-task heuristics, and the resulting dense reward functions largely decide how well a continual RL algorithm appears to perform.
- It is therefore appealing to minimize manual reward design and instead rely on algorithms that can learn goal-related signals more naturally.
- We instantiate this setting on a ten-task Meta-World Sawyer manipulation sequence, trained with eight million environment steps per task and five seeds per configuration.
- A natural solver for this setup is contrastive goal-conditioned RL.
- The contrastive critic $f(s,a,g)=\phi(s,a)^\top\psi(g)$ is trained with InfoNCE on hindsight-relabeled positives and turns the value-learning problem into a classification problem.
- What the critic encodes is reachability structure, not a task-specific scalar return, and reachability structure is shared across a sequence of manipulation tasks that use the same robot embodiment.
- Continual RL also imposes a separate requirement on the policy class: an agent that must acquire new skills continually needs enough expressivity and enough effective computation to support them, as argued by Myers et al. (2026) in *On Computation and Reinforcement Learning*.
- Standard actor-critic RL has known scalability limitations; contrastive goal-conditioned RL has recently been shown to scale cleanly to very deep residual networks (Wang et al., NeurIPS 2025), which makes it a strong substrate for a continual agent that needs to keep absorbing skills.
- The continual setup exposes questions about contrastive RL that a single-task setup cannot.
- How do the encoders $\phi$ and $\psi$ adapt as new tasks arrive, how does the representation space shift and evolve across a task sequence, and does the representation learned on later tasks remain compatible with goals from earlier tasks.
- These questions sit next to an empirical literature on representation-level drift in continual learning (Caccia et al. 2021; Zhang, Dou, Wu, 2022 on feature forgetting; Anthes et al. 2024 on drift under orthogonal optimisation), which has mostly been developed in supervised continual learning and only recently carried into continual RL (TeLAPA, 2026; C-CHAIN, ICML 2025).
- A continual sparse-reward contrastive setting gives us a concrete environment in which to probe these representation-drift questions on a value function rather than a classifier.
- It also suggests natural extensions: a fixed goal under changing state spaces, so that only the critic's input distribution shifts; or a multi-armed-bandit-with-state variant in which one route leads to stochastic success and the environment changes which route that is over the task sequence.
- Our framework combines three ingredients in the sparse-reward continual setting.
- The first ingredient is the contrastive goal-conditioned RL critic described above.
- The second ingredient is a policy decomposition: the policy at task $k$ is expressed as $\theta'=\theta_{\text{base}}+\sum_{j=1}^{k-1}\alpha_j v_j + v_k$, with $\alpha=\text{softmax}(\beta\cdot\alpha_{\text{scale}})$.
- $\theta_{\text{base}}$ is frozen after task 0, each $v_j$ is a knowledge vector learned on a past task, and $v_k$ is the current task's knowledge vector.
- By default only the output head is decomposed, while the encoder body receives gradients so that representations can drift without growing the number of knowledge vectors.
- The third ingredient is a bounded knowledge pool $\mathcal{V}=\{v_1,\ldots,v_{k-1}\}$, capped at $K_{\max}=5$ vectors.
- Whenever the pool overflows, the two most cosine-similar vectors are merged by averaging.
- Policy decomposition and bounded knowledge pools have been studied in prior continual-RL work, most recently by Hu et al. (NeurIPS 2025) in a dense-reward SAC setting.
- Continual learning with per-task replay has a natural offline-to-online structure that previous work has not made use of.
- By the time the agent reaches task $k$, it owns replay buffers from tasks $0,\ldots,k-1$, and these buffers are already full of HER-relabeled goals that describe states the robot can actually reach.
- Discarding this data at the task boundary is wasteful; it is effectively an offline dataset sitting next to the online stream of the current task.
- We introduce a hard-weighted negative bank that turns these past-task goals into additional contrastive negatives for the current critic.
- A naive version of this idea fails, and the failure is informative: Meta-World Sawyer tasks occupy disjoint workspace regions, so a goal drawn from a past task's buffer is trivially distinguishable from a current-task goal by raw coordinates alone, and the critic then achieves high categorical accuracy without learning a useful representation.
- The hard-weighted bank addresses this by scoring a candidate pool of past-task goals against each anchor in the batch and keeping only the top $M$ — the candidates that currently receive the highest score and therefore provide the strongest gradient signal.
- These hard negatives are appended to the in-batch logits with a scalar weight $w_{\text{bank}}=0.3$, so that bank negatives cannot dominate the softmax.
- The full framework is then: contrastive goal-conditioned RL, with a policy decomposition and a knowledge pool on the actor, in a sparse-reward continual setting, with an offline-to-online hard-weighted negative bank on the contrastive loss.
- We evaluate through a nine-cell ablation over actor and critic evolution modes.
- The actor can be reset, carried forward persistently, or decomposed; the critic can be reset, carried forward persistently, or decomposed.
- Running all nine cells separates the contribution of actor-side mechanisms from the contribution of critic-side mechanisms, which earlier work has conflated because in SAC-based continual RL the critic does not transfer well regardless of what happens on the actor side.
- We ask two primary empirical questions.
- First, when the critic is contrastive rather than SAC, does persisting it across tasks help?
- Second, given a persistent contrastive critic, does the policy decomposition and knowledge pool on the actor add value on top of that?
- The answer to the first question extends the finding of Wołczyk et al. (NeurIPS 2022), who showed in dense-reward SAC that critic transfer dominates actor transfer for forward transfer, into the sparse-reward contrastive regime.
- The answer to the second question is driven by a different mechanism: the decomposition converts critic-side retention into actor-side retention, which is what drives backward transfer.
- The predicted headline configuration is a decomposed actor with a persistent contrastive critic.
- A third question concerns actor-side representation quality over long sequences.
- We measure dormant-neuron ratio (with the $\tau=0.025$ threshold from Sokar et al. 2023), neural-collapse statistics NRC1 and NRC2 (Papyan et al. 2020), feature rank (Kumar et al. 2021; Lyle et al. 2022), and entropy on the actor encoder.
- These degrade over long task sequences, and we expect actor plasticity loss — not critic forgetting — to eventually bottleneck performance.
- We track these metrics across the full ten-task sequence for all nine cells, and correlate delayed success jumps on late tasks with shifts in actor-feature rank.
- A fourth question concerns the negative bank: does reusing past-task data as contrastive negatives help, and if so, under what filtering.
- We compare three settings: no bank, a vanilla bank, and the hard-weighted bank.
- We expect vanilla cross-task negatives to hurt, the hard-weighted bank to help, and the gap to be explained by how much of the critic's softmax probability ends up on trivially-separable workspace regions versus on genuinely goal-relevant features.
- The full nine-cell ablation runs with five seeds on the ten-task Sawyer sequence, with eight million environment steps per task, yielding $9\times 5=45$ full-scale runs.
- Around the headline cell we run sensitivity sweeps over $K_{\max}$, $w_{\text{bank}}$, and the target entropy of the adaptive-entropy actor.
- A sparse-reward SAC baseline and a sparse-reward re-implementation of policy-decomposition-with-SAC calibrate the role of the contrastive critic under matched continual machinery.
- A dense-reward SAC baseline connects our numbers to published Continual World results.
- The broader message is that progress in continual RL comes from pairing a value function whose representation is task-agnostic (a contrastive critic) with lightweight parameter-level mechanisms (policy decomposition, bounded knowledge pool) that translate critic-side retention into actor-side retention, and from using the data already accumulated from past tasks as a principled contrastive signal rather than discarding it.
- Sparse reward is what forces this discipline: under dense reward, a well-engineered reward function can carry an algorithm to a respectable number, and the representational question does not have to be answered.
- The continual setting is also an interesting lens back onto contrastive RL itself: it exposes how $\phi$ and $\psi$ adapt, how the representation space evolves when new goal distributions arrive, and when additional compute or expressivity in the policy becomes necessary to absorb new skills.
- These are open directions that the scalability of contrastive GCRL makes feasible to pursue, and we view the setting introduced here as a starting point for that line of work.

---

## Target and logistics

- Target venue: NeurIPS 2026. Abstract due May 4 AOE, full paper May 6 AOE.
- Branch of record: `section3_done`.
- Experimental driver: `run_continual_contrastive.py`; SLURM launcher: `draft_3.sh` single-run, `draft_4.sh` batch.
- Ablation configs: `experiment_configs.py`, 9 cells × 5 seeds = 45 runs.
- Full citation list with annotations and verified references: `docs/citations.md`.
- Algorithm pseudocode: `docs/algorithm_pseudocode.md`.
- Negative bank design note: `docs/negative_bank.md`.
- Progress tracker: `docs/paper_planning_tracking.md`.

---

## Writing schedule

- Apr 22 — launch the full 9 × 5 grid on NYUAD HPC.
- Apr 28 — first-pass figures: nine-cell learning curves, forgetting matrix, bank diagnostics, plasticity metrics.
- Apr 30 — draft introduction, problem setting, related work, method.
- May 3 — draft experiments and discussion.
- May 4 morning — internal review with Prof. Ross; submit abstract May 4 AOE.
- May 6 AOE — final polish, appendices, submission.

---

## Open questions for Prof. Ross

- Should the nine-cell grid be the headline figure of the experiments section, or should we lead with the decomposed-actor + persistent-critic cell alone and push the grid to an appendix?
- Should dense-reward SAC and dense-reward policy-decomposition baselines appear in the main body for calibration with the literature, or only in an appendix?
- Should the twenty-task stress test live in the main experiments, or as an appendix robustness study?
- Does the negative-bank analysis deserve a standalone narrative slot, given how cleanly it isolates a qualitative failure mode of naive cross-task negatives?
