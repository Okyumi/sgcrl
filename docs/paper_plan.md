# Paper Plan — NeurIPS 2026

- In this paper we consider a sparse-reward continual reinforcement learning setting.
- The agent faces a sequence of tasks that share a state space, action space, and robot embodiment; each task has its own transition kernel and its own goal.
- The only reward signal the agent receives on any task is a sparse goal-reaching signal associated with that task's goal — no per-step dense reward and no per-task reward shaping.
- The agent must learn each new task while retaining competence on earlier ones.
- Previous work on continual reinforcement learning has almost always operated in a dense-reward setting, with per-task reward functions engineered and tuned by humans.
- Previous work on sparse-reward reinforcement learning has almost always operated in a single-task setting, where the difficulty is exploration on one fixed task rather than preservation of skill across a sequence of tasks.
- What is missing from the literature is the combination: an agent that learns continuously across a sequence of tasks, with no dense reward engineered by humans, seeing only the goal for each task and a sparse signal when it is reached.
- This combination is exactly the regime in which a continually learning agent would actually be deployed. On a real robot, or in any simulator that is meant to mirror deployment, we ultimately only care whether each task gets completed.
- Engineering a dense reward per task is labour-intensive, brittle, and conflates two separate questions — how well an algorithm preserves and reuses knowledge across tasks, and how well a reward function has been shaped for each task.
- Removing the dense rewards forces the algorithm to answer the first question on its own terms, using only the structure that transfers across tasks sharing a single robot.
- A natural class of algorithms for this setting is reinforcement learning with self-supervised objectives that let the agent explore the representation space of a task by conditioning on its goal.
- A particularly clean instance is contrastive goal-conditioned reinforcement learning, in which the critic is a dual encoder $f(s,a,g)=\phi(s,a)^\top\psi(g)$ trained with a classification objective on hindsight-relabeled positives and in-batch negatives.
- The critic in this formulation does not estimate a task-specific scalar return; it estimates reachability structure — which states are reachable from which state-action pairs — and reachability structure is the part of the value function that should transfer across a sequence of tasks that use the same robot.
- Contrastive goal-conditioned RL also has a practical property that matters in the continual regime: it scales cleanly to very deep residual networks (Wang et al., NeurIPS 2025), which matters because a continual agent that must keep absorbing new skills needs a policy class with enough expressivity and enough effective computation to support them (Myers et al., 2026, *On Computation and Reinforcement Learning*).
- We therefore use contrastive goal-conditioned RL as the underlying algorithm in this sparse-reward continual setting and build a framework around it.
- The framework combines three ingredients.
- The first ingredient is the contrastive goal-conditioned critic described above, carried forward across tasks so that the reachability structure it represents is not thrown away at every task boundary.
- The second ingredient is a policy decomposition: at task $k$ the policy is expressed as $\theta'=\theta_{\text{base}}+\sum_{j=1}^{k-1}\alpha_j v_j + v_k$, with $\alpha=\text{softmax}(\beta\cdot\alpha_{\text{scale}})$.
- $\theta_{\text{base}}$ is frozen after task 0, each $v_j$ is a knowledge vector learned on a past task, and $v_k$ is the current task's knowledge vector; by default only the output head is decomposed, while the encoder body receives gradients so that representations can drift without growing the number of knowledge vectors.
- The third ingredient is a bounded knowledge pool $\mathcal{V}=\{v_1,\ldots,v_{k-1}\}$, capped at $K_{\max}=5$ vectors; whenever the pool overflows, the two most cosine-similar vectors are merged by averaging.
- Policy decomposition and bounded knowledge pools have been studied in prior continual-RL work, most recently by Hu et al. (NeurIPS 2025) in a dense-reward SAC setting.
- Continual learning with per-task replay has a natural offline-to-online structure that previous work has not exploited.
- By the time the agent reaches task $k$, it owns replay buffers from tasks $0,\ldots,k-1$, already full of hindsight-relabeled goals that describe states the robot has actually reached.
- Discarding this data at the task boundary is wasteful; it is effectively an offline dataset sitting next to the online stream of the current task.
- We introduce a hard-weighted negative bank that turns these past-task goals into additional contrastive negatives for the current critic.
- A naive version of this idea fails, and the failure is informative: Meta-World Sawyer tasks occupy disjoint workspace regions, so a goal drawn from a past task's buffer is trivially distinguishable from a current-task goal by raw coordinates alone, and the critic then achieves high categorical accuracy without learning a useful representation.
- The hard-weighted bank addresses this by scoring a candidate pool of past-task goals against each anchor in the batch and keeping only the top $M$ — the candidates that currently receive the highest score and therefore provide the strongest gradient signal.
- These hard negatives are appended to the in-batch logits with a scalar weight $w_{\text{bank}}=0.3$, so that bank negatives cannot dominate the softmax.
- The full framework is then: contrastive goal-conditioned RL, with a policy decomposition and a knowledge pool on the actor, in a sparse-reward continual setting, with an offline-to-online hard-weighted negative bank on the contrastive loss.
- We evaluate the framework on a ten-task Meta-World Sawyer manipulation sequence, trained with eight million environment steps per task and five seeds per configuration.
- The central evaluation is a nine-cell ablation over actor and critic evolution modes.
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
- These degrade over long task sequences, and actor plasticity loss — not critic forgetting — is expected to eventually bottleneck performance.
- We track these metrics across the full ten-task sequence for all nine cells, and correlate delayed success jumps on late tasks with shifts in actor-feature rank.
- A fourth question concerns the negative bank: does reusing past-task data as contrastive negatives help, and if so, under what filtering.
- We compare three settings: no bank, a vanilla bank, and the hard-weighted bank.
- The vanilla bank is expected to hurt, the hard-weighted bank to help, and the gap to be explained by how much of the critic's softmax probability ends up on trivially-separable workspace regions versus on genuinely goal-relevant features.
- The continual sparse-reward contrastive setting also opens a set of representation-level questions on contrastive RL itself.
- How do the encoders $\phi$ and $\psi$ evolve across a task sequence, do they drift in ways that preserve stable behaviour (as in the supervised continual-learning drift literature, Caccia et al. 2021; Zhang, Dou, Wu 2022; Anthes et al. 2024) or in ways that fracture behaviour on earlier goals, and does a shared representation space emerge that covers the goals of all tasks seen so far.
- Closely related questions have started to be studied in non-contrastive continual RL (TeLAPA, 2026; C-CHAIN, ICML 2025), and the sparse-reward contrastive setting gives a clean testbed to ask them for a value function rather than a classifier.
- The broader message is that progress in continual RL comes from pairing a value function whose representation is task-agnostic (a contrastive critic) with lightweight parameter-level mechanisms (policy decomposition, bounded knowledge pool) that translate critic-side retention into actor-side retention, and from using the data already accumulated from past tasks as a principled contrastive signal rather than discarding it.
- Sparse reward is what forces this discipline: under dense reward, a well-engineered reward function can carry an algorithm to a respectable number, and the representational question does not have to be answered.
- Many design choices in our framework are intentionally simple, and we expect future work to obtain stronger results on this setting than we report here.
- Methodological improvements are likely on every axis we touch: the contrastive objective itself, the form of the policy decomposition, the way the knowledge pool is maintained, the way past-task data is reused, and the way plasticity is managed on the actor.
- The setting admits a wide range of algorithmic approaches: imitation learning, self-supervised representation learning, world-modelling, and offline-to-online learning all plausibly apply, and results from such directions would be welcome.
- We introduce the sparse-reward continual goal-conditioned setting as a target for the community and invite work that extends, rethinks, or replaces the framework we propose here.
- Natural directions include a fixed-goal variant under changing state spaces, so that only the critic's input distribution shifts; a multi-armed-bandit-with-state variant in which one route leads to stochastic success and the environment changes which route that is across the task sequence; and larger and longer task sequences that stress representation retention further.

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
