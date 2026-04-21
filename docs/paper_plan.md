# Paper Plan — NeurIPS 2026

- We study continual reinforcement learning in a sparse-reward, goal-conditioned setting.
- In continual RL, an agent faces a sequence of tasks and must learn each new task while retaining competence on earlier ones.
- In practice, most continual RL benchmarks and algorithms rely on dense, hand-crafted reward functions, one per task.
- These dense rewards are engineered to carry per-task expert knowledge, and they largely decide how well an algorithm appears to perform.
- This hides the representational question that continual RL is actually supposed to be about: whether a policy and a value function can develop structure that transfers across a task sequence.
- It also does not reflect how an agent would be deployed on a real robot, where the reward is effectively a success event.
- In this paper we remove the dense rewards entirely.
- The only reward signal the agent sees is the goal-reaching event $r_g(s_t,a_t)=(1-\gamma)\,p(s_{t+1}=g\mid s_t,a_t)$.
- We instantiate this setting on a ten-task Meta-World Sawyer manipulation sequence, trained with eight million environment steps per task and five seeds per configuration.
- We call this setting sparse-reward continual goal-conditioned RL, and we treat the setting itself as a contribution of the paper.
- To operate in this setting we propose a framework that puts together three ingredients.
- The first ingredient is a contrastive goal-conditioned RL critic $f(s,a,g)=\phi(s,a)^\top\psi(g)$, trained with InfoNCE on hindsight-relabeled positives.
- Contrastive GCRL is not our contribution: we use it as the underlying RL algorithm, following the line of work of Eysenbach et al. 2022, Liu et al. 2025, and Wang et al. 2025.
- What matters for us is what the contrastive critic encodes: reachability structure, rather than a task-specific scalar return.
- Reachability structure is exactly what should be shared across a sequence of manipulation tasks that use the same robot embodiment.
- The second ingredient is a policy decomposition.
- We express the policy at task $k$ as $\theta'=\theta_{\text{base}}+\sum_{j=1}^{k-1}\alpha_j v_j + v_k$, with $\alpha=\text{softmax}(\beta\cdot\alpha_{\text{scale}})$.
- Here $\theta_{\text{base}}$ is frozen after task 0, each $v_j$ is a knowledge vector learned on a past task, and $v_k$ is the current task's knowledge vector.
- By default only the output head is decomposed, while the encoder body receives gradients so that representations can drift without growing the number of knowledge vectors.
- The third ingredient is a bounded knowledge pool $\mathcal{V}=\{v_1,\ldots,v_{k-1}\}$.
- The pool is capped at $K_{\max}=5$ vectors; whenever it overflows, the two most cosine-similar vectors are merged by averaging.
- Policy decomposition and a bounded knowledge pool are not original to this paper.
- Prior work has studied them in continual RL, most recently Hu et al. (NeurIPS 2025) in a dense-reward SAC setting.
- We cite that line of work and use the same functional form; we do not position this paper as an extension of it.
- Our claim is about the combination: what happens when a policy decomposition and a knowledge pool are paired with a contrastive goal-conditioned critic, under sparse rewards.
- The fourth piece is a negative-sampling mechanism that is specific to the continual contrastive setting.
- By the time the agent reaches task $k$, it owns replay buffers from tasks $0,\ldots,k-1$, all full of HER-relabeled goals.
- These past-task goals are wasted data: the InfoNCE loss on the current task only uses in-batch negatives.
- The natural thing to do is to reuse those past goals as extra contrastive negatives when training the current critic.
- The naive version of this idea fails, and the failure is informative.
- Meta-World Sawyer tasks occupy disjoint workspace regions, so a goal drawn from a past task's buffer is trivially distinguishable from a current-task goal by raw coordinates alone.
- The critic then achieves high categorical accuracy without learning a useful representation, and learning slows down relative to not using the extra negatives at all.
- We address this with a hard-weighted negative bank.
- For each anchor in the batch, we score a candidate pool of past-task goals against that anchor and keep only the top $M$ — the candidates that currently receive the highest score and therefore provide the strongest gradient signal.
- We then append these hard negatives to the in-batch logits with a scalar weight $w_{\text{bank}}=0.3$, so that even hard-mined bank negatives cannot dominate the softmax.
- The resulting framework is: contrastive GCRL, with a policy decomposition and a knowledge pool on the actor, in a sparse-reward continual setting, with an offline-to-online hard-weighted negative bank on the contrastive loss.
- To evaluate the framework we study a nine-cell ablation over actor and critic evolution modes.
- The actor can be reset, carried forward persistently, or decomposed; the critic can be reset, carried forward persistently, or decomposed.
- Running all nine cells lets us separate the contribution of actor-side mechanisms from the contribution of critic-side mechanisms.
- Previous work has conflated the two, because in SAC-based continual RL the critic does not transfer well regardless of what happens on the actor side.
- We ask two primary empirical questions.
- First, when the critic is contrastive rather than SAC, does persisting it across tasks help?
- Second, given a persistent contrastive critic, does the policy decomposition and knowledge pool on the actor add value on top of that?
- We expect the answer to the first question to be yes, extending the SAC-dense-reward finding of Wołczyk et al. (NeurIPS 2022) that critic transfer dominates actor transfer for forward transfer.
- We expect the answer to the second question to be yes as well, but for a different reason: the decomposition is what converts critic-side retention into actor-side retention, which is what drives backward transfer.
- Our predicted headline configuration is therefore the cell with a decomposed actor and a persistent contrastive critic.
- We also ask a third question, motivated by previous observations in our own runs.
- Actor-side representation quality, measured by dormant-neuron ratio (with the $\tau=0.025$ threshold from Sokar et al. 2023), neural-collapse statistics NRC1 and NRC2 (Papyan et al. 2020), and feature rank, degrades over long task sequences.
- In long sequences, actor plasticity loss — not critic forgetting — is what eventually bottlenecks performance.
- We track these metrics across the full ten-task sequence for all nine cells, and correlate delayed success jumps on late tasks with shifts in actor-feature rank.
- Finally, the negative bank gives us a clean fourth question: does reusing past-task data as contrastive negatives help, and if so, what kind of reuse.
- We compare three settings: no bank, a vanilla bank, and the hard-weighted bank.
- We predict that vanilla cross-task negatives hurt, that the hard-weighted bank helps, and that the gap is explained by how much of the critic's softmax probability ends up on trivially-separable workspace regions versus on genuinely goal-relevant features.
- To make all of this testable, we run the full nine-cell ablation with five seeds, on the ten-task Sawyer sequence, with eight million environment steps per task, giving $9\times 5=45$ full-scale runs.
- Around the headline cell we run sensitivity sweeps over $K_{\max}$, over $w_{\text{bank}}$, and over the target entropy of the adaptive-entropy actor.
- For calibration against the literature we additionally run a sparse-reward SAC baseline and a sparse-reward re-implementation of policy-decomposition-with-SAC; a dense-reward SAC baseline is reported only to connect our numbers to published Continual World results.
- The broader message we want to deliver is not that contrastive RL beats SAC, or that decomposition beats no-decomposition.
- It is that the right way to do continual RL is to pick a value function whose representation is task-agnostic (which is what a contrastive critic gives), then use lightweight parameter-level mechanisms (policy decomposition, bounded knowledge pool) to translate critic-side retention into actor-side retention, and finally use the data you already have from past tasks in a principled way (hard-weighted negative bank) rather than discard it.
- Sparse reward is what forces this discipline: under dense reward, a poorly-designed reward can carry an algorithm to a decent number, so the representational question does not have to be answered.

---

## Target and logistics

- Target venue: NeurIPS 2026. Abstract due May 4 AOE, full paper May 6 AOE.
- Branch of record: `section3_done`.
- Experimental driver: `run_continual_contrastive.py`; SLURM launcher: `draft_3.sh` single-run, `draft_4.sh` batch.
- Ablation configs: `experiment_configs.py`, 9 cells × 5 seeds = 45 runs.
- Full citation list with annotations and verified references: `docs/citations.md`.
- Algorithm pseudocode: `docs/algorithm_pseudocode.md`.
- Negative bank design note: `docs/negative_bank.md`.

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
