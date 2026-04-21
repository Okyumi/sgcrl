# Paper Plan — NeurIPS 2026

- We study continual reinforcement learning in a sparse-reward, goal-conditioned setting.
- In continual RL, an agent faces a sequence of tasks and must learn each new task while retaining competence on earlier ones.
- Most continual RL benchmarks and algorithms rely on dense, hand-crafted reward functions, one per task.
- Dense rewards carry per-task expert knowledge and largely decide how well an algorithm appears to perform.
- They obscure the representational question that continual RL is supposed to be about: whether a policy and a value function develop structure that transfers across a task sequence.
- They also do not reflect how an agent is deployed on a real robot, where reward is effectively a success event.
- We remove dense rewards.
- The only reward signal is the goal-reaching event $r_g(s_t,a_t)=(1-\gamma)\,p(s_{t+1}=g\mid s_t,a_t)$.
- We instantiate this setting on a ten-task Meta-World Sawyer manipulation sequence, trained with eight million environment steps per task and five seeds per configuration.
- We call this setting sparse-reward continual goal-conditioned RL, and the setting itself is a contribution of the paper.
- Our framework combines three ingredients in this setting.
- The first ingredient is a contrastive goal-conditioned RL critic $f(s,a,g)=\phi(s,a)^\top\psi(g)$, trained with InfoNCE on hindsight-relabeled positives.
- The contrastive critic encodes reachability structure, not a task-specific scalar return.
- Reachability structure is shared across a sequence of manipulation tasks that use the same robot embodiment.
- The second ingredient is a policy decomposition.
- The policy at task $k$ is expressed as $\theta'=\theta_{\text{base}}+\sum_{j=1}^{k-1}\alpha_j v_j + v_k$, with $\alpha=\text{softmax}(\beta\cdot\alpha_{\text{scale}})$.
- $\theta_{\text{base}}$ is frozen after task 0, each $v_j$ is a knowledge vector learned on a past task, and $v_k$ is the current task's knowledge vector.
- By default only the output head is decomposed, while the encoder body receives gradients so that representations can drift without growing the number of knowledge vectors.
- The third ingredient is a bounded knowledge pool $\mathcal{V}=\{v_1,\ldots,v_{k-1}\}$, capped at $K_{\max}=5$ vectors.
- Whenever the pool overflows, the two most cosine-similar vectors are merged by averaging.
- Policy decomposition and bounded knowledge pools have been studied in prior work on continual RL, most recently by Hu et al. (NeurIPS 2025) in a dense-reward SAC setting.
- A fourth piece is specific to the continual contrastive setting.
- By task $k$, the agent owns replay buffers from tasks $0,\ldots,k-1$, all full of HER-relabeled goals.
- These past-task goals can serve as extra contrastive negatives when training the current critic.
- A naive version of this idea fails, and the failure is informative.
- Meta-World Sawyer tasks occupy disjoint workspace regions, so a goal drawn from a past task's buffer is trivially distinguishable from a current-task goal by raw coordinates alone.
- The critic then achieves high categorical accuracy without learning a useful representation, and learning slows down relative to using no extra negatives.
- We address this with a hard-weighted negative bank.
- For each anchor in the batch, we score a candidate pool of past-task goals and keep only the top $M$ — the candidates that currently receive the highest score and therefore provide the strongest gradient signal.
- These hard negatives are appended to the in-batch logits with a scalar weight $w_{\text{bank}}=0.3$, so that bank negatives cannot dominate the softmax.
- The framework is then: contrastive goal-conditioned RL, with a policy decomposition and a knowledge pool on the actor, in a sparse-reward continual setting, with a hard-weighted negative bank on the contrastive loss.
- We evaluate through a nine-cell ablation over actor and critic evolution modes.
- The actor can be reset, carried forward persistently, or decomposed; the critic can be reset, carried forward persistently, or decomposed.
- Running all nine cells separates the contribution of actor-side mechanisms from the contribution of critic-side mechanisms.
- Earlier work conflated the two, because in SAC-based continual RL the critic does not transfer well regardless of what happens on the actor side.
- We ask two primary empirical questions.
- First, when the critic is contrastive rather than SAC, does persisting it across tasks help?
- Second, given a persistent contrastive critic, does the policy decomposition and knowledge pool on the actor add value on top of that?
- The answer to the first question extends the finding of Wołczyk et al. (NeurIPS 2022), who showed in dense-reward SAC that critic transfer dominates actor transfer for forward transfer, into the sparse-reward contrastive regime.
- The answer to the second question is driven by a different mechanism: the decomposition converts critic-side retention into actor-side retention, which is what drives backward transfer.
- The predicted headline configuration is a decomposed actor with a persistent contrastive critic.
- A third question concerns actor-side representation quality over long sequences.
- We measure dormant-neuron ratio (with the $\tau=0.025$ threshold from Sokar et al. 2023), neural-collapse statistics NRC1 and NRC2 (Papyan et al. 2020), and feature rank on the actor encoder.
- These degrade over long task sequences, and we expect actor plasticity loss — not critic forgetting — to eventually bottleneck performance.
- We track these metrics across the full ten-task sequence for all nine cells, and correlate delayed success jumps on late tasks with shifts in actor-feature rank.
- A fourth question concerns the negative bank: does reusing past-task data as contrastive negatives help, and if so, under what filtering.
- We compare three settings: no bank, a vanilla bank, and the hard-weighted bank.
- We expect vanilla cross-task negatives to hurt, the hard-weighted bank to help, and the gap to be explained by how much of the critic's softmax probability ends up on trivially-separable workspace regions versus on genuinely goal-relevant features.
- The full nine-cell ablation runs with five seeds on the ten-task Sawyer sequence, with eight million environment steps per task, yielding $9\times 5=45$ full-scale runs.
- Around the headline cell we run sensitivity sweeps over $K_{\max}$, $w_{\text{bank}}$, and the target entropy of the adaptive-entropy actor.
- A sparse-reward SAC baseline and a sparse-reward re-implementation of policy-decomposition-with-SAC calibrate the role of the contrastive critic under matched continual machinery.
- A dense-reward SAC baseline connects our numbers to published Continual World results.
- The broader message is not that contrastive RL beats SAC, or that decomposition beats no-decomposition.
- Progress in continual RL comes from pairing a value function whose representation is task-agnostic (a contrastive critic) with lightweight parameter-level mechanisms (policy decomposition, bounded knowledge pool) that translate critic-side retention into actor-side retention, and from using the data already accumulated from past tasks as a principled contrastive signal rather than discarding it.
- Sparse reward is what forces this discipline: under dense reward, a well-engineered reward function can carry an algorithm to a respectable number, and the representational question does not have to be answered.

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
