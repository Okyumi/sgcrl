# Paper Plan — NeurIPS 2026

We study continual reinforcement learning under sparse reward.
An agent is presented with a sequence of tasks on a shared embodiment, one task after another.
Each task specifies a goal, and the only reward the agent receives is a terminal signal indicating whether the goal has been reached at the end of the episode.
The agent must learn each new task while retaining what it has learned on earlier ones.
Continual reinforcement learning has almost always been studied under dense, manually engineered per-task rewards.
Sparse reward reinforcement learning has almost always been studied on a single fixed task.
The combination, continual learning under sparse goal-based reward, remains underexplored, even though this is the setting in which a continually learning agent would actually be deployed.
Dense reward functions make learning tractable, but they conflate two different questions: how well an algorithm preserves and reuses skill across tasks, and how well the reward function has been shaped on each task.
Removing the dense reward forces the algorithm to rely on structure that transfers across tasks sharing a single embodiment.
A natural class of solvers for this setting are reinforcement learning algorithms with self-supervised objectives that let the agent explore the representation space of a task by conditioning on its goal, or more generally any suitable goal-conditioned reinforcement learning algorithm.
One proper candidate is contrastive goal-conditioned reinforcement learning.
In contrastive goal-conditioned RL the critic is a dual encoder trained by classification on hindsight-relabeled positives and in-batch negatives.
The critic encodes reachability structure rather than a task-specific scalar return, and reachability structure is the part of the value function that should transfer across a sequence of tasks that share a robot.
Contrastive goal-conditioned RL also scales cleanly to deep residual networks, which matters because a continual agent needs enough expressivity and effective computation to keep absorbing new skills.
We therefore use contrastive goal-conditioned RL as the underlying algorithm and build a framework around it.

The framework carries the contrastive critic forward across tasks so that reachability structure is not thrown away at every task boundary.
On the actor we apply knowledge decomposition, expressing the policy at task $k$ as a frozen base plus a weighted sum of learned knowledge vectors plus a current-task vector, with weights produced by a softmax over learnable scaling parameters.
We maintain a bounded knowledge pool: when the pool exceeds its capacity, the two most cosine-similar vectors are merged by averaging.
Knowledge decomposition and bounded knowledge pools have been studied before in continual reinforcement learning, and we apply those ideas here to the actor of a contrastive goal-conditioned agent.
Continual learning has a natural offline-to-online structure: by the time the agent reaches task $k$, it already owns replay buffers from tasks $1, \ldots, k-1$, full of hindsight-relabeled goals that describe states the agent has reached in the past.
Discarding this data at the task boundary is wasteful.
We therefore feed past-task goals back into the critic as additional contrastive negatives.
A naive implementation of this idea fails, and the failure is informative: when tasks occupy disjoint workspace regions, a past-task goal is trivially distinguishable from a current-task goal by raw coordinates alone, so the critic saturates its categorical accuracy without learning useful features.
We introduce a hard-weighted negative bank that, for each anchor in the batch, scores a candidate pool of past-task goals and keeps the top few that currently receive the highest score; these hard negatives are appended to the in-batch logits with a scalar down-weight.

We evaluate the framework on a ten-task Sawyer manipulation sequence with eight million environment steps per task and five seeds per configuration.
The central study is a nine-cell ablation over actor and critic evolution modes: each is either reset at every task boundary, carried forward persistently, or decomposed.
Running all nine cells separates the contribution of actor-side mechanisms from the contribution of critic-side mechanisms, which earlier work has tended to conflate.
Four empirical questions organise the study:
  - whether persisting a contrastive critic across tasks helps, under sparse reward;
  - whether knowledge decomposition on the actor adds value on top of a persistent contrastive critic;
  - how actor-side representation quality evolves across long sequences, and whether actor plasticity loss eventually bottlenecks performance before critic forgetting does;
  - whether reusing past-task data as contrastive negatives helps, and under what filtering.

We expect the headline configuration to be a decomposed actor with a persistent contrastive critic.
Persisting the critic should improve forward transfer, extending an earlier dense-reward SAC result on critic transfer into the sparse contrastive regime.
Knowledge decomposition on the actor should drive backward transfer by converting critic-side retention into actor-side retention.
Actor plasticity loss, tracked through dormant-neuron ratio, neural-collapse statistics, and feature rank, should eventually become the binding constraint on long sequences.
A vanilla negative bank should hurt while the hard-weighted bank should help, and the gap should be explained by how much of the critic's softmax probability lands on trivially separable workspace regions versus on genuinely goal-relevant features.
The broader message is that progress in continual reinforcement learning comes from pairing a value function whose representation transfers across tasks with lightweight parameter-level mechanisms that translate that retention into behaviour, and from turning past-task replay into principled signal rather than discarding it at the task boundary.
Sparse reward is what forces this discipline, because under dense reward a well-shaped reward function can carry an algorithm to a respectable number without ever answering the representational question.

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
