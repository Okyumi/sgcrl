# Scaling-CRL and JaxGCRL Environments — Specs and Continual-Learning Feasibility

This document catalogues every environment used in the scaling-CRL study
([Wang et al., 2025, NeurIPS Best Paper](https://arxiv.org/abs/2503.14858)).
All of its environments come from the JaxGCRL benchmark
([Bortkiewicz et al., 2025, ICLR](https://arxiv.org/abs/2408.11052)),
plus extensions released in the JaxGCRL repository. Specs are verified
against the code at
[github.com/MichalBortkiewicz/JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL)
(the `jaxgcrl/envs/` directory and the XML assets).

## Environment specifications

| Environment | Embodiment | Task category | Description | Obs dim | Action dim | Goal dim | Goal definition | Horizon |
|---|---|---|---|---|---|---|---|---|
| **Ant**              | Ant, 4-legged quadruped | Locomotion (open field) | Walk to a goal sampled on a circle of radius 10 around the start. | 29 | 8 | 2 | $(x,y)$ of the torso (`goal_indices=[0,1]`) | 1000 |
| **Ant U-Maze**       | Ant | Navigation | Navigate a U-shaped maze; goals from a listed set. | 29 | 8 | 2 | Torso $(x,y)$ | 1000 |
| **Ant Big Maze**     | Ant | Navigation | Larger multi-corridor maze; goals from a listed set. | 29 | 8 | 2 | Torso $(x,y)$ | 1000 |
| **Ant Hardest Maze** | Ant | Navigation | Hardest maze topology in the suite. | 29 | 8 | 2 | Torso $(x,y)$ | 1000 |
| **Ant Push**         | Ant + movable box | Navigation + manipulation | Push a box out of the way (wrong direction makes the task unsolvable) to reach the goal. | 36 (ant + box pose) | 8 | 2 | Torso $(x,y)$ | 1000 |
| **Ant Soccer** (a.k.a. Ant Ball) | Ant + sphere | Manipulation via locomotion | Push a sphere to a goal on a circle of radius 5 around the start. | 31 (ant + ball pose) | 8 | 2 | Sphere $(x,y)$ (`goal_indices=[29,30]`) | 1000 |
| **Ant Ball Maze**    | Ant + sphere | Navigation + manipulation | Push a sphere through a maze. | 31 | 8 | 2 | Sphere $(x,y)$ (`goal_indices=[28,29]`) | 1000 |
| **Humanoid**         | Humanoid (MuJoCo, 21-DoF) | Locomotion | Walk to a goal sampled on a disc of radius 1–5 around the start. | 268 | 17 | 3 | Pelvis $(x,y,z)$ (`goal_indices=[0,1,2]`) | 1000 |
| **Humanoid U-Maze**  | Humanoid | Navigation | Navigate a U-maze. | 268 | 17 | 3 | Pelvis $(x,y,z)$ | 1000 |
| **Humanoid Big Maze**| Humanoid | Navigation | Larger maze. | 268 | 17 | 3 | Pelvis $(x,y,z)$ | 1000 |
| **Reacher**          | 2-segment arm | Manipulation (2-D) | Move the end-effector to a target on a disc of radius 0–0.2. | 11 | 2 | 3 | Target position (`goal_indices=[4,5,6]`) | 1000 |
| **Half-Cheetah**     | 2-D cheetah | Locomotion | Reach one of two fixed locations on either side of start. | 18 | 6 | 1 | Root $x$ (`goal_indices=[0]`) | 1000 |
| **Pusher** (Pusher Easy / Pusher Hard) | 3-D robotic arm + puck | Manipulation | Use the arm to push a movable puck to a goal position. | 20 | 7 | 3 | Puck $(x,y,z)$ (`goal_indices=[10,11,12]`) | 1000 |
| **Pusher 2**         | 3-D arm + two objects | Manipulation | Two-object pushing variant. | 24 | 7 | 6 | Two-object positions (`goal_indices=[10..15]`) | 1000 |
| **Arm Reach** (Panda) | 7-DoF Panda + gripper | Manipulation | Move the end-effector to a target position. | 13 obs + 3 goal | 4 (joints 1,2,4,6) | 3 | End-effector $(x,y,z)$ (`goal_indices=[7,8,9]`) | varies |
| **Arm Grasp** (Panda) | Panda + cube | Manipulation | Grasp a cube and hold it at a target pose. | 24 obs + 7 goal | 5 (joints 1,2,4,6 + finger) | 7 | Cube pose + orientation | varies |
| **Arm Push Easy** (Panda) | Panda + cube | Manipulation | Push a cube to a target position. | 18 obs + 3 goal | 5 | 3 | Cube $(x,y,z)$ | varies |
| **Arm Push Hard** (Panda) | Panda + cube | Manipulation | Harder push variant. | 18 obs + 3 goal | 5 | 3 | Cube $(x,y,z)$ | varies |
| **Arm Binpick Easy** (Panda) | Panda + cube + bin | Manipulation | Pick a cube and place it in a bin. | 18 obs + 3 goal | 5 | 3 | Cube $(x,y,z)$ | varies |
| **Arm Binpick Easy EEF** (Panda) | Panda (EEF control) + cube + bin | Manipulation | Bin-picking with delta-EEF control. | 11 obs + 3 goal | 4 (ΔEEF $x,y,z$ + gripper) | 3 | Cube $(x,y,z)$ | varies |
| **Arm Binpick Hard** (Panda) | Panda + cube + bin | Manipulation | Harder bin-pick. | 18 obs + 3 goal | 5 | 3 | Cube $(x,y,z)$ | varies |
| **Simple Maze**      | 2-D point | Navigation | Point navigation through a maze. | 4 | 2 | 2 | Point $(x,y)$ | 1000 |

Observation dims for the Ant / Humanoid / Pusher / Half-Cheetah families
reflect the JaxGCRL implementation: Ant = `qpos[2:15] + qvel[14] + target(2)` = 29;
Humanoid follows the deep-mind humanoid state of 268 dims; Pusher
follows the MuJoCo Pusher-v2 layout (23 qpos/qvel + 3-dim target = 20
for the standard variant). The scaling-CRL paper's Table 1 reports a
single "Dim" column that conflates a few different notions (it lists
268 for Humanoid and 61 for Ant Big Maze, which includes the maze
state), so the table above uses the exact numbers from the code.

## Embodiment groups by compatible action / state shape

Constructing a continual-learning sequence requires the action space and
the state space of the policy network to be compatible across tasks.
The environments cluster cleanly into four such groups.

**Ant group (action dim 8, obs dim 29).**
Ant, Ant U-Maze, Ant Big Maze, Ant Hardest Maze. All share the same
body, the same action dim, and the same obs dim. The goal is always
the torso $(x,y)$. These string together without any observation or
action padding.

**Ant-with-object group (action dim 8, obs dim 31–36).**
Ant Soccer, Ant Ball Maze, Ant Push. Same body and action dim as the
Ant group, but the observation is extended with the moved object's
pose (and, for Ant Push, the box pose). The goal varies: torso
$(x,y)$ for Ant Push, object $(x,y)$ for Ant Soccer / Ant Ball Maze.

**Humanoid group (action dim 17, obs dim 268).**
Humanoid, Humanoid U-Maze, Humanoid Big Maze. Shared body, action,
and obs dims. Goal is pelvis $(x,y,z)$ in all three.

**Panda arm group (action dim 4–5, obs dim 13–24).**
Arm Reach, Arm Grasp, Arm Push Easy / Hard, Arm Binpick Easy / Hard,
Arm Binpick Easy EEF. The 4-vs-5-dim action split separates EEF
control (Reach and Binpick-EEF: 4-dim) from joint-angle control
(Grasp, Push, Binpick: 5-dim). Obs dims range from 11 to 24
depending on how many objects are present.

## Continual-learning feasibility

A continual-learning sequence is feasible within any single embodiment
group. Across groups, the action and observation spaces differ enough
that a shared policy network cannot simply carry forward — padding,
masking, or per-task adapters would be required.

**Recommended sequences.**

- *Ant navigation sequence.* Ant $\to$ Ant U-Maze $\to$ Ant Big Maze
  $\to$ Ant Hardest Maze. Four tasks, identical shapes, increasing
  navigation difficulty. This is the cleanest continual sequence the
  suite offers and is the most direct analogue of the ten-task
  Continual World / Sawyer sequence we already use.
- *Ant locomotion + manipulation sequence.* Ant $\to$ Ant Soccer $\to$
  Ant Ball Maze $\to$ Ant Push. Same body and action dim; obs dim
  changes by a small, predictable amount (29 $\to$ 31 $\to$ 31 $\to$
  36) as objects are added. This sequence requires a small amount of
  observation padding (zero-fill object pose when absent) and a fixed
  goal-indices convention. Useful as a stress test of representation
  transfer across whether-there-is-an-object changes.
- *Humanoid navigation sequence.* Humanoid $\to$ Humanoid U-Maze $\to$
  Humanoid Big Maze. Three tasks, same shapes, increasing navigation
  difficulty. Humanoid is also the setting in which scaling-CRL
  reports its largest effects (50$\times$ on Humanoid U-Maze), which
  makes it the most interesting target for a continual study if
  we want our framework to compose with deep scaling.
- *Panda manipulation sequence (joint control).* Arm Reach $\to$
  Arm Push Easy $\to$ Arm Grasp $\to$ Arm Binpick Easy $\to$
  Arm Push Hard $\to$ Arm Binpick Hard. Six tasks, all 5-dim joint
  action (swap Reach for a 5-dim variant or accept one 4-dim entry).
  Goal dim varies (3 vs. 7); a small amount of goal padding is
  needed. This is the closest analogue in JaxGCRL to our current
  Meta-World Sawyer sequence.

**Sequences that need padding or masking.**

- *Mixed-embodiment sequences* (e.g. Ant $\to$ Humanoid, Ant $\to$
  Panda) require separate action heads per embodiment and an
  observation encoder that tolerates different input dims. These are
  feasible but cease to be drop-in sequences; they would turn the
  continual study into a multi-embodiment study, which is a different
  research question.

**Sequences that are unsuitable.**

- *Mixing EEF-control and joint-control Panda tasks* without an action
  head switch will not work cleanly (4 vs. 5-dim action).
- *Reacher, Half-Cheetah, Pusher, Pusher 2* each have their own
  embodiment and shape; they are stand-alone tasks in JaxGCRL and do
  not concatenate naturally into a continual sequence with each
  other.

## Practical takeaway

The scaling-CRL paper is built on the JaxGCRL suite, which contains
four embodiment groups. Three of those four groups already provide
shape-compatible task sequences that can be strung together into a
continual-learning curriculum without touching the policy network:

- Ant navigation (4 tasks), the cleanest option;
- Humanoid navigation (3 tasks), the most interesting option for
  composing with deep-network scaling;
- Panda manipulation under joint control (5–6 tasks), the closest
  analogue of the Continual World sequence.

The Ant-with-object extension adds one more option at the cost of a
small, systematic observation padding. Any sequence that crosses
embodiments introduces the larger question of action-head routing and
is a different experiment.

## References

- [Wang, K., Javali, A., Bortkiewicz, M., Trzciński, T., Eysenbach, B. *1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities.* NeurIPS 2025 (Best Paper).](https://arxiv.org/abs/2503.14858)
- [Bortkiewicz, M., Pałucki, W., Myers, V., Dziarmaga, T., Arczewski, T., Kuciński, Ł., Eysenbach, B. *Accelerating Goal-Conditioned RL Algorithms and Research (JaxGCRL).* ICLR 2025.](https://arxiv.org/abs/2408.11052)
- [JaxGCRL source code.](https://github.com/MichalBortkiewicz/JaxGCRL)
