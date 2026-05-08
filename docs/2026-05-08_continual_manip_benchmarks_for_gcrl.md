# Continual manipulation benchmarks for goal-conditioned contrastive RL

Date: 2026-05-08

Quick literature scan: which continual / sequential manipulation
benchmarks could host a contrastive goal-conditioned RL algorithm
besides the current 10-task Sawyer Meta-World setup the project uses?

---

## What the algorithm needs

The decomposed-critic SGCRL algorithm we are running today imposes the
following requirements on a benchmark:

1. **Goal-conditioned, with a goal that is a state-vector slice.** The
   InfoNCE critic compares `phi(s, a)` against `psi(g)` where `g` is
   index-aligned with `s`. The current setup uses
   `STATE_DIM_UNIFIED = GOAL_DIM_UNIFIED = 11` and
   `STABLE_INDICES = (0, 1, 2, 3)` (end-effector xyz + gripper).
2. **State-based observations** with a fixed dimensionality across
   tasks (the contrastive networks are MLPs over the obs vector).
3. **Sequential / continual task layout.** Multiple tasks performed
   one after another, sharing the same robot embodiment so a shared
   body has anything to share.
4. **Sparse reward + HER compatibility.** The CRL critic uses
   future-state hindsight relabeling (the existing `flatten_fn` in
   `run_continual_contrastive.py`); a benchmark whose tasks are
   intrinsically dense-reward needs a sparse-reward variant.
5. **MuJoCo or Brax compatibility** to keep the existing JAX +
   reverb + acme stack with no rewrites.

---

## Candidates

### 1. Continual-World (CW10 / CW20) — what we already use, in essence

Wolczyk et al. (NeurIPS 2021) [arXiv 2105.10919]. Built on
Meta-World V1, ten Sawyer tasks ordered by transfer matrix:
`hammer-v1, push-wall-v1, faucet-close-v1, push-back-v1,
stick-pull-v1, handle-press-side-v1, push-v1, shelf-place-v1,
window-close-v1, peg-unplug-side-v1`. CW20 is CW10 repeated twice;
the longer CW30 sequence exists in the paper's Appendix G.

**Project relevance:** the project's `CONTINUAL_TASK_SEQUENCE` in
`contrastive/continual_config.py` is **exactly the CW10 sequence**
(see [the 9 lines there](../contrastive/continual_config.py#L8-L17)).
We are already using Continual-World, just rebuilt against
state-based Meta-World V2 observations and a custom
`TaskIDGymWrapper` that pads to 11 dims.

**GCRL fit:** the original benchmark uses dense reward + SAC. Our
setup deviates by treating the next-state vector as the goal and
running InfoNCE+HER on top — this is non-standard for CW but works
because the robot embodiment is fixed across tasks. No prior CW
paper has run a contrastive GCRL algorithm on the sequence; that
combination is novel to this project.

**Drawbacks:** Meta-World V1's reward shaping was redesigned in V2
(see Meta-World+ NeurIPS 2025 [arXiv 2505.11289]); CW10 is V1-based.
Future work could port to V2 sparse rewards, but that is a separate
piece of work.

---

### 2. LIBERO

Liu et al. (NeurIPS 2023 D&B) [arXiv 2306.03310]. 130 tasks across
four suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal,
LIBERO-100 (split into LIBERO-90 pretrain + LIBERO-10 downstream).
Single Franka Panda embodiment. Designed explicitly for **lifelong
robot learning**, with controlled distribution shifts per suite.

**GCRL fit: poor.** LIBERO is a **language-conditioned** benchmark —
each task is specified by a natural-language instruction
("pick up the alphabet soup and place it in the basket"), not by a
goal-state vector. There is no obvious `obs_to_goal` map. Adopting
LIBERO would require either:

- replacing the goal encoder `psi` with a language encoder (this
  changes the algorithm), OR
- defining task-specific goal-state slices for each of the 130
  tasks (substantial manual annotation).

LIBERO is also **demonstration-heavy**: it ships with 50 human
teleop demos per task and most published baselines fine-tune from
those demos. Our setup is from-scratch RL — the demos would not be
used and the task difficulty without them is unclear.

**Verdict:** would require an algorithmic change (text or vision
goal encoder). Not a drop-in alternative.

---

### 3. CompoSuite

Mendez et al. (CoLLAs 2022) [arXiv 2207.04136]. 256 compositional
manipulation tasks built from {robot} × {object} × {objective} ×
{obstacle}. Lifelong / compositional benchmark.

**GCRL fit: medium.** CompoSuite tasks have an explicit goal in the
form of a target pose / target configuration, and observations are
state-based (proprioception + object state). The compositional
structure is appealing for the decomposed-critic story: a body that
captures generic dynamics ought to transfer across robot×object
combinations. But:

- Variation across the **robot axis** (Panda, Sawyer, Kinova3, IIWA)
  changes the obs dimensionality. Our `b_shared` assumes a fixed
  obs layout.
- The benchmark does not ship a sparse-reward variant; reward is
  dense per-step distance to target. HER could be retrofitted (the
  task does have a pose-goal), but this is engineering work.

**Verdict:** plausible second benchmark for ablations restricted to
a single-robot subset (e.g., Panda × all objects × all obstacles).
Not a drop-in but the closest in spirit to the decomposed-critic
hypothesis.

---

### 4. OGBench (Park et al., ICLR 2025)

[arXiv 2410.20092], project page seohong.me/projects/ogbench/. Eight
environment types, 85 datasets, designed specifically for
**offline goal-conditioned RL**. Manipulation suite has three
families: **Cube** (1–4 cubes), **Scene** (drawer, window, button,
cube; up to 24 atomic skills chained), **Puzzle** (4×4, 4×5, 4×6 grid
of buttons whose colour you set). UR5e arm. Both state-based and
pixel observations supported.

**GCRL fit: very high in spirit, but offline-only.** Goal is a
state-vector configuration; tasks decompose into atomic behaviours;
hindsight stitching is a first-class concern. The Scene suite
(up to 24 atomic skills per goal) is a natural sequential-skill
benchmark.

**Drawbacks:** OGBench is **offline**. Online GCRL on the same envs
is a non-trivial port — the simulator and observation space carry
over, but the published baselines and datasets do not. The
sequential / continual angle is not pre-arranged; one would have
to define a task ordering manually.

**Verdict:** the most natural future benchmark if we ever want to
study **offline** decomposed-critic, or **online** decomposed-critic
on a more complex robot than Sawyer. Not a one-week port for the
NeurIPS 2026 paper.

---

### 5. RLBench

James et al. (IROS 2020) [arXiv 1909.12271]. 100 hand-designed tasks,
Franka Panda, both proprioceptive and visual observations,
demonstration-heavy.

**GCRL fit: medium-low.** Tasks are individually specified, no
sequential/continual layout out of the box. Goal is implicit (each
task has its own success function); there is no clean
state-vector goal that mirrors `obs_to_goal_2d`. Heavy reliance on
demonstration-based methods in the published baselines.

**Verdict:** not aligned with the contrastive GCRL paradigm without
substantial benchmark-side work.

---

### 6. Meta-World ML10 / ML45

The meta-RL split of Meta-World [Yu et al. 2019]; ML10 has 10 train
+ 5 held-out tasks, ML45 has 45 + 5. Same Sawyer embodiment as
CW10.

**GCRL fit: medium.** Tasks share the embodiment; in MT mode the
observation is the state augmented with a one-hot task ID and the
**goal is part of the observation**, which is exactly the layout
our project uses. But ML10 / ML45 are designed for **few-shot
adaptation to held-out tasks**, not for studying forgetting /
forward transfer in a sequential setting.

**Verdict:** essentially the same task pool as CW10/CW20, just
rearranged for a different research question. Worth keeping as a
related-work citation but not a benchmark to adopt.

---

### 7. MTBench (Zhao et al., RLC 2025)

[`rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_140.pdf`]. 50
manipulation tasks + 20 locomotion tasks, all in IsaacGym, designed
for massively-parallelised multi-task RL. Builds on top of
Meta-World task semantics.

**GCRL fit: medium.** State-based, multi-task, Meta-World-style
goal embedding. The IsaacGym backend is GPU-native, which would
require porting our acme/reverb/MuJoCo stack to a different
physics engine. The benchmark is **multi-task**, not continual; one
would need to define a task ordering manually.

**Verdict:** more useful for scaling experiments than for the
continual-RL story this paper is about.

---

### 8. Franka Kitchen (Gymnasium-Robotics)

[Original release: relay-policy-learning, Gupta et al. 2019;
Gymnasium-Robotics docs at
robotics.farama.org/envs/franka_kitchen/]. Single Franka, kitchen
scene with 7 manipulable items (microwave, kettle, light switch,
sliding cabinet, hinge cabinet, top burner, bottom burner). Goal
is multi-task: a list of items to actuate to a desired joint
configuration.

**GCRL fit: very high in spirit.** The goal is literally a joint
configuration vector that lives in the same space as the
observation; this is the exact `obs_to_goal_2d` pattern. Sparse
reward (binary indicator per sub-task completion) is the default.

**Drawbacks:** the canonical task is **multi-task within a single
episode** (the agent is asked to actuate, e.g., the microwave AND
kettle AND light switch in one episode). Decomposing this into a
sequential continual setup is a benchmark-design choice and not
standardised. The 7-task pool is also smaller than CW10.

**Verdict:** the closest analog to our current Sawyer setup in
terms of GCRL fit, on a different robot. Most useful as a
robustness check for paper claims that depend on the embodiment
not being Sawyer-specific.

---

## Summary table

| Benchmark | Task count | Modality | Goal type | Sequential layout | HER | GCRL track record | Drop-in fit |
|---|---|---|---|---|---|---|---|
| CW10 / CW20 | 10 / 20 | state | task one-hot in obs | yes (transfer-ordered) | retrofitted | none in CW; novel here | **already in use** |
| LIBERO | 130 | vision (mostly) | language | yes (per-suite curriculum) | not natural | demo-based | poor (needs lang encoder) |
| CompoSuite | 256 | state | target config | not pre-ordered | retrofittable | none | medium (single-robot subset) |
| OGBench | varies | state + pixel | state vector | partial (Scene chains) | first-class | extensive (offline) | medium-high (offline-only as published) |
| RLBench | 100 | vision-heavy | per-task | no | not natural | demo-based | poor |
| ML10 / ML45 | 15 / 50 | state | task one-hot | meta-split, not sequential | retrofitted | some | overlap with CW |
| MTBench | 50 (manip) | state | task one-hot | not sequential | retrofittable | none | medium (IsaacGym port) |
| Franka Kitchen | 7 | state | joint-config | not sequential by default | first-class | some | high (different embodiment) |

---

## Recommendation for proposal-1 / NeurIPS 2026

- **Keep CW10 as the primary benchmark.** It is the only benchmark
  that (a) is state-based, (b) is intrinsically sequential, (c)
  shares a single robot embodiment across tasks, (d) is what the
  scaling-CRL paper [Wang et al., NeurIPS 2025
  arXiv:2503.14858] uses for its single-task GCRL evaluations on the
  same robot, so our results are directly comparable.
- **Cite Continual-World as the source of the sequence**. The
  project's `CONTINUAL_TASK_SEQUENCE` is the CW10 sequence and we
  should say so explicitly in §3 of the paper.
- **For paper robustness, add a single Franka Kitchen sequential
  experiment as an appendix.** Embodiment robustness check;
  doesn't require changing the algorithm. Defines a fixed task
  ordering of the 7 kitchen items.
- **Defer LIBERO, OGBench, CompoSuite, RLBench, MTBench.** Each
  requires either an algorithmic change (language encoder), an
  offline-to-online port, or a multi-task-to-sequential reordering,
  which are paper-sized projects.

---

## Sources

- Wolczyk et al., *Continual World*, NeurIPS 2021 — [arXiv:2105.10919](https://arxiv.org/abs/2105.10919); deepsense.ai overview at [deepsense.ai/.../continual-world](https://deepsense.ai/resource/continual-world-a-robotic-benchmark-for-continual-reinforcement-learning-2/); CW10 sequence on [Continual World project page](https://sites.google.com/view/continualworld/home).
- Liu et al., *LIBERO*, NeurIPS 2023 D&B — [arXiv:2306.03310](https://arxiv.org/abs/2306.03310); [GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO).
- Mendez et al., *CompoSuite*, CoLLAs 2022 — [arXiv:2207.04136](https://arxiv.org/abs/2207.04136); [GitHub](https://github.com/Lifelong-ML/CompoSuite).
- Park et al., *OGBench*, ICLR 2025 — [arXiv:2410.20092](https://arxiv.org/abs/2410.20092); [project page](https://seohong.me/projects/ogbench/).
- James et al., *RLBench*, IROS 2020 — [arXiv:1909.12271](https://arxiv.org/abs/1909.12271).
- Yu et al., *Meta-World*, CoRL 2019 — [Farama-Foundation/Metaworld](https://github.com/Farama-Foundation/Metaworld).
- Park et al., *Meta-World+*, NeurIPS 2025 — [arXiv:2505.11289](https://arxiv.org/html/2505.11289v1).
- Zhao et al., *MTBench*, RLC 2025 — [PDF](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_140.pdf).
- Gupta et al., *Relay Policy Learning / Franka Kitchen*, CoRL 2019 — [Gymnasium-Robotics docs](https://robotics.farama.org/envs/franka_kitchen/).
- Eysenbach et al., *Contrastive Learning as GCRL*, NeurIPS 2022 — [arXiv:2206.07568](https://arxiv.org/abs/2206.07568); [project page](https://ben-eysenbach.github.io/contrastive_rl/).
- Bortkiewicz et al., *JaxGCRL: Accelerating GCRL Algorithms and Research*, ICLR 2025 — [arXiv:2408.11052](https://arxiv.org/abs/2408.11052); [project page](https://michalbortkiewicz.github.io/JaxGCRL/).
- Wang et al., *1000 Layer Networks for Self-Supervised RL*, NeurIPS 2025 — [arXiv:2503.14858](https://arxiv.org/html/2503.14858v3); [project page](https://wang-kevin3290.github.io/scaling-crl/).
