# Continual Goal-Conditioned Contrastive RL: Algorithm Pseudocode

This document describes the complete algorithm as implemented in `section3_done`.
It combines **goal-conditioned contrastive RL** (SGCRL / "A Single Goal is All You Need")
with **CKA-RL**-style continual actor adaptation, applied to a 10-task Meta-World sequence.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| φ(s,a) | State-action encoder (sa_encoder MLP) |
| ψ(g) | Goal encoder (g_encoder MLP) |
| π_θ(a \| s,g) | Policy network parameterised by θ |
| θ_base | Frozen base policy parameters (trained on task 0) |
| v_k | Current task's knowledge vector (same pytree structure as θ_base) |
| V = {v_1, ..., v_{k-1}} | Knowledge pool of past task vectors |
| β_k | Logits for blending weights over V; α_k = softmax(β_k · α_scale) |
| α_scale | Learnable scalar temperature for α |
| θ' = θ_base + Σ_j α_j v_j + v_k | Composed policy parameters |
| τ_k | Task k (environment + fixed goal) |
| D_k | Replay buffer for task k (per-task, fresh each task) |
| B | Batch size |
| N_sgd | Number of SGD steps per learner step (scanned via lax.scan) |
| K_max | Maximum knowledge pool size before merging |
| num_tasks | Total number of tasks in the sequence (10) |

---

## Observation Layout

Each environment observation has the following structure:

```
Raw env:   [state_spatial (11),  goal_spatial (11)]
                                                   
After TaskIDWrapper:                               
           [state_spatial (11), task_one_hot (num_tasks), goal_spatial (11)]
            └──────── obs_dim (11 + num_tasks) ─────┘   └── goal_dim ──┘
```

During training, the data pipeline relabels goals using hindsight (future states
from the same trajectory). Since a goal is a future state, it has the **same
dimensionality as the state** (including the task identifier):

```
Transition observation:  [state (obs_dim),  goal (obs_dim)]
                          state includes     goal includes
                          task_one_hot       task_one_hot
```

The critic splits this observation at obs_dim:
- **sa_encoder** receives: concat(state, action) = concat([state_spatial, task_one_hot], action)
- **g_encoder** receives: goal = [goal_spatial, task_one_hot]

---

## Algorithm

### Phase 1: Base Task (k = 0)

```
INITIALISE:
  Randomly initialise policy parameters θ
  Randomly initialise critic parameters (φ, ψ)  — sa_encoder and g_encoder
  Set θ_base ← θ
  Set v_0 ← zeros_like(θ)       # knowledge vector for base task
  Set knowledge pool V ← {}
  Create fresh replay buffer D_0 for task τ_0

PREFILL D_0:
  Collect episodes in τ_0 using random policy until |D_0| ≥ prefill_size
  (prefill_size = max(min_replay_size, B × N_sgd + max_episode_steps))

TRAIN on τ_0 for base_steps environment steps:
  Repeat:
    # --- Actor step ---
    Run one full episode in τ_0 using π_{θ_base + v_0}
    Store episode in D_0

    # --- Learner step (N_sgd gradient updates via lax.scan) ---
    Sample batch of B × N_sgd transitions from D_0
    For each transition (s, a, r, s'):
      Sample a future state s_future from the same trajectory
      Extract goal: g ← obs_to_goal(s_future)    # includes task_one_hot
      Relabel: observation ← [s, g], next_observation ← [s', g]

    For each mini-batch of B transitions (scanned over N_sgd steps):
      # ---- Critic update (InfoNCE) ----
      Compute logit matrix:  L_ij = φ(s_i, a_i)^T · ψ(g_j)    for all i,j in batch
      If use_cpc:
        L_critic = mean_i [ softmax_cross_entropy(L_i, I_i) + 0.01 · logsumexp(L_i)² ]
      Else:
        L_critic = mean [ sigmoid_BCE(L, I) ]
      Update (φ, ψ) ← (φ, ψ) - η_critic · ∇_{φ,ψ} L_critic
      Update target critic: (φ̄, ψ̄) ← (1 - τ)(φ̄, ψ̄) + τ(φ, ψ)

      # ---- Actor update ----
      Compose: θ' ← θ_base + v_0
      Sample: â ~ π_{θ'}(· | s, g)
      Compute: Q(s, â, g) = -‖φ(s, â) - ψ(g)‖₂
      L_actor = -mean[ Q(s, â, g) ]
      Compute ∇_{θ'} L_actor
      # Since θ' = θ_base + v_0 and θ_base is initial params,
      # ∇_{v_0} L_actor = ∇_{θ'} L_actor  (additive parameterisation)
      Update v_0 ← v_0 - η_actor · ∇_{v_0} L_actor

AFTER BASE TASK:
  θ_base ← θ_base + v_0          # fold training delta into base
  V ← V ∪ { zeros_like(θ_base) } # pool starts with a zero vector (per pseudocode V={0})
  Save checkpoint: (θ_base, V, φ, ψ, φ̄, ψ̄, optimizer states)
  Stop replay server for D_0
```

### Phase 2: Continual Tasks (k = 1, 2, ..., num_tasks - 1)

```
For each task k = 1, 2, ..., num_tasks - 1:

  LOAD from previous checkpoint:
    θ_base (frozen), V = {v_1, ..., v_{|V|}}, (φ, ψ), (φ̄, ψ̄), critic optimizer state

  INITIALISE for task k:
    v_k ← zeros_like(θ_base)
    β_k ~ N(0, 0.01) with length |V|        # blending logits
    α_scale ← 1.0                            # learnable temperature
    Fresh optimizers for v_k, β_k, α_scale
    Create fresh replay buffer D_k for task τ_k

  PREFILL D_k:
    α_k ← softmax(β_k · α_scale)
    pool_contribution ← Σ_{j=1}^{|V|} α_j · V[j]
    θ' ← θ_base + pool_contribution + v_k
    Collect episodes in τ_k using π_{θ'} until |D_k| ≥ prefill_size

  TRAIN on τ_k for steps_per_task environment steps:
    Repeat:
      # --- Actor step ---
      Compute α_k ← softmax(β_k · α_scale)
      pool_contribution ← Σ_{j=1}^{|V|} α_j · V[j]
      θ' ← θ_base + pool_contribution + v_k
      Run one full episode in τ_k using π_{θ'}
      Store episode in D_k

      # --- Learner step ---
      Sample batch of B × N_sgd transitions from D_k (with hindsight goal relabeling)

      # Pool contribution computed OUTSIDE JIT (variable-length pool)
      α_k ← softmax(β_k · α_scale)
      pool_c ← Σ_{j=1}^{|V|} α_j · V[j]

      # Inner SGD loop (N_sgd steps via jax.lax.scan):
      Reshape transitions into N_sgd mini-batches of B
      For each mini-batch (scanned):
        # ---- Compose policy ----
        θ' ← θ_base + pool_c + v_k

        # ---- Critic update (InfoNCE) ----
        L_ij = φ(s_i, a_i)^T · ψ(g_j)
        L_critic = InfoNCE loss (same as base phase)
        Update (φ, ψ) ← (φ, ψ) - η_critic · ∇_{φ,ψ} L_critic
        Update target: (φ̄, ψ̄) ← (1 - τ)(φ̄, ψ̄) + τ(φ, ψ)

        # ---- Actor update (v_k only) ----
        â ~ π_{θ'}(· | s, g)
        Q = -‖φ(s, â) - ψ(g)‖₂
        L_actor = -mean[ Q ]
        ∇_{θ'} L_actor → used as ∇_{v_k} L_actor (additive param)
        Update v_k ← v_k - η_actor · ∇_{v_k} L_actor
        # θ_base and V are NOT updated (frozen during task k)

      # ---- β_k and α_scale update (OUTSIDE JIT, once per learner step) ----
      # Compute ∂L_actor/∂β_k and ∂L_actor/∂α_scale via jax.grad
      # through the composition: β_k → α_k → pool_c → θ' → L_actor
      Update β_k ← β_k - η_beta · ∇_{β_k} L_actor
      Update α_scale ← α_scale - η_alpha_scale · ∇_{α_scale} L_actor

  AFTER TASK k:
    V ← V ∪ { v_k }                 # append learned knowledge vector

    # ---- Pool management ----
    If |V| > K_max:
      Compute pairwise cosine similarity between all vectors in V
      (i*, j*) ← argmax_{i≠j} cosine_sim(flatten(V[i]), flatten(V[j]))
      v_merge ← (V[i*] + V[j*]) / 2
      V ← (V \ {V[i*], V[j*]}) ∪ { v_merge }

    Save checkpoint: (θ_base, V, φ, ψ, φ̄, ψ̄, critic optimizer state)
    Stop replay server for D_k
```

---

## Key Design Decisions

### 1. Critic is persistent, actor is composed
The critic (φ, ψ) is **never reset** — it carries forward across all tasks and is
fine-tuned with L_InfoNCE on each new task's data. The actor is freshly composed
for each task via θ' = θ_base + Σ α_j v_j + v_k, where θ_base is frozen after
the base phase.

### 2. Two loss functions for the critic
- **For InfoNCE (contrastive classification):** logits = φ(s,a)^T ψ(g) (inner product)
- **For actor gradient signal:** Q(s,a,g) = -‖φ(s,a) - ψ(g)‖₂ (L2 distance)

These are consistent: the inner product is used for the self-supervised contrastive
objective, and the L2 distance provides the actor's reward signal. Both use the
same encoder parameters (φ, ψ).

### 3. Task conditioning via one-hot identifier
A one-hot task vector is inserted between state and goal in the observation, so both
the state-action encoder φ(s,a) and the goal encoder ψ(g) see the task identity.
The goal is a future state from the same task, so it naturally carries the same
task identifier.

### 4. Per-task replay buffer
Each task gets a fresh Reverb server and replay buffer. No experience leaks across
tasks. The buffer is filled from scratch (prefill phase) before training begins.

### 5. lax.scan for inner SGD loop
The N_sgd gradient steps per learner step are executed via `jax.lax.scan`,
matching the original SGCRL learner pattern. The pool contribution is computed
outside JIT (because the pool has variable length) and broadcast across all
N_sgd steps within a single `lax.scan` call.

### 6. β_k gradients outside JIT
Because the pool length varies across tasks, the gradient of the actor loss with
respect to β_k and α_scale is computed outside JAX JIT, using the last mini-batch
from the lax.scan loop.

---

## Mapping to Code

| Pseudocode concept | Implementation file | Function / class |
|--------------------|-------------------|-----------------|
| Outer task loop | `run_continual_contrastive.py` | `main()` |
| Single task training | `run_continual_contrastive.py` | `train_single_task()` |
| θ' composition | `contrastive/continual_learning.py` | `_compute_pool_contribution()`, `update_step()` |
| InfoNCE critic loss | `contrastive/continual_learning.py` | `critic_loss_fn()` |
| L2 actor loss | `contrastive/continual_learning.py` | `actor_loss_fn()` |
| lax.scan inner loop | `contrastive/continual_learning.py` | `_scan_update()` |
| β_k / α_scale update | `contrastive/continual_learning.py` | `_update_beta_and_alpha_scale()` |
| Knowledge pool | `contrastive/knowledge_pool.py` | `KnowledgePool` |
| Pool merging | `contrastive/knowledge_pool.py` | `merge_if_needed()` |
| Task ID wrapper | `contrastive/utils.py` | `TaskIDWrapper` |
| Hindsight goal relabeling | `run_continual_contrastive.py` | `flatten_fn()` |
| φ(s,a) and ψ(g) networks | `contrastive/networks.py` | `_repr_fn()` |
| Policy network | `contrastive/networks.py` | `_actor_fn()` |
| Checkpointing | `run_continual_contrastive.py` | `save_ckpt()`, `load_ckpt()` |
