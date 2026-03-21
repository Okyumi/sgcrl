# Continual Goal-Conditioned Contrastive RL: Algorithm Pseudocode

This document describes the complete algorithm as implemented in `section3_done`.
It combines **goal-conditioned contrastive RL** (SGCRL) with **CKA-RL**-style
continual actor adaptation, applied to a 10-task Meta-World sequence.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| φ(s,a) | State-action encoder (sa_encoder MLP), input = concat(state, action) |
| ψ(g) | Goal encoder (g_encoder MLP) |
| φ(s,a)^T ψ(g) | Inner-product critic score (same as SGCRL) |
| π_θ(a \| s,g) | Policy network parameterised by θ |
| θ_base | Frozen base policy parameters (trained on task 0) |
| v_k | Current task's knowledge vector (same pytree structure as θ_base) |
| V = {v_1, ..., v_{k-1}} | Knowledge pool of past task vectors |
| β_k | Logits for blending weights over V; α_k = softmax(β_k · α_scale) |
| θ' = θ_base + Σ_j α_j v_j + v_k | Composed policy parameters |
| τ_k | Task k (environment + fixed goal) |
| D_k | Replay buffer for task k (fresh each task) |
| B | Batch size |
| N_sgd | SGD steps per learner step (via lax.scan) |
| K_max | Maximum knowledge pool size before merging |

---

## Observation Layout

Task ID is appended at the gym level in `env_utils.py` (via `TaskIDGymWrapper`),
to **both** state and goal vectors:

```
Raw env:   [state_spatial (11),  goal_spatial (11)]

After TaskIDGymWrapper:
           [state_spatial (11), task_one_hot (10), goal_spatial (11), task_one_hot (10)]
           |---- state (obs_dim = 21) ----|  |---- goal (obs_dim = 21) ----|
```

State and goal have **identical dimensionality** (both include task_id).
This is consistent with hindsight relabeling: a goal is a future state
from the same task, so it naturally carries the same task identifier.

The contrastive critic splits the observation at `obs_dim`:
- **sa_encoder φ** receives: `concat(state, action)` where state includes task_id
- **g_encoder ψ** receives: `goal` which includes task_id

---

## Actor Loss: Inner Product (matching SGCRL)

The actor loss uses the **inner product** from the contrastive critic, exactly
as in the SGCRL paper (Algorithm 1):

```
q_action = φ(s_i, a_i)^T ψ(g_j)     [B × B logit matrix]
actor_loss = -diag(q_action)          [maximize diagonal = matched pairs]
```

This is the same score used in the InfoNCE critic loss. The SGCRL paper
ablation (Section 4.2) shows that the inner-product critic architecture
is essential for effective exploration. We do NOT use L2 distance.

---

## Algorithm

### Phase 1: Base Task (k = 0)

```
INITIALISE:
  Randomly init policy params θ, critic params (φ, ψ)
  θ_base ← θ,  v_0 ← zeros_like(θ),  V ← {}
  Create fresh replay D_0 for τ_0

PREFILL D_0:
  Collect episodes using random policy until |D_0| ≥ prefill_size

TRAIN on τ_0 for base_steps env steps:
  Repeat:
    # --- Actor step ---
    Run one episode in τ_0 using π_{θ_base + v_0}, store in D_0

    # --- Learner step (N_sgd updates via lax.scan) ---
    Sample B × N_sgd transitions from D_0
    For each (s, a, r, s'), sample future state s_f from same trajectory,
      set goal g ← obs_to_goal(s_f)  [g includes task_id, same dim as s]
      Relabel: obs ← [s, g],  next_obs ← [s', g]

    For each mini-batch of B (scanned):
      # Critic update (InfoNCE)
      L_ij = φ(s_i, a_i)^T ψ(g_j)       ∀ i,j in batch
      L_critic = InfoNCE(L, I) + 0.01 · logsumexp(L)²
      Update (φ, ψ)
      Target update: (φ̄, ψ̄) ← (1-τ)(φ̄, ψ̄) + τ(φ, ψ)

      # Actor update
      θ' ← θ_base + v_0
      â ~ π_{θ'}(·|s,g)
      actor_loss = -diag(φ(s, â)^T ψ(g))    [inner product]
      Update v_0 via ∇_{v_0} actor_loss

AFTER BASE TASK:
  θ_base ← θ_base + v_0       [fold delta into base]
  V ← { zeros_like(θ_base) }  [pool starts with zero vector]
  Save checkpoint, stop D_0
```

### Phase 2: Continual Tasks (k = 1, ..., N-1)

```
For each task k:
  LOAD: θ_base (frozen), V, (φ, ψ), (φ̄, ψ̄), critic optimizer state
  INIT: v_k ← 0,  β_k ~ N(0, 0.01) of length |V|,  α_scale ← 1.0
  Create fresh replay D_k for τ_k

  TRAIN on τ_k for steps_per_task env steps:
    Repeat:
      # Actor step
      α_k ← softmax(β_k · α_scale)
      pool_c ← Σ_j α_j · V[j]          [outside JIT: variable-length pool]
      θ' ← θ_base + pool_c + v_k
      Run one episode with π_{θ'}, store in D_k

      # Learner step
      Sample transitions, relabel goals (hindsight)
      pool_c ← Σ_j α_j · V[j]          [recompute outside JIT]

      lax.scan over N_sgd mini-batches:
        θ' ← θ_base + pool_c + v_k

        # Critic: InfoNCE (same as base phase)
        L_ij = φ(s_i, a_i)^T ψ(g_j)
        Update (φ, ψ)

        # Actor: inner product (only v_k gets gradients)
        â ~ π_{θ'}(·|s,g)
        actor_loss = -diag(φ(s, â)^T ψ(g))
        ∇_{v_k} actor_loss = ∇_{θ'} actor_loss   [additive param]
        Update v_k
        [θ_base and V are frozen]

      # β_k, α_scale update (outside JIT, once per step)
      ∇_{β_k} actor_loss, ∇_{α_scale} actor_loss via jax.grad
      Update β_k, α_scale

  AFTER TASK k:
    V ← V ∪ { v_k }
    If |V| > K_max:
      (i*,j*) ← argmax cosine_sim(V[i], V[j])
      v_merge ← (V[i*] + V[j*]) / 2
      V ← (V \ {V[i*], V[j*]}) ∪ { v_merge }
    Save checkpoint, stop D_k
```

---

## Key Design Decisions

### 1. Inner-product critic (matching SGCRL)
Both the InfoNCE loss and the actor loss use φ(s,a)^T ψ(g). The SGCRL paper
(Section 4.2) shows the inner-product architecture is crucial — it drives
exploration through the learned contrastive representations.

### 2. Persistent critic, composed actor (matching GCRL_plan.md)
The critic (φ, ψ) is never reset across tasks. The actor is freshly composed
per task via θ' = θ_base + Σ α_j v_j + v_k. This follows the CKA-RL design
for the actor and extends SGCRL's contrastive critic to the continual setting.

### 3. Task ID in env_utils.py (both state and goal)
A one-hot task vector is appended to both state and goal at the gym level.
Since goal = future state, both have identical dims. CKA-RL does NOT use
task_id in observations (its SAC critic is reinitialized per task), but our
persistent contrastive critic needs it to distinguish tasks.

### 4. Per-task replay buffer
Fresh Reverb server per task. No cross-task data leakage.

### 5. DistanceObserver removed
`DistanceObserver` is a monitoring utility from the original SGCRL code that
measures L2 distance to goal during episodes. It is not part of the algorithm.
We keep only `SuccessObserver` which tracks task success rate.

---

## Code Mapping

| Concept | File | Location |
|---------|------|----------|
| Task ID wrapper | `env_utils.py` | `TaskIDGymWrapper` |
| Environment creation | `contrastive/utils.py` | `make_environment()` |
| Outer task loop | `run_continual_contrastive.py` | `main()` |
| Per-task training | `run_continual_contrastive.py` | `train_single_task()` |
| θ' composition | `contrastive/continual_learning.py` | `_compute_pool_contribution()`, `update_step()` |
| InfoNCE critic loss | `contrastive/continual_learning.py` | `critic_loss_fn()` |
| Inner-product actor loss | `contrastive/continual_learning.py` | `actor_loss_fn()` |
| lax.scan inner loop | `contrastive/continual_learning.py` | `_scan_update()` |
| β_k / α_scale update | `contrastive/continual_learning.py` | `_update_beta_and_alpha_scale()` |
| Knowledge pool + merge | `contrastive/knowledge_pool.py` | `KnowledgePool` |
| Hindsight goal relabeling | `run_continual_contrastive.py` | `flatten_fn()` |
| φ(s,a), ψ(g) networks | `contrastive/networks.py` | `_repr_fn()`, `_combine_repr()` |
| CKA-RL reference | `cka-rl-meta-world/models/cka_rl.py` | `CkaRlAgent`, `FuseLinear` |
