# Continual Goal-Conditioned Contrastive RL: Algorithm Pseudocode

Combines **SGCRL** (contrastive goal-conditioned RL) with **CKA-RL** (continual knowledge adaptation).

---

## Notation

| Symbol | Meaning |
|--------|---------|
| φ(s,a) | State-action encoder (sa_encoder MLP) |
| ψ(g) | Goal encoder (g_encoder MLP) |
| φ(s,a)^T ψ(g) | Inner-product critic score (SGCRL) |
| π_θ(a \| s,g) | Policy: body MLP + NormalTanhDistribution head |
| θ_base | Frozen base policy (trained on task 0) |
| v_k | Current task's knowledge vector (head-only or full-policy) |
| V = {v_1, ..., v_{k-1}} | Actor knowledge pool |
| α_k = softmax(β_k · α_scale) | Blending weights over V |
| θ' = θ_base + Σ_j α_j v_j + v_k | Composed policy parameters |
| q_base | Frozen base critic (critic_mode=cka only) |
| W = {w_1, ..., w_{k-1}} | Critic knowledge pool (critic_mode=cka only) |
| w_k | Current task's critic vector (critic_mode=cka only) |
| D_k | Per-task replay buffer (fresh each task) |

---

## Phase 1: Base Task (k = 0)

The base task trains exactly like standard SGCRL — full policy, no decomposition.

```
INIT: θ, (φ, ψ) random.  v_0 ← 0.  V ← {}.  Fresh D_0.

TRAIN on τ_0 for base_steps:
  Repeat:
    Run one episode with π_{θ+v_0}, store in D_0
    Sample batch, relabel goals (hindsight future states)
    lax.scan over N_sgd mini-batches:
      CRITIC: L_InfoNCE with softmax CE + logsumexp penalty
        Update (φ, ψ)
        Target update: (φ̄, ψ̄) ← soft_update(φ, ψ)
      ACTOR: loss = -diag(φ(s, â)^T ψ(g))     [inner product]
        Update v_0 with FULL gradients (no masking, all layers)

AFTER TASK 0:
  θ_base ← θ + v_0           [fold delta into base; θ_base is fully trained]
  V ← { zeros_like(θ_base) }
  If critic_mode='cka': q_base ← (φ, ψ)  [freeze critic base]
  Save checkpoint.
```

Note: `adapt_heads_only` and `encoder_from_base` have NO effect during task 0.
The base phase always trains the complete policy (body + head).

## Phase 2: Continual Tasks (k = 1, ..., N-1)

```
For each task k:
  LOAD: θ_base (frozen), V, (φ, ψ), optimizer states

  INIT:
    v_k ← 0 (head-only if adapt_heads_only=True)
    β_k ~ N(0, 0.01), length |V|
    α_scale ← 1.0
    Fresh D_k

    Critic init depends on critic_mode:
      persistent → (φ, ψ) = carry forward from task k-1
      reset      → (φ, ψ) = random reinit
      cka        → (φ, ψ) = q_base + Σ α_j w_j (composed from base + pool)

  TRAIN on τ_k:
    Repeat:
      α_k ← softmax(β_k · α_scale)
      pool_c ← Σ_j α_j · V[j]            [outside JIT]
      θ' ← θ_base + pool_c + v_k
      Run one episode with π_{θ'}, store in D_k

      lax.scan over N_sgd mini-batches:
        CRITIC: InfoNCE (same as base phase)
        ACTOR: loss = -diag(φ(s, â)^T ψ(g))
          If adapt_heads_only: zero body gradients
          Update v_k (θ_base and V frozen)
      β_k, α_scale update (outside JIT)

      Every eval_every steps: evaluate on tasks 0..k

  AFTER TASK k:
    V ← V ∪ { v_k }
    If |V| > K_max: merge most-similar pair (cosine)
    If critic_mode='cka':
      w_k ← (φ, ψ) - q_base - critic_pool_c
      W ← W ∪ { w_k }, merge if |W| > K_max
    Evaluate on all tasks 0..k (forgetting measurement)
    Save checkpoint.
```

## K-Sample-Argmax Evaluation

```
Given policy params θ', critic params (φ, ψ), state s, goal g:
  For k = 1..K: sample a_k ~ π_{θ'}(·|s, g)
  Score: q_k = φ(s, a_k)^T ψ(g)   [inner product]
  Select: a = a_{argmax_k q_k}
```

---

## Code Mapping

| Concept | File | Location |
|---------|------|----------|
| Task ID wrapper | `env_utils.py` | `TaskIDGymWrapper` |
| Environment creation | `contrastive/utils.py` | `make_environment()` |
| Outer task loop | `run_continual_contrastive.py` | `main()` |
| Per-task training | `run_continual_contrastive.py` | `train_single_task()` |
| Cross-task evaluation | `run_continual_contrastive.py` | `evaluate_on_task()` |
| K-sample-argmax | `contrastive/networks.py` | `apply_policy_k_sample_argmax()` |
| θ' composition | `contrastive/continual_learning.py` | `_compute_pool_contribution()` |
| Head-only gradient mask | `contrastive/continual_learning.py` | `_mask_leaf()` in `update_step()` |
| InfoNCE critic loss | `contrastive/continual_learning.py` | `critic_loss_fn()` |
| Inner-product actor loss | `contrastive/continual_learning.py` | `actor_loss_fn()` |
| lax.scan inner loop | `contrastive/continual_learning.py` | `_scan_update()` |
| Critic CKA composition | `contrastive/continual_learning.py` | `_compute_critic_pool_contribution()` |
| Knowledge pool + merge | `contrastive/knowledge_pool.py` | `KnowledgePool` |
| Hindsight goal relabeling | `run_continual_contrastive.py` | `flatten_fn()` |
| φ(s,a), ψ(g) networks | `contrastive/networks.py` | `_repr_fn()`, `_combine_repr()` |
| CKA-RL reference | `cka-rl-meta-world/models/cka_rl.py` | `CkaRlAgent`, `FuseLinear` |
| 10/20-task sequences | `contrastive/continual_config.py` | `CONTINUAL_TASK_SEQUENCE`, `CONTINUAL_TASK_SEQUENCE_20` |
