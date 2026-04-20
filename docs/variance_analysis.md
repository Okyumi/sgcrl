# Variance Analysis — April 20, 2026

## Observation

High inter-seed variance in the continual contrastive RL method. One seed shows dramatically worse performance (never reaches success rate 1.0 over 8M steps) while others converge quickly. The bad seed shows:
- Actor weight norm ~47 vs ~600-700 for healthy seeds
- Much lower actor final layer norm
- Lower critic feature rank
- Suspected high actor dormant ratio

## Root Cause Analysis

After re-reading the full algorithm (initialization, training loop, CKA decomposition, pool mechanics), here are the sources of variance and their amplification mechanisms.

### 1. Actor Initialization Lottery (PRIMARY CAUSE)

At task 0, the actor is initialized via `networks.policy_network.init(key_policy)` where `key_policy` comes from `jax.random.split(PRNGKey(seed + task_id * 1000))`. Different seeds → different initial weights.

The ResidualMLP actor has skip connections: `x_out = x_in + block(x_in)`. If the initial block weights are small (which happens stochastically), the output is dominated by the skip connection (identity), and the network behaves nearly linearly. This creates a **low-expressiveness trap**:
- Low weight norms early → small gradients (especially for Swish, where `swish'(x) ≈ 0` for `x ≈ 0`)
- Small gradients → weights stay small → norms stay low
- The actor can't produce diverse enough actions → poor exploration
- Poor exploration → the critic receives low-quality data → critic features collapse

This is a self-reinforcing feedback loop. The automatic actor reset (dormancy-triggered) is designed to break it.

### 2. Contrastive Loss Amplifies Early Randomness

The CPC InfoNCE loss:
```
L = -log(exp(φ(s,a)·ψ(g*)) / Σ_g' exp(φ(s,a)·ψ(g')))
```

This is a **softmax over the batch**. The difficulty of the contrastive task depends on how similar the negative goals are to the positive goal. Early in training, the encoder representations are random, so all goals look equally similar (scores ≈ 0). As the encoder improves, it must discriminate increasingly fine-grained goal differences.

The critical moment is the **first few thousand gradient steps**: if the actor produces poor exploratory actions (because its initialization is in a low-norm regime), the critic receives data where the agent never actually reaches diverse states. The contrastive loss can then be "solved" trivially (all negatives are identical because the agent stays in one place) without learning useful representations. This is a form of **representation collapse** driven by the actor-critic coupling.

### 3. The v_k Additive Decomposition Doesn't Help Early Training

At task 0, the composed policy is:
```
π = θ_base + 0 + v_k    (pool is empty, pool_c = 0)
```

Where `θ_base = policy_network.init(key_policy)` and `v_k = zeros`. The actor gradients update `v_k`, so the effective policy is `θ_base + v_k`. This is equivalent to standard training with initialization `θ_base`. The additive decomposition doesn't change the variance properties at task 0 — it's exactly the same as training a single network from random init.

### 4. The Replay Buffer Compounds the Problem

The replay buffer stores trajectories from ALL stages of training. If the actor's early trajectories are poor (due to bad init), these bad trajectories persist in the buffer and dilute the good data collected later. With `min_replay_size=1000` and `max_replay_size` >> `min_replay_size`, the early bad data gets replayed many times during the critical initial training phase.

This is the **primacy bias** ([Nikishin et al., 2022](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf)): overfitting to early (poor) data prevents the agent from benefiting from later (better) data.

### 5. No Entropy Regularization

The current setup uses `entropy_coefficient=0.0` (no entropy bonus). In standard SAC, entropy regularization encourages diverse actions even when the Q-function is poorly calibrated. Without it, the actor can converge to deterministic (or near-deterministic) actions early on, especially if the Q-function is noisy. This locks in the bad initialization's action distribution.

## Ideas for Reducing Variance

### A. Automatic Actor Reset (ALREADY IMPLEMENTED)

The dormancy-triggered reset (`--actor_auto_reset`) detects when the actor's trunk has ≥10% dormant neurons and reinitializes from scratch. This directly breaks the low-norm trap for bad seeds. If the actor is healthy, no reset fires.

**Status**: Already implemented and pushed.

### B. Early-Phase Entropy Injection

Add a non-zero entropy coefficient for the first N steps of task 0 only, then anneal to 0. This ensures diverse exploration during the critical early phase regardless of initialization quality.

```
α(t) = α_init * max(0, 1 - t/T_anneal)    for task 0
α(t) = 0                                     for task 1+
```

With `α_init = 0.1` and `T_anneal = 500K`, the actor is encouraged to explore broadly for the first 500K steps before settling into the deterministic regime.

**Pros**: Simple, directly addresses the exploration-initialization coupling.
**Cons**: Introduces 2 new hyperparameters. May slow convergence for good seeds.

### C. Spectral Normalization of the Actor

Spectral normalization constrains the Lipschitz constant of each layer, preventing the network from having extreme weight norms (both too large AND too small). This reduces sensitivity to initialization scale.

**Pros**: Addresses the root cause (weight norm variance).
**Cons**: Changes the optimization dynamics globally, not just for bad seeds. May hurt performance.

### D. Warmup Learning Rate for the Actor

Use a linear warmup for the actor learning rate: start at 0 and ramp to `actor_learning_rate` over the first N steps. This prevents large early updates that could lock in a bad trajectory.

```
lr(t) = actor_lr * min(1, t/T_warmup)
```

**Pros**: Standard technique in supervised learning, well-understood.
**Cons**: Slows early learning for ALL seeds, including good ones.

### E. Replay Buffer Priority or Age-Weighting

Weight replay samples by recency, giving more weight to recent (better) transitions. This reduces the primacy bias by ensuring the actor trains on fresh data rather than stale early trajectories.

**Pros**: Directly addresses the primacy bias mechanism.
**Cons**: Adds complexity; PER is expensive; uniform sampling is simpler to analyze.

### F. Twin-Actor with Best Selection

Initialize two actors with different seeds. After a warmup period, evaluate both and keep the better one. This halves the probability of a bad initialization at the cost of 2× actor compute during warmup.

**Pros**: Simple, doesn't change the algorithm, just the initialization procedure.
**Cons**: 2× actor cost during warmup. Doesn't address the fundamental issue.

## Recommendation

**Short-term (no new experiments needed)**: The automatic actor reset (A) is already in place. Run the batch grid with it enabled and compare variance to the previous runs.

**Medium-term (one new hyperparameter)**: Early-phase entropy injection (B) is the most principled approach. It directly addresses the exploration-initialization coupling without changing the algorithm's steady-state behavior. Implement as `--task0_entropy_init=0.1 --task0_entropy_anneal=500000`.

**Long-term (if variance persists)**: Investigate whether the contrastive loss itself is contributing to representation collapse under poor exploration. The logsumexp penalty (`--logsumexp_penalty=0.01`) is meant to prevent this, but it may not be sufficient when the actor provides uniformly bad data.
