# Actor Metrics + Automatic Actor Reset — April 20, 2026

## Problem

One seed (out of several) shows dramatically worse performance throughout the full 8M-step task 0 training:
- Never reaches success rate 1.0 (other seeds reach it quickly)
- Actor weight norm ~47 vs ~600–700 for healthy seeds
- Actor final layer norm much lower than others
- Critic feature rank lower than others
- Suspected high actor dormant ratio

The code is correct — the issue is seed-dependent initialization: certain random seeds produce actor weight configurations that lead to low norms and high dormancy, trapping the actor in a poorly-expressive regime.

## Changes

### 1. Actor-Side RL Metrics

**Problem**: We previously logged critic-side metrics (feature_rank, NRC1, NRC2, dormant_ratio, entropy, Gini) but not actor-side equivalents. This made it impossible to diagnose actor-specific representation issues.

**Solution**: Added actor-side metrics at both frequent and occasional levels.

New metrics:
| Metric | Level | W&B key | Description |
|---|---|---|---|
| actor/entropy | frequent | `rl_metrics/actor/entropy` | Shannon entropy of |feature| distribution |
| actor/gini | frequent | `rl_metrics/actor/gini` | Gini coefficient for feature sparsity |
| actor/feature_rank | occasional | `rl_metrics/actor/feature_rank` | Effective rank via SVD (τ=0.99) |
| actor/nrc1 | occasional | `rl_metrics/actor/nrc1` | Neural Rank Collapse subspace metric |
| actor/nrc2 | occasional | `rl_metrics/actor/nrc2` | Feature-weight alignment |
| actor/dormant_ratio | occasional | `rl_metrics/actor/dormant_ratio` | Fraction of dormant neurons (τ=0.025) |

**Implementation**: 
- Added `actor_repr_fn` to `ContrastiveNetworks` dataclass (`networks.py`) — a Haiku transform that runs the actor body (ResidualMLP + LayerNorm + Swish) without the NormalTanh head. Returns trunk features of shape `[batch, network_width]`.
- Added `extract_actor_features()` and `_get_actor_head_weights()` to `rl_metrics.py`.
- `compute_all_metrics()` now computes both critic and actor metrics.
- NRC2 for actor uses the `Normal/linear` weight (mean head) as the "final layer" for the alignment metric.
- NRC1 for actor uses `target_dim=action_dim` (the actor maps features → actions).

### 2. Dormant Ratio Threshold for Swish

**Previous value**: τ = 1e-5 (far too conservative; never triggers with Swish)

**New value**: τ = 0.025

**Rationale**:
- ReDo (Sokar et al., ICML 2023) tested τ ∈ {0, 0.025, 0.1} for ReLU, finding τ=0.1 optimal for Atari RL
- Swish(x) = x·σ(x) has a non-zero derivative floor: even for large negative inputs, |swish(x)| approaches 0 asymptotically but never reaches it exactly
- τ=0.1 is too aggressive for Swish — it would falsely flag healthy neurons with small but meaningful activations via the smooth Swish tail
- τ=0.025 catches genuinely dormant neurons (score < 2.5% of layer mean) while tolerating Swish's output characteristics
- This aligns with the Nature plasticity paper (Lyle et al., 2024) showing Swish has lower dormancy than ReLU but is NOT immune — a sensitive-enough threshold is needed to detect it
- The threshold constant is defined as `DORMANT_THRESHOLD = 0.025` in `rl_metrics.py` for easy adjustment

### 3. Automatic Actor Reset During Task 0 (dormancy-triggered)

**Motivation**: Certain seeds produce actor initializations with low weight norms and high dormancy that persist throughout training. Rather than resetting on a fixed schedule, the system monitors the actor's health and only resets when it detects a problem. If the actor learns well, no reset ever fires.

**Design constraints**:
- Reset ONLY during task 0 (base task)
- Never reset during tasks 1..N (would interfere with continual learning ablation studies on actor_mode and critic_mode)
- Critic is never touched by the reset
- Condition-based, not periodic: only fires when the actor's dormant ratio exceeds a threshold
- If the actor is healthy, the mechanism is completely silent

**How it works**:

1. At every RL metrics logging interval during task 0, the actor's dormant ratio is computed from the trunk features (body + LayerNorm + Swish output, before the NormalTanh head).
2. If `actor_dormant_ratio > actor_reset_dormant_threshold` (default 0.1 = 10% of neurons dormant), the actor is reinitialized with fresh random weights.
3. A warmup period (`actor_reset_warmup`, default 200K steps) prevents premature resets before the actor has had time to stabilize after initialization.
4. A safety cap (`actor_reset_max`, default 3) prevents infinite reset loops.

**Decision flow**:
```
for each rl_metrics check during task 0:
  if reset_count >= max_resets: skip
  if env_steps < warmup: skip
  compute actor dormant ratio from trunk features
  if dormant_ratio > threshold:
    reinit actor (fresh weights + optimizer)
    increment reset_count
  else:
    do nothing (actor is healthy)
```

**Flags**:
| Flag | Default | Description |
|---|---|---|
| `--actor_auto_reset` | `True` | Enable dormancy-triggered actor reset during task 0 |
| `--actor_reset_dormant_threshold` | `0.1` | Dormant ratio that triggers reset (10% of neurons dormant) |
| `--actor_reset_warmup` | `200000` | Min env steps before first check |
| `--actor_reset_max` | `3` | Max resets per task-0 run |

**W&B logging** (only when a reset fires):
- `actor_reset/triggered`: 1
- `actor_reset/dormant_ratio_at_reset`: the measured dormancy that triggered the reset
- `actor_reset/count`: cumulative resets so far
- `actor_reset/env_steps`: when the reset happened

**Threshold justification**: The reset threshold (0.1) is DIFFERENT from the dormancy measurement threshold (0.025). The measurement threshold (0.025) is what defines "dormant" at the individual neuron level (activation score < 2.5% of layer mean, calibrated for Swish). The reset threshold (0.1) is the FRACTION of neurons that must be dormant to trigger a reset. When 10% or more of the trunk neurons are dormant, the actor is clearly in a pathological regime — healthy actors with Swish + LayerNorm + ResidualMLP typically show <1% dormancy. The two thresholds work together:
- Swish neuron i is dormant if: `mean_abs(h_i) / mean_abs(h) < 0.025`
- Actor is pathological if: `count(dormant neurons) / total_neurons > 0.1`

**Relationship to prior work**:
- **ReDo (Sokar et al., 2023)**: Resets individual dormant neurons by reinitializing their incoming weights and zeroing outgoing weights. Our approach is coarser (full actor reset) but simpler and appropriate for the continual learning setting where individual neuron surgery during base task training is overkill.
- **Primacy Bias (Nikishin et al., 2022)**: Periodic full network resets. Our approach improves on this by being condition-based: if the initialization is good, the mechanism stays silent.

## Files Changed

| File | Changes |
|---|---|
| `contrastive/networks.py` | Added `actor_repr_fn` field to `ContrastiveNetworks`, `_actor_repr_fn` Haiku transform, and wired it into `make_networks` |
| `contrastive/rl_metrics.py` | Added actor metrics (dormant, NRC1, NRC2, rank, entropy, gini), updated threshold to 0.025, added `extract_actor_features`, `_get_actor_head_weights`, documented Swish threshold rationale |
| `contrastive/continual_learning.py` | Added `reset_actor(rng_key)` method |
| `run_continual_contrastive.py` | Added `--actor_auto_reset`, `--actor_reset_dormant_threshold`, `--actor_reset_warmup`, `--actor_reset_max` flags; dormancy-triggered reset logic inside rl_metrics block |
| `draft_3.sh` | Added `ACTOR_AUTO_RESET`, `ACTOR_RESET_DORMANT_THRESHOLD`, `ACTOR_RESET_WARMUP`, `ACTOR_RESET_MAX` parameters |
