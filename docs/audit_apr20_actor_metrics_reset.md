# Actor Metrics + Periodic Reset — April 20, 2026

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

### 3. Periodic Actor Reset During Task 0

**Motivation**: Certain seeds produce actor initializations with low weight norms and high dormancy that persist throughout training. Since the actor is initialized independently of the critic, and the problem manifests only in the actor, the solution is to give the actor multiple chances at a good initialization.

**Design constraints**:
- Reset ONLY during task 0 (base task)
- Never reset during tasks 1..N (would interfere with continual learning ablation studies on actor_mode and critic_mode)
- Critic is never touched by the reset
- Must use fresh random weights (not the same initialization that caused the trap)

**Implementation**:
- `--actor_reset_interval` flag (default 0 = disabled). When > 0, every N env steps during task 0, the actor is reinitialized.
- `ContinualContrastiveLearner.reset_actor(rng_key)` method:
  1. Generates fresh policy params via `policy_network.init(rng_key)` 
  2. Sets `policy_base_params = fresh params`
  3. Resets `v_k` to zeros
  4. Resets `v_k_optimizer_state`
  5. Asserts `task_id == 0` (safety check)
- Uses a separate RNG stream (`PRNGKey(seed + 9999)`) so each reset produces different random weights
- Logs `actor_reset/count` and `actor_reset/env_steps` to W&B

**Usage examples**:
```bash
# Reset actor every 500K steps during task 0
ACTOR_RESET_INTERVAL=500000 SEED=42 sbatch draft_3.sh

# Reset actor every 1M steps during task 0
ACTOR_RESET_INTERVAL=1000000 SEED=42 sbatch draft_3.sh

# Disabled (default)
SEED=42 sbatch draft_3.sh
```

**Recommended interval**: For 8M-step base tasks, `actor_reset_interval=2000000` (reset at 2M, 4M, 6M). This gives the actor 4 initialization chances while still allowing 2M steps of training after the last reset. Adjust based on how early the bad-seed pathology becomes apparent.

**Relationship to the Primacy Bias paper (Nikishin et al., 2022)**: The periodic actor reset follows the same principle — periodically reinitializing network weights to combat loss of plasticity. The key difference is that our reset is restricted to the actor during task 0, preserving the integrity of the critic (whose representations are shared across tasks) and the continual learning ablation design.

## Files Changed

| File | Changes |
|---|---|
| `contrastive/networks.py` | Added `actor_repr_fn` field to `ContrastiveNetworks`, `_actor_repr_fn` Haiku transform, and wired it into `make_networks` |
| `contrastive/rl_metrics.py` | Added actor metrics (dormant, NRC1, NRC2, rank, entropy, gini), updated threshold to 0.025, added `extract_actor_features`, `_get_actor_head_weights`, documented Swish threshold rationale |
| `contrastive/continual_learning.py` | Added `reset_actor(rng_key)` method |
| `run_continual_contrastive.py` | Added `--actor_reset_interval` flag, periodic reset logic in training loop, W&B config logging |
| `draft_3.sh` | Added `ACTOR_RESET_INTERVAL` parameter |
