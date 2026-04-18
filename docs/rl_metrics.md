# RL Representation Metrics

## Overview

The `--log_rl_metrics` flag (enabled by default) adds representation-quality metrics to the training pipeline. These metrics analyze the actor and critic encoder representations, helping understand what happens to the learned features across tasks in the continual learning setting.

## Metric Descriptions

### Frequent Metrics (every `eval_every` steps)

These are computationally cheap and logged at the same frequency as the evaluator.

| Metric | Key | Description |
|---|---|---|
| Weight norm | `actor/weight_norm`, `critic/weight_norm` | L2 norm of all parameters. Tracks weight growth/collapse. |
| Final layer norm | `actor/final_layer_norm` | L2 norm of the actor's output layer weights. |
| Feature entropy | `critic_sa/entropy`, `critic_g/entropy` | Shannon entropy of \|feature\| distribution. Higher = more uniformly spread activations. |
| Gini sparsity | `critic_sa/gini`, `critic_g/gini` | Gini coefficient ∈ [0,1]. Higher = sparser features (fewer neurons carry information). |

### Occasional Metrics (every `5 × eval_every` steps)

Moderate computational cost. Require SVD or full-batch feature computation.

| Metric | Key | Description |
|---|---|---|
| Feature rank | `critic_sa/feature_rank`, `critic_g/feature_rank` | Effective rank: minimum k s.t. top-k singular values explain ≥ 99% of variance. Low rank = collapsed representations. |
| NRC1 | `critic_sa/nrc1`, `critic_g/nrc1` | Neural Rank Collapse metric 1. Measures how much features lie in a low-dimensional subspace. Lower = more collapsed. |
| NRC2 | `critic_sa/nrc2`, `critic_g/nrc2` | Neural Rank Collapse metric 2. Measures alignment between features and the final layer's row space. Lower = features are aligned with what the output layer can use. |
| Dormant ratio | `critic_sa/dormant_ratio`, `critic_g/dormant_ratio` | Fraction of neurons with negligible activation (< 0.0025% of average). Higher = more dead neurons. |

### Rare Metrics (every `20 × eval_every` steps)

Expensive. TWO-NN intrinsic dimension requires O(N²) pairwise distances.

| Metric | Key | Description |
|---|---|---|
| Intrinsic dimension | `critic_sa/intrinsic_dim`, `critic_g/intrinsic_dim` | TWO-NN estimate of the manifold dimensionality of the feature space. Facco et al. (2017). |

## Logging Schedule

With default `eval_every=50000`:

| Level | Frequency | Metrics |
|---|---|---|
| Frequent | Every 50K steps | weight_norm, final_layer_norm, entropy, gini |
| Occasional | Every 250K steps | + feature_rank, NRC1, NRC2, dormant_ratio |
| Rare | Every 1M steps | + intrinsic_dimension |

## Implementation

**File:** `contrastive/rl_metrics.py`

All metrics are implemented in pure JAX (no PyTorch). The intrinsic dimension computation uses NumPy internally for the O(N²) distance matrix.

**Feature extraction:** Uses `networks.repr_fn(params, obs, action)` to get (sa_repr, g_repr) — the encoder outputs before the energy function. These are the 64-dimensional representations from the sa_encoder and g_encoder.

**Batch sampling:** At each metric logging step, one batch is sampled from the Reverb replay buffer. The batch size matches the training batch size (default 256).

**Error handling:** The metrics block is wrapped in try/except. If any metric computation fails (e.g., insufficient samples in the replay buffer), a warning is printed and training continues.

## How to Disable

```bash
LOG_RL_METRICS=false sbatch draft_3.sh
```

Or: `python run_continual_contrastive.py --nolog_rl_metrics`

## Ported From

Original PyTorch implementation: `rl_metrics.py` (attached). The port preserves the exact mathematical definitions while adapting to JAX arrays and the Haiku parameter pytree structure.

Key adaptations:
- PyTorch hooks → direct feature extraction via `networks.repr_fn`
- `nn.Module` parameter iteration → `jax.tree_util.tree_leaves`
- PyTorch SVD → `jnp.linalg.svd` / `jnp.linalg.svdvals`
- TWO-NN batched distance computation → NumPy (avoids JIT compilation of O(N²) pairwise ops)
