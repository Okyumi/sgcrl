# Default Network Architecture: ResidualMLP

**Date:** April 18, 2026

## Change Summary

The default network architecture was changed from **plain MLP** to **ResidualMLP** based on single-task scaling ablation results. This applies to all three network components: the sa_encoder (critic), g_encoder (critic), and actor body.

## What Changed

| Setting | Before | After |
|---|---|---|
| `use_residual` default | `False` | `True` |
| Activation | ReLU | Swish |
| Normalization | None | LayerNorm |
| Skip connections | None | Yes (every 4 layers) |
| Initialization | Glorot uniform | LeCun uniform |

All other settings are unchanged:

| Setting | Value (unchanged) |
|---|---|
| `critic_depth` | 4 (1 residual block) |
| `actor_depth` | 4 (1 residual block) |
| `network_width` | 256 |
| `energy_fn` | `inner_product` |
| `logsumexp_penalty` | 0.01 |
| `repr_dim` | 64 |

## Where the Change Was Made

1. `contrastive/config.py` line 72: `use_residual: bool = True`
2. `contrastive/networks.py` line 152: `use_residual = True` (function signature)
3. `run_continual_contrastive.py` line 109: `flags.DEFINE_bool('use_residual', True, ...)`
4. `draft_3.sh` line 50: `USE_RESIDUAL="${USE_RESIDUAL:-true}"`

## How the Architecture Maps to Code

With `use_residual=True` and `depth=4`:

### Critic (sa_encoder and g_encoder)

Each encoder is `ResidualMLP(output_dim=64, width=256, depth=4)` in `contrastive/networks.py` lines 196-202.

Architecture per encoder:
```
Input → Dense(256) → LayerNorm → Swish          (input projection)
      → Dense(256) → LayerNorm → Swish  ─┐
      → Dense(256) → LayerNorm → Swish   │      (1 residual block,
      → Dense(256) → LayerNorm → Swish   │       4 Dense layers)
      → Dense(256) → LayerNorm → Swish  ─┤
      + skip ─────────────────────────────┘
      → Dense(64)                                (output projection)
```

Total: 6 Dense layers per encoder (1 input + 4 block + 1 output).
The two encoders (sa_encoder, g_encoder) are independent — not weight-shared.

### Actor

The actor body is `ResidualMLP(output_dim=256, width=256, depth=4)` in `contrastive/networks.py` lines 258-260, followed by `NormalTanhDistribution` (the policy head).

Architecture:
```
Input → [same 6-layer ResidualMLP as above, but output_dim=256]
      → NormalTanhDistribution (Dense → mean/logstd)
```

### Energy function

Unchanged: inner product `φ(s,a)ᵀψ(g)` via `jax.numpy.einsum('ik,jk->ij', sa_repr, g_repr)` at line 237.

## Motivation

Single-task scaling ablation on `shelf_place` (experiment B1 in `2026-04-06_scaling_study.md`) showed that `ResidualMLP` with depth=4 and inner-product energy outperformed the plain MLP baseline. The key architectural ingredients are LayerNorm, Swish activation, and skip connections — even at the shallowest depth (1 block), these provide a meaningful improvement.

## How to Revert

To run with the old plain MLP architecture:
```bash
USE_RESIDUAL=false sbatch draft_3.sh
```
Or pass `--nouse_residual` to the Python script directly.
