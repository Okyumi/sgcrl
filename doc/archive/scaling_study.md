# Scaling Ablation Study

Based on Wang et al. (2025), "1000 Layer Networks for Self-Supervised RL" ([arXiv:2503.14858](https://arxiv.org/abs/2503.14858)).

---

## 1. Architectural Comparison

### Their architecture (scaling-crl codebase)

Each encoder (sa_encoder, g_encoder, actor):

1. **Input projection:** Dense(width) → LayerNorm → Swish
2. **N residual blocks**, each containing:
   ```
   identity = x
   x = Dense(width) → LayerNorm → Swish
   x = Dense(width) → LayerNorm → Swish
   x = Dense(width) → LayerNorm → Swish
   x = Dense(width) → LayerNorm → Swish
   x = x + identity   (skip connection)
   ```
3. **Output projection:** Dense(repr_dim)

- Total Dense layers in blocks = `depth`. Block size = 4, so `depth / 4` blocks.
- Activation: **Swish** (critical for scaling — ReLU degrades significantly)
- Normalization: **LayerNorm** (critical — removal causes major degradation)
- Initialization: **LeCun uniform** (`variance_scaling(1/3, "fan_in", "uniform")`)
- Energy function: **Negative L2 distance** `−‖φ(s,a) − ψ(g)‖₂`
- Logsumexp penalty coefficient: **0.1**
- Repr dim: 64

### Our architecture (before this change)

- **Plain MLP** via `hk.nets.MLP`: Dense → ReLU → Dense → ReLU → Dense(repr_dim)
- **No LayerNorm, no skip connections, no Swish**
- Activation: **ReLU**
- Initialization: **Glorot uniform** (`VarianceScaling(1.0, "fan_avg", "uniform")`)
- Energy function: **Inner product** `φ(s,a)ᵀψ(g)`
- Logsumexp penalty: **0.01**
- Hidden sizes: (256, 256) → effectively depth=2

### What we changed

Added `--use_residual` flag that enables the ResidualMLP architecture in `contrastive/networks.py`:

| Component | Default (SGCRL) | `--use_residual` (scaling) |
|---|---|---|
| Architecture | Plain MLP (hk.nets.MLP) | ResidualMLP (LayerNorm + Swish + skip) |
| Activation | ReLU | Swish |
| Normalization | None | LayerNorm |
| Skip connections | None | Every 4 layers |
| Init | Glorot uniform | LeCun uniform |
| Configurable depth | No (fixed by hidden_layer_sizes) | Yes (critic_depth, actor_depth) |

The energy function and logsumexp penalty are separately configurable via `--energy_fn` and `--logsumexp_penalty`.

### What was already aligned

| Parameter | 1000-layer paper | Ours |
|---|---|---|
| Learning rate (actor + critic) | 3e-4 | 3e-4 |
| Discount | 0.99 | 0.99 |
| Repr dim | 64 | 64 |
| Width | 256 | 256 |
| Contrastive loss | InfoNCE (CPC) | InfoNCE (CPC) |
| Batch size | 256 | 256 |

### What differs by design (not aligned)

| Parameter | 1000-layer paper | Ours | Reason |
|---|---|---|---|
| Environment | Brax/MJX (GPU-vectorized) | MetaWorld (CPU, single env) | Different benchmark |
| Num envs | 512 | 1 | CPU-based Meta-World |
| Steps | 100M–400M | 8M | Meta-World tasks are simpler |
| UTD ratio | 1:40 | 64:1 | Different data collection rate |
| Energy fn | L2 distance | Inner product (SGCRL) | SGCRL paper uses inner product |

---

## 2. Task Selection: `shelf_place`

**Why `shelf_place`?**

1. **Multi-step coordination required:** The Sawyer arm must grasp an object, lift it, and place it on a shelf — a 3-phase skill that demands precise sequential control.
2. **Likely weak baseline performance:** Among the 10 tasks in our sequence, multi-step manipulation tasks like shelf_place and stick_pull consistently show lower success rates in goal-conditioned settings because the contrastive critic must capture longer-horizon reachability.
3. **Scaling hypothesis is most testable here:** The 1000-layer paper shows that depth benefits are most pronounced on tasks with high observation dimensionality and complex state-goal topology. While our obs_dim (11) is modest, the multi-step nature of shelf_place creates a complex reachability landscape — exactly where deeper representations should help.
4. **It's task 7 in the 10-task sequence**, which means it also sits in the continual learning pipeline where forward transfer matters — a good candidate for later continual scaling experiments.

---

## 3. Experiment Commands

All experiments are single-task (no continual learning) using `NUM_TASKS=1` to isolate the scaling effect.

All experiments use `SINGLE_TASK=sawyer_shelf_place` to override the task sequence and train on one environment only.

### A. Baselines (plain MLP)

```bash
# A1: SGCRL default — plain MLP (256,256), ReLU, inner product, lse=0.01
SINGLE_TASK=sawyer_shelf_place SEED=11 sbatch draft_3.sh

# A2: Match 1000-layer paper's loss — plain MLP, L2 energy, lse=0.1
SINGLE_TASK=sawyer_shelf_place SEED=11 \
  ENERGY_FN=l2 LOGSUMEXP_PENALTY=0.1 \
  sbatch draft_3.sh
```

### B. Residual MLP — varying depth (inner product energy)

```bash
# B1: Depth 4 (1 residual block)
SINGLE_TASK=sawyer_shelf_place SEED=11 \
  USE_RESIDUAL=true CRITIC_DEPTH=4 ACTOR_DEPTH=4 \
  sbatch draft_3.sh

# B2: Depth 8 (2 residual blocks)
SINGLE_TASK=sawyer_shelf_place SEED=11 \
  USE_RESIDUAL=true CRITIC_DEPTH=8 ACTOR_DEPTH=8 \
  sbatch draft_3.sh

# B3: Depth 16 (4 residual blocks)
SINGLE_TASK=sawyer_shelf_place SEED=11 \
  USE_RESIDUAL=true CRITIC_DEPTH=16 ACTOR_DEPTH=16 \
  sbatch draft_3.sh

# B4: Depth 32 (8 residual blocks)
SINGLE_TASK=sawyer_shelf_place SEED=11 \
  USE_RESIDUAL=true CRITIC_DEPTH=32 ACTOR_DEPTH=32 \
  sbatch draft_3.sh
```

### C. Full 1000-layer recipe (ResidualMLP + L2 energy + lse=0.1)

```bash
# C1: Depth 4 + L2 + lse=0.1
SINGLE_TASK=sawyer_shelf_place SEED=11 \
  USE_RESIDUAL=true CRITIC_DEPTH=4 ACTOR_DEPTH=4 \
  ENERGY_FN=l2 LOGSUMEXP_PENALTY=0.1 \
  sbatch draft_3.sh

# C2: Depth 8 + L2 + lse=0.1
SINGLE_TASK=sawyer_shelf_place SEED=11 \
  USE_RESIDUAL=true CRITIC_DEPTH=8 ACTOR_DEPTH=8 \
  ENERGY_FN=l2 LOGSUMEXP_PENALTY=0.1 \
  sbatch draft_3.sh

# C3: Depth 16 + L2 + lse=0.1
SINGLE_TASK=sawyer_shelf_place SEED=11 \
  USE_RESIDUAL=true CRITIC_DEPTH=16 ACTOR_DEPTH=16 \
  ENERGY_FN=l2 LOGSUMEXP_PENALTY=0.1 \
  sbatch draft_3.sh
```

### D. Width ablation (fixed depth=8, inner product)

```bash
# D1: Width 128
SINGLE_TASK=sawyer_shelf_place SEED=11 \
  USE_RESIDUAL=true CRITIC_DEPTH=8 ACTOR_DEPTH=8 \
  NETWORK_WIDTH=128 \
  sbatch draft_3.sh

# D2: Width 512
SINGLE_TASK=sawyer_shelf_place SEED=11 \
  USE_RESIDUAL=true CRITIC_DEPTH=8 ACTOR_DEPTH=8 \
  NETWORK_WIDTH=512 \
  sbatch draft_3.sh
```

### Summary table

| ID | Architecture | Depth | Width | Energy | LSE | Purpose |
|---|---|---|---|---|---|---|
| A1 | MLP | 2 | 256 | inner prod | 0.01 | SGCRL baseline |
| A2 | MLP | 2 | 256 | L2 | 0.1 | Loss-only ablation |
| B1 | ResidualMLP | 4 | 256 | inner prod | 0.01 | Arch-only ablation (shallowest) |
| B2 | ResidualMLP | 8 | 256 | inner prod | 0.01 | Arch-only, moderate depth |
| B3 | ResidualMLP | 16 | 256 | inner prod | 0.01 | Arch-only, deeper |
| B4 | ResidualMLP | 32 | 256 | inner prod | 0.01 | Arch-only, deep |
| C1 | ResidualMLP | 4 | 256 | L2 | 0.1 | Full recipe, shallowest |
| C2 | ResidualMLP | 8 | 256 | L2 | 0.1 | Full recipe, moderate |
| C3 | ResidualMLP | 16 | 256 | L2 | 0.1 | Full recipe, deeper |
| D1 | ResidualMLP | 8 | 128 | inner prod | 0.01 | Width ablation (narrow) |
| D2 | ResidualMLP | 8 | 512 | inner prod | 0.01 | Width ablation (wide) |

---

## 4. What to Look For

1. **Does the ResidualMLP (B series) outperform plain MLP (A1) at the same width?** If yes, the architectural components (LayerNorm + Swish + skip) help even at shallow depth.
2. **Does increasing depth (B1 → B2 → B3 → B4) improve performance?** This tests the core claim: depth scaling works for contrastive RL.
3. **Does L2 energy (C series) outperform inner product (B series)?** This isolates the energy function choice.
4. **Does width matter at fixed depth (D series)?** Width vs depth comparison.
5. **Does training become unstable at high depth?** Monitor actor loss and critic loss for signs of instability.

---

## 5. Implementation Notes

### Files changed

- `contrastive/networks.py`: Added `ResidualMLP` Haiku module and new kwargs to `make_networks` (`use_residual`, `network_width`, `critic_depth`, `actor_depth`, `energy_fn`).
- `contrastive/config.py`: Added scaling config fields (`use_residual`, `network_width`, `critic_depth`, `actor_depth`, `energy_fn`, `logsumexp_penalty`).
- `contrastive/continual_learning.py`: Changed hardcoded `0.01` logsumexp penalty to `config.logsumexp_penalty`.
- `run_continual_contrastive.py`: Added flags and threaded them through to config and make_networks.
- `draft_3.sh`: Added env vars for all scaling flags.

### Backward compatibility

When `--use_residual=False` (default), the code behaves exactly as before:
- Plain `hk.nets.MLP` with ReLU
- Inner product energy function
- Logsumexp penalty = 0.01
- Hidden sizes = (256, 256)

No existing experiment is affected.

### Forward: continual learning with scaled networks

The scaling flags are in `ContrastiveConfig` and threaded through `make_networks`, which is shared between single-task and continual training. To use a scaled architecture in the full continual setting:

```bash
# Example: continual learning with depth-8 residual networks
CRITIC_MODE=persistent SEED=11 \
  USE_RESIDUAL=true CRITIC_DEPTH=8 ACTOR_DEPTH=8 \
  sbatch draft_3.sh
```

The CKA actor decomposition (θ_base + pool + v_k) works with any network architecture — it operates on the Haiku pytree, not on specific layer types. The knowledge pool, gradient masking, and post-task extraction are all architecture-agnostic.

---

## References

[1] Wang et al., "1000 Layer Networks for Self-Supervised RL," arXiv:2503.14858, 2025. Code: https://github.com/wang-kevin3290/scaling-crl
[2] Liu et al., "A Single Goal Is All You Need," ICLR 2025.
[3] Eysenbach et al., "Contrastive Learning as Goal-Conditioned RL," NeurIPS 2022.
