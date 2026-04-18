# Code Structure Guide

## Repository Layout

```
.
├── run_continual_contrastive.py    # Main experiment driver
├── contrastive/                    # Core algorithm
│   ├── __init__.py                 # Exports make_networks
│   ├── networks.py                 # Network definitions (ResidualMLP, critic, actor)
│   ├── continual_learning.py       # ContinualContrastiveLearner (training step)
│   ├── knowledge_pool.py           # KnowledgePool (vector storage + merging)
│   ├── config.py                   # ContrastiveConfig dataclass
│   ├── continual_config.py         # ContinualConfig + task sequences
│   ├── agents.py                   # Actor wrappers
│   ├── builder.py                  # Distributed builder (LaunchPad)
│   ├── continual_builder.py        # Continual distributed builder
│   ├── distributed_layout.py       # LaunchPad distributed layout
│   ├── learning.py                 # Original (non-continual) learner
│   ├── rl_metrics.py               # Representation metrics (JAX): norms, rank, NRC, dormancy, ID
│   └── utils.py                    # Environment creation, observers, InitiallyRandomActor
├── env_utils.py                    # MetaWorld Sawyer wrappers + TaskIDGymWrapper
├── distributional.py               # NormalTanhDistribution (policy head)
├── default.py                      # Logger utilities
├── lp_contrastive.py               # Single-task LaunchPad runner (reference)
├── lp_continual_contrastive.py     # Continual LaunchPad runner (reference)
├── draft_3.sh                      # SLURM launcher
├── requirements.txt                # Python dependencies
└── set_up/                         # HPC environment setup scripts
```

---

## Component Guide

### 1. Network Architecture

**File:** `contrastive/networks.py`

The function `make_networks()` creates all three network components:

#### Critic (sa_encoder + g_encoder)

- Two independent encoders: `sa_encoder(state, action) → φ ∈ ℝ⁶⁴` and `g_encoder(goal) → ψ ∈ ℝ⁶⁴`
- With `use_residual=True` (default): each is a `ResidualMLP(output_dim=64, width=256, depth=4)`:
  ```
  Input → Dense(256) → LayerNorm → Swish      (input projection)
        → 4× (Dense(256) → LayerNorm → Swish) + skip   (1 residual block)
        → Dense(64)                             (output projection)
  ```
- Energy function: inner product `φ(s,a)ᵀψ(g)` via `einsum('ik,jk->ij', sa_repr, g_repr)` (default), or negative L2 distance (configurable via `energy_fn='l2'`)
- Location: `_repr_fn()` (lines 180–226), `_combine_repr()` (lines 229–237), `_critic_fn()` (lines 239–249)

**To modify the critic architecture:** Edit the `if use_residual:` branch in `_repr_fn()` (lines 196–202) for the encoder structure, or `_combine_repr()` (lines 229–237) for the energy function.

#### Actor

- Body: `ResidualMLP(output_dim=256, width=256, depth=4)` (same architecture as critic encoders but output_dim=256)
- Head: `NormalTanhDistribution` — maps trunk output to mean + log_std, samples via tanh-squashed Gaussian
- Location: `_actor_fn()` (lines 251–273)

**To modify the actor architecture:** Edit the `if use_residual:` branch in `_actor_fn()` (lines 256–263).

#### ResidualMLP Module

- Class definition: lines 88–138
- Block structure: 4× (Dense → LayerNorm → Swish) per residual block + skip connection
- Init: LeCun uniform (`VarianceScaling(1/3, 'fan_in', 'uniform')`)
- Configurable: `width` (hidden dim), `depth` (Dense layers in blocks, must be multiple of 4)

### 2. Continual Learning Logic

**File:** `contrastive/continual_learning.py` — `ContinualContrastiveLearner`

This is the per-task learner. It handles:

#### Initialization (lines 459–550)

- **Actor init:** When `theta_base is None` (reset/base task), creates fresh random policy. When `theta_base` is provided (CKA/persistent), uses it as the base.
- **Critic init:** Depends on `critic_mode`:
  - `persistent`: carry forward `prev_q_params`
  - `reset`: fresh `networks.q_network.init(key_q)`
  - `cka`: compose `q_base + critic_pool_c`
- **Optimizer/target init:** Controlled by `critic_was_freshly_init` boolean

#### Training Step (the `_update_step` function)

- CKA composition: `θ' = θ_base + pool_c + v_k` (line ~317)
- Critic loss: InfoNCE with CPC regularizer (line ~236)
- Actor loss: `-diag(φ(s, π(s,g))ᵀ ψ(g))` (line ~291)
- Gradient masking: when `adapt_heads_only=True` and `encoder_from_base=True`, body gradients are zeroed

**To modify the training loop:** Edit `_update_step()` for per-step logic, or `_scan_update()` for the `lax.scan` inner loop.

#### Knowledge Pool Contribution

- `_compute_pool_contribution()`: computes `Σ α_j v_j` from the pool
- `_compute_critic_pool_contribution()`: same for critic CKA mode
- Location: look for these method names in the class

### 3. Knowledge Pool

**File:** `contrastive/knowledge_pool.py` — `KnowledgePool`

- `append(v_k)`: add a knowledge vector
- `merge_if_needed()`: if pool size exceeds `k_max`, merge the two most cosine-similar vectors
- `get_vectors()`: returns list of all stored vectors
- `state_dict()` / `load_state_dict()`: serialization for checkpointing

**To modify pool merging strategy:** Edit `merge_if_needed()`.

### 4. Training Orchestration

**File:** `run_continual_contrastive.py`

#### Main Loop (`main()`, line ~675)

Controls the task sequence:
1. Auto-resume: scan for existing checkpoints → determine `start_task`
2. For each `task_id` in `range(start_task, num_tasks)`:
   - Prepare actor/critic state based on `actor_mode` and `critic_mode`
   - Call `train_single_task()`
   - Post-task extraction: split v_k into head/body, fold body into θ_base
   - Cross-task evaluation
   - Save checkpoint
   - Update state for next task

#### Per-Task Training (`train_single_task()`, line ~260)

Creates per-task resources:
- Environment (MetaWorld + wrappers)
- Reverb replay buffer
- ContinualContrastiveLearner
- Training actor (stochastic) + evaluator actor (deterministic)
- Runs the training loop: collect → SGD → periodic eval

**To modify the task loop logic:** Edit `main()` for between-task handling, or `train_single_task()` for within-task training.

### 5. Checkpoint System

**File:** `run_continual_contrastive.py` (lines 139–180)

#### Path Format

```
{ckpt_dir}/actor_{mode}_critic_{mode}_tid_{bool}_heads_{bool}/seed_{seed}/task_{id}.pkl
```

Each unique configuration produces a unique directory. No cross-contamination.

#### Checkpoint Contents

| Key | Content |
|---|---|
| `theta_base` | Base policy params (None for reset actor) |
| `pool_vectors` | Knowledge pool state dict |
| `q_params` | Critic parameters (φ, ψ) |
| `target_q_params` | Target critic parameters |
| `q_optimizer_state` | Adam state for critic |
| `q_base` | Frozen critic base (CKA critic only) |
| `critic_pool_vectors` | Critic knowledge pool (CKA critic only) |

#### Auto-Resume

When `--start_task=0` (default), the runner scans for existing checkpoints matching the exact same config and seed. If found, resumes from the next task.

**To modify checkpoint contents:** Edit `save_ckpt()` and `load_ckpt()`.

### 6. Environment Wrappers

**File:** `env_utils.py`

- `SawyerXEnv` classes: thin wrappers around MetaWorld V2 Sawyer environments, adding unified observation space (state ∈ ℝ¹¹, goal ∈ ℝ¹¹)
- `TaskIDGymWrapper`: appends one-hot task ID to observations when `use_task_id=True`
- `contrastive/utils.py → make_environment()`: creates the wrapped environment with observers (SuccessObserver, DistanceObserver)

**To add a new Sawyer task:** See `doc/Metaworld_documentation/ADDING_METAWORLD_TASKS_GUIDE.md`.

### 7. Configuration

**File:** `contrastive/config.py` — `ContrastiveConfig` dataclass

Key fields:

| Field | Default | Description |
|---|---|---|
| `batch_size` | 256 | SGD batch size |
| `actor_learning_rate` | 3e-4 | Actor optimizer lr |
| `critic_learning_rate` | 3e-4 | Critic optimizer lr |
| `discount` | 0.99 | Discount factor γ |
| `tau` | 0.005 | Target network smoothing |
| `hidden_layer_sizes` | (256, 256) | Plain MLP hidden sizes (ignored when use_residual=True) |
| `num_sgd_steps_per_step` | 64 | UTD ratio (SGD steps per env step) |
| `repr_dim` | 64 | Representation dimension for φ and ψ |
| `use_residual` | True | Use ResidualMLP architecture |
| `network_width` | 256 | ResidualMLP hidden dimension |
| `critic_depth` | 4 | Dense layers in critic residual blocks |
| `actor_depth` | 4 | Dense layers in actor residual blocks |
| `energy_fn` | 'inner_product' | Critic energy: 'inner_product' or 'l2' |
| `logsumexp_penalty` | 0.01 | CPC regularizer coefficient |
| `use_cpc` | False | Set to True by `--alg=contrastive_cpc` |

**File:** `contrastive/continual_config.py` — `ContinualConfig` dataclass

| Field | Default | Description |
|---|---|---|
| `num_tasks` | 10 | Number of tasks in sequence |
| `steps_per_task` | 8,000,000 | Env steps per continual task |
| `base_steps` | 8,000,000 | Env steps for base task (task 0) |
| `k_max` | 10 | Max knowledge pool size |

### 8. Task Sequences

**File:** `contrastive/continual_config.py`

10-task sequence:
```
hammer, push_wall, faucet_close, push_back, stick_pull,
handle_press_side, push, shelf_place, window_close, peg_unplug_side
```

20-task sequence: the 10-task sequence repeated twice.

**To modify the task sequence:** Edit `CONTINUAL_TASK_SEQUENCE` in `continual_config.py`, and ensure corresponding wrappers exist in `env_utils.py`.

### 9. SLURM Launcher

**File:** `draft_3.sh`

All configuration is via environment variables with sensible defaults. The script builds a `FLAGS` string and runs `python run_continual_contrastive.py $FLAGS`.

Key variables: `SEED`, `ACTOR_MODE`, `CRITIC_MODE`, `USE_TASK_ID`, `ADAPT_HEADS_ONLY`, `STEPS_PER_TASK`, `SINGLE_TASK`, `USE_RESIDUAL`, etc.

**To add a new flag:** Define it as an env var with default, echo it in the info block, and append to the FLAGS string.
