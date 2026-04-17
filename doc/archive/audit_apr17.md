# Code Audit & Refactor — April 17, 2026

## Summary of Changes

1. Replaced `--reset_actor` (boolean) with `--actor_mode` (string: `cka` / `reset` / `persistent`)
2. Implemented `persistent` actor mode (Group C)
3. Updated checkpoint naming: `actor_{mode}_critic_{mode}_tid_{bool}_heads_{bool}`
4. Updated defaults: `steps_per_task=8M`, `base_steps=8M`, `k_max=10`
5. Fixed `critic_mode='cka'` handling when `theta_base is None` (reset actor + CKA critic)
6. Simplified optimizer/target init with `critic_was_freshly_init` boolean
7. Updated `draft_3.sh` to use `ACTOR_MODE` instead of `RESET_ACTOR`

---

## Configuration Verification

### Notation

- **Task 0**: base task (first in sequence)
- **Task k > 0**: subsequent continual tasks
- **θ_base**: frozen base policy from task 0
- **v_k**: knowledge vector for task k
- **pool**: knowledge pool containing {v_1, ..., v_{k-1}}
- **pool_c**: Σ α_j v_j (weighted sum of pool vectors)
- **θ'**: composed policy = θ_base + pool_c + v_k

### Group A: Reset Actor

#### A1: `actor_mode=reset, critic_mode=reset` (baseline1)

| Phase | Actor | Critic |
|---|---|---|
| Before task k>0 | `_theta_base = None`, `_pool = empty` (line 849-852) | `prev_q` passed through |
| Learner init | `theta_base is None` → fresh random policy (line 461-462) | `critic_mode='reset'` → `q = init()` (line 476, else branch at 477) |
| Training | v_k trained from scratch on random policy | Critic trained from scratch |
| After task | `theta_base = None`, `pool = empty` (line 884-887) | Critic returned to caller |
| Checkpoint | `theta_base=None`, `pool=empty`, `q_params`, `tgt_q` | All saved |
| Resume | theta_base=None, pool=empty, prev_q loaded but will be re-init'd | Correct |

**Verdict: ✅ Correct.** Each task is fully independent. No transfer.

#### A2: `actor_mode=reset, critic_mode=persistent` (baseline2)

| Phase | Actor | Critic |
|---|---|---|
| Before task k>0 | `_theta_base = None`, `_pool = empty` | `prev_q` from previous task |
| Learner init | Fresh random policy | `critic_mode='persistent'` AND `prev_q is not None` → `q = prev_q` (line 466-467) |
| Training | v_k trained from scratch | Critic fine-tuned from previous task |
| After task | `theta_base = None`, `pool = empty` | Critic saved |
| Checkpoint | theta_base=None, prev_q (persistent) | Correct |
| Resume | theta_base=None, prev_q loaded and carried forward | Correct |

**Verdict: ✅ Correct.** Actor reset, critic carries forward.

**Bug fixed in this commit:** Previously, when `theta_base is None`, the code unconditionally did `q = init()`, ignoring `critic_mode='persistent'`. Now it checks `critic_mode`.

#### A3: `actor_mode=reset, critic_mode=cka`

| Phase | Actor | Critic |
|---|---|---|
| Before task k>0 | `_theta_base = None`, `_pool = empty` | `q_base`, `critic_pool` passed |
| Learner init | Fresh random policy | `critic_mode='cka'` AND `q_base is not None` → `q = q_base + pool_c` (line 468-475) |
| Training | v_k trained from scratch | Critic trains; w_k extracted after |
| After task | theta_base=None, pool=empty | `critic_pool.append(w_k)` |
| Checkpoint | theta_base=None, q_base, critic_pool_vectors | Correct |
| Resume | theta_base=None, q_base and critic_pool loaded | Correct |

**Verdict: ✅ Correct.** Actor reset, critic uses CKA decomposition.

**Bug fixed in this commit:** Previously, `critic_mode='cka'` was not handled in the `theta_base is None` branch.

### Group B: CKA Actor

#### B1: `actor_mode=cka, critic_mode=persistent`

| Phase | Actor | Critic |
|---|---|---|
| Before task k>0 | `_theta_base = theta_base` (from prev task), `_pool = pool` | prev_q from previous task |
| Learner init | `theta_base is not None` → CKA composition: θ' = θ_base + pool_c + v_k | `critic_mode='persistent'` → `q = prev_q` (line 482-485) |
| Training | v_k trained; gradients flow to v_k only (body folded into θ_base after) | Critic fine-tuned |
| After task | θ_base updated (body folded), pool.append(v_k_head) | Critic saved |
| Checkpoint | theta_base (with body), pool_vectors, q_params | Correct |
| Resume | theta_base and pool loaded; prev_q carried forward | Correct |

**Verdict: ✅ Correct.** Standard CKA-RL actor with persistent critic.

#### B2: `actor_mode=cka, critic_mode=cka`

| Phase | Actor | Critic |
|---|---|---|
| Before task k>0 | CKA composition for actor | CKA composition for critic |
| Learner init | θ' = θ_base + pool_c + v_k | q' = q_base + critic_pool_c + w_k |
| Training | v_k trained | w_k trained (extracted post-task) |
| After task | pool.append(v_k_head) | critic_pool.append(w_k) |
| Checkpoint | theta_base, pool_vectors, q_base, critic_pool_vectors | Correct |
| Resume | All loaded correctly | Correct |

**Verdict: ✅ Correct.** Full CKA decomposition for both actor and critic.

#### B3: `actor_mode=cka, critic_mode=reset` (baseline4)

| Phase | Actor | Critic |
|---|---|---|
| Before task k>0 | CKA composition for actor | prev_q passed but will be re-init'd |
| Learner init | θ' = θ_base + pool_c + v_k | `critic_mode='reset'` → `q = init()` (line 486-488) |
| Training | v_k trained | Critic trained from scratch |
| After task | pool.append(v_k_head) | Critic returned |
| Checkpoint | theta_base, pool_vectors, q_params (fresh per task) | Correct |
| Resume | theta_base and pool loaded; critic will be re-init'd | Correct |

**Verdict: ✅ Correct.** CKA-RL actor with SAC-style critic reset (matches CKA-RL baseline behavior).

### Group C: Persistent Actor

#### C1: `actor_mode=persistent, critic_mode=persistent`

| Phase | Actor | Critic |
|---|---|---|
| Before task k>0 | `_theta_base = theta_base` (= composed_policy from prev task), `_pool = empty` | prev_q carried forward |
| Learner init | `theta_base is not None` → enters CKA branch, but pool is empty → pool_c = 0 → θ' = theta_base + 0 + v_k | `critic_mode='persistent'` → `q = prev_q` |
| Training | v_k captures the delta from the previous task's trained policy | Critic fine-tuned |
| After task | `theta_base = composed_policy` (= prev_theta_base + v_k), pool stays empty (line 888-891) | Critic saved |
| Checkpoint | theta_base (fully trained policy), empty pool, q_params | Correct |
| Resume | theta_base loaded (the continuously trained policy), pool empty | Correct |

**Verdict: ✅ Correct.** Both actor and critic are continuously trained. No decomposition.

#### C2: `actor_mode=persistent, critic_mode=reset` (baseline3)

Same as C1 except critic is reinitialized each task.

**Verdict: ✅ Correct.**

#### C3: `actor_mode=persistent, critic_mode=cka`

Same as C1 except critic uses CKA decomposition.

**Verdict: ✅ Correct.**

---

## Checkpoint Path Verification

New format: `{ckpt_dir}/actor_{mode}_critic_{mode}_tid_{bool}_heads_{bool}/seed_{seed}/task_{id}.pkl`

Examples:
```
actor_cka_critic_persistent_tid_False_heads_True/seed_6/task_0.pkl
actor_reset_critic_reset_tid_False_heads_True/seed_6/task_0.pkl
actor_persistent_critic_persistent_tid_False_heads_True/seed_6/task_0.pkl
```

All 9 configurations produce unique directory names. No cross-contamination is possible.

Auto-resume uses `_ckpt_path(... actor_mode=FLAGS.actor_mode, critic_mode=FLAGS.critic_mode ...)` — it only finds checkpoints from the exact same configuration.

---

## Changes Detail

### `run_continual_contrastive.py`

| What | Why |
|---|---|
| `--reset_actor` (bool) → `--actor_mode` (string) | Clearer interface: `cka` / `reset` / `persistent` |
| `steps_per_task` default 1M → 8M | Match SGCRL training length |
| `base_steps` default 1M → 8M | Match SGCRL training length |
| `k_max` default 5 → 10 | More pool capacity |
| Checkpoint config_key format changed | Actor mode now explicit in path |
| Persistent actor: before-task passes composed_policy as theta_base | Actor continues from where it left off |
| Persistent actor: after-task sets theta_base = composed_policy | Carries forward the fully trained policy |
| All `FLAGS.reset_actor` → `FLAGS.actor_mode` comparisons | Unified interface |

### `contrastive/continual_learning.py`

| What | Why |
|---|---|
| `critic_mode='cka'` in `theta_base is None` branch | Reset actor + CKA critic was broken |
| `critic_was_freshly_init` boolean | Clean, unified logic for optimizer/target init |

### `draft_3.sh`

| What | Why |
|---|---|
| `RESET_ACTOR` → `ACTOR_MODE` (default: `cka`) | Matches new flag interface |
| `K_MAX` default → 10 | Matches new default |

### `contrastive/continual_config.py`

| What | Why |
|---|---|
| `k_max` 5 → 10 | Matches new default |
| `steps_per_task` → 8M | Matches SGCRL |
| `base_steps` → 8M | Matches SGCRL |

---

## Experiment Commands (seed 6, use_task_id=false, adapt_heads_only=true)

### Group A: Reset Actor

```bash
# A1: reset actor + reset critic (baseline1: fully independent)
ACTOR_MODE=reset CRITIC_MODE=reset USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# A2: reset actor + persistent critic (baseline2: critic-only transfer)
ACTOR_MODE=reset CRITIC_MODE=persistent USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# A3: reset actor + CKA critic
ACTOR_MODE=reset CRITIC_MODE=cka USE_TASK_ID=false SEED=6 sbatch draft_3.sh
```

### Group B: CKA Actor

```bash
# B1: CKA actor + persistent critic
ACTOR_MODE=cka CRITIC_MODE=persistent USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# B2: CKA actor + CKA critic
ACTOR_MODE=cka CRITIC_MODE=cka USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# B3: CKA actor + reset critic (baseline4: CKA-RL with GCRL)
ACTOR_MODE=cka CRITIC_MODE=reset USE_TASK_ID=false SEED=6 sbatch draft_3.sh
```

### Group C: Persistent Actor

```bash
# C1: persistent actor + persistent critic
ACTOR_MODE=persistent CRITIC_MODE=persistent USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# C2: persistent actor + reset critic (baseline3)
ACTOR_MODE=persistent CRITIC_MODE=reset USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# C3: persistent actor + CKA critic
ACTOR_MODE=persistent CRITIC_MODE=cka USE_TASK_ID=false SEED=6 sbatch draft_3.sh
```

Note: all commands use the user's requested defaults:
- `use_task_id=false` (passed explicitly)
- `adapt_heads_only=true` (default in draft_3.sh)
- `encoder_from_base=false` (default in draft_3.sh)
