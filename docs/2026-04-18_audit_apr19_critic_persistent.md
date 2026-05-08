# Investigation: critic_mode=persistent — April 19, 2026

## Report

**Observation**: User reports "essentially no learning at all even for the first task when critic_mode=persistent."

**Investigation approach**: Exhaustive line-by-line trace of every code path affected by `critic_mode` in both `run_continual_contrastive.py` and `contrastive/continual_learning.py`.

---

## Finding: Task 0 is provably identical across all critic modes

After tracing every code path, I can confirm that at **task 0**, the behavior is **provably identical** for `critic_mode=persistent`, `reset`, and `cka`. Here is the proof:

### State entering task 0

For all modes, the main loop starts with:
```python
theta_base = None
prev_q = None
prev_tgt_q = None
prev_q_opt = None
```

### Actor mode branching (lines 922-935)

For all modes at task_id=0, the `else` branch is taken:
```python
_theta_base = theta_base   # None
_pool = pool               # empty KnowledgePool
```

### Learner initialization (continual_learning.py lines 460-567)

Since `theta_base is None`:
```python
policy_params = networks.policy_network.init(key_policy)  # fresh
theta_base = policy_params

# Critic: prev_q_params is None, so all modes fall to:
q_params = networks.q_network.init(key_q)  # fresh random init
```

The `critic_mode` is only checked in the `if prev_q_params is not None and critic_mode == 'persistent'` condition — which is NOT taken because `prev_q_params is None`.

### critic_was_freshly_init (line 529-533)

```python
critic_was_freshly_init = (
    task_id == 0                    # TRUE
    or critic_mode in ('reset', 'cka')
    or prev_q_optimizer_state is None
)
```

Result: `True` for all modes (first condition is True).

Consequence:
- `q_opt_state = q_optimizer.init(q_params)` — fresh optimizer for ALL modes
- `target_q = q_params` — target = initial params for ALL modes

### Remaining initialization

`v_k`, `beta_k`, `alpha_scale`, `pool_contribution` — all identical regardless of `critic_mode`.

### Training loop

The `update_step` function uses `state.q_params`, `state.target_q_params`, `state.q_optimizer_state` — all of which were initialized identically.

### RNG keys

`rng = jax.random.PRNGKey(seed + task_id * 1000)` — independent of `critic_mode`.

**Conclusion: There is no code path through which `critic_mode` can affect task 0 behavior.**

---

## Possible explanations for the user's observation

### 1. Issue is at task 1+, not task 0

The most likely scenario. At task 1, `critic_mode=persistent` carries forward the critic from task 0:
- `q_params = prev_q_params` (trained critic from task 0)
- `q_opt_state = prev_q_optimizer_state` (optimizer state from task 0)
- `target_q = prev_target_q_params` (target network from task 0)

If the user was looking at aggregate metrics or W&B panels that don't clearly separate task boundaries, task 1's poor performance could be mistaken for "no learning at all, even at task 0."

### 2. Different experiment flags between comparison runs

If the user compared a `persistent` run vs a `reset` run but used slightly different flags (e.g., different `use_task_id`, `adapt_heads_only`, or `encoder_from_base`), the performance difference could be attributed to the wrong variable.

### 3. W&B panel configuration

Different W&B runs may use different x-axes (env_steps vs learner_steps), which could make one appear flat.

---

## Verified behavior at task 1+ with critic_mode=persistent

For completeness, here's what happens at task 1:

| Aspect | persistent | reset | cka |
|---|---|---|---|
| q_params init | Carried from task 0 | Fresh random | q_base + pool_c |
| q_opt_state | Carried from task 0 | Fresh | Fresh |
| target_q | Carried from task 0 | = q_params (fresh) | = composed q |
| critic_was_freshly_init | False | True | True |

The persistent mode correctly carries forward all critic state. The optimizer state (Adam m/v running averages) is preserved, which provides momentum from the previous task's training.

### Potential training dynamics issue (NOT a code bug)

With `use_cpc=True` (the default), the **target network is never used in the CPC loss**. The CPC loss (`optax.softmax_cross_entropy + logsumexp_penalty`) only depends on `q_params`, not `target_q_params`. The target update is computed but has no effect on the loss.

This means:
- At task 1 with persistent, the target_q_params from task 0 are carried forward but never used
- The actual learning signal comes entirely from the CPC InfoNCE loss on the current task's data
- There is no "stability anchor" that the target network would normally provide

This is NOT a bug — it's the intended CPC design. But it's worth noting.

---

## Recommendation

To isolate the issue, the user should:

1. **Compare W&B curves for task 0 only** across `critic_mode=persistent` vs `critic_mode=reset` with all other flags identical
2. **Check exact flag values** — ensure `adapt_heads_only`, `encoder_from_base`, `use_task_id`, and `actor_mode` are the same between comparison runs
3. **Look at per-task metrics** (critic_loss, categorical_accuracy, logits_pos) rather than aggregate success rate
4. If the issue is truly at task 1+, the investigation should focus on whether the persistent critic's representations from task 0 (hammer) transfer to task 1 (push_wall) — this is the core research question, not a bug

---

## Files examined

- `run_continual_contrastive.py` — lines 270-765 (train_single_task), 769-1016 (main)
- `contrastive/continual_learning.py` — full file (818 lines)
- `contrastive/networks.py` — make_networks, _critic_fn, _actor_fn, _repr_fn
- `contrastive/config.py` — ContrastiveConfig defaults
- `contrastive/utils.py` — InitiallyRandomActor
- `env_utils.py` — TaskIDGymWrapper, load()
- `draft_3.sh` — default flag values
