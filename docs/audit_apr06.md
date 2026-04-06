# Code Audit — April 6, 2026

Detailed answers to cross-task evaluation, flag semantics, default comparison with CKA-RL, final evaluation protocol, auto-resume, and experiment commands.

---

## 1. Cross-Task Evaluation

### CKA-RL — code level

CKA-RL's codebase (`cka-rl-meta-world/`) has **no cross-task evaluation code**. Here is exactly what it does:

1. **Training loop** (`run_sac.py` line 367–511): Trains the model on a single task for `total_timesteps=1M` steps. During training, `charts/success` is logged per episode (stochastic actor).

2. **End-of-task eval** (`run_sac.py` line 512–514): After training, `eval_agent()` runs `num_evals=10` deterministic episodes and logs `charts/test_success`. This is evaluated on the **same environment that was just trained** — NOT on past tasks.

3. **Model saving** (`run_sac.py` line 520–522): `actor.model.save()` writes `fc.pt`, `fc_mean.pt`, `fc_logstd.pt`. The critic is NOT saved — it is reinitialized from scratch at each task.

4. **Orchestrator** (`run_experiments.py`): Loops through 20 tasks sequentially. For CKA-RL, passes `--prev-units` with all past task directories. The model constructor loads `fc` (body) from `latest_dir` and sets up `FuseLinear` vectors from all previous heads. There is **no separate cross-task evaluation step** after all tasks are done.

5. **Results extraction** (`process_results.py`):
   - `compute_performance()` (line 142–197): Reads `charts/success` (the **stochastic training curve**) from TensorBoard for each task. Takes the mean of the **last 10 data points** of each task's own training run. This gives the success rate at the END of training each task — not a re-evaluation with the final model.
   - `compute_forward_transfer()` (line 60–139): Computes AUC of the training curve for each method vs. the `simple` (from-scratch) baseline. FT = (area_above_baseline − area_below_baseline) / (1 − AUC_baseline).

**Key insight:** The CKA-RL codebase **never re-evaluates past tasks after the full sequence**. The "Performance" table in the paper is the success rate at the END of each task's OWN training, not a cross-task evaluation with the final model.

### CKA-RL — paper level

The paper (Section 5.4, Table 3) describes a separate "Cross-task Performance" evaluation:

> "To assess how well the final policy consolidates information acquired throughout the learning process, we freeze the model after the last task and evaluate its performance on all previously seen tasks."

Table 3 shows the **final policy π_θN** evaluated on all tasks. The average cross-task performance (0.3966) is much lower than the sequential average performance (0.7498), confirming that significant forgetting occurs even with CKA-RL.

**Mismatch:** The main "Performance" metric (Tables 1–2) uses end-of-training success per task (during its own training phase). The cross-task evaluation (Table 3) is a separate analysis using the single final model, shown only in the appendix. The code in the repo has no implementation of this cross-task eval — it was likely done offline or with a script not included.

### Self-CompoNet (the predecessor)

Self-CompoNet (Malagón et al., ICML 2024) uses a different architecture (modular growing network, not knowledge vectors), but the evaluation protocol is similar:

- Performance is measured as the success rate at the end of each task's training.
- Each task's own module is available at evaluation — the architecture grows, so past task modules are frozen and always usable.
- The encoder evolves freely across tasks (new modules are added, old ones are frozen).

Self-CompoNet does not have a single "final model" problem because the architecture explicitly stores per-task modules. CKA-RL pools knowledge vectors and can reconstruct per-task policies via different alpha blending — but this reconstruction is not done in the code or the main evaluation.

### Our code

Our cross-task evaluation (`run_continual_contrastive.py` lines 839–859) evaluates **all past tasks** after each task completes, using `composed_policy` — the composed policy from the current task (θ_base + pool_c + v_k with the CURRENT task's alpha and v_k). This measures: "can the current composed policy still solve past tasks?" This is actually closer to the CKA-RL paper's Table 3 protocol (evaluate with the latest model on all tasks) than their code's Protocol (which only records per-task training success).

**Current limitation:** We do not reconstruct per-task policies for evaluation. This is discussed in point 4 below.

---

## 2. `encoder_from_base` vs `adapt_heads_only`

These two flags control **different** things. They are NOT equivalent.

### `adapt_heads_only` (default: `True`)

**Controls: what goes into the knowledge pool after each task.**

This is a POST-TASK extraction flag. It has no effect during training.

- `True`: After task k, split v_k into head (output layers) and body (MLP encoder). The body portion is folded into θ_base: `θ_base_body += v_k_body`. Only the head portion is stored in the pool. Pool vectors have zero-valued body entries.
- `False`: After task k, v_k is stored in the pool in its entirety (body + head). θ_base is NOT updated.

**Code:** `run_continual_contrastive.py` lines 592–617.

### `encoder_from_base` (default: `False`)

**Controls: whether the body receives gradients during training.**

This is a TRAINING-TIME flag.

- `False`: Body receives full gradients via v_k. The optimizer updates v_k for all parameters (body + head). This means the body effectively fine-tunes on each task (with the delta accumulated in v_k, then folded back if `adapt_heads_only=True`).
- `True`: Body gradients are masked to zero for tasks k > 0. Only head parameters in v_k receive non-zero gradients. The encoder is frozen at θ_base from task 0.

**Code:** `contrastive/continual_learning.py` lines 156–159.

### Why they are not equivalent

| | `adapt_heads_only=True` + `encoder_from_base=False` (our default) | `adapt_heads_only=True` + `encoder_from_base=True` |
|---|---|---|
| **Body during training** | Receives full gradients via v_k | Gradients masked to zero |
| **Body after task** | v_k_body folded into θ_base (encoder evolves) | v_k_body is zero (encoder frozen) |
| **Pool body entries** | All zeros (body folded into base) | All zeros (body never changed) |
| **Matches CKA-RL** | YES — CKA-RL's body (self.fc) gets full gradients | NO — CKA-RL never freezes the body |

Think of it this way:
- `adapt_heads_only` decides the pool structure (head-only vectors vs. full vectors)
- `encoder_from_base` decides training dynamics (whether the encoder changes at all)

Setting `adapt_heads_only=True` with `encoder_from_base=False` means: "the encoder fine-tunes freely on each task (matching CKA-RL), but we only store head deltas in the pool (matching CKA-RL's FuseLinear-only decomposition)."

Setting `encoder_from_base=True` would freeze the encoder entirely — this is NOT what CKA-RL does.

---

## 3. Do Our Defaults Match CKA-RL?

### Settings that match

| Setting | CKA-RL | Ours | Match? |
|---|---|---|---|
| `fuse_shared=False, fuse_heads=True` | Knowledge vectors only in head | `adapt_heads_only=True` | ✅ |
| `encoder_from_base=False` | Body fine-tuned each task | `encoder_from_base=False` | ✅ |
| Body loading | From `latest_dir` (previous task) | Body folded into θ_base (equivalent) | ✅ |
| Critic | Reinitialized each task | `critic_mode=persistent` | ❌ |
| Discount | `gamma=0.99` | `discount=0.99` | ✅ |
| `tau` (target network) | `0.005` | `0.005` | ✅ |
| Task sequence | 20 tasks | 10 tasks (default) | ❌ (use `--use_20_tasks`) |
| Steps per task | 1M | 1M (`--steps_per_task=1000000`) | ✅ |
| Pool size | `pool_size=9` (default) → can hold 9 vectors | `k_max=5` | ❌ |

### Settings that differ (by design)

| Setting | CKA-RL | Ours | Reason |
|---|---|---|---|
| **RL algorithm** | SAC | Contrastive GCRL (SGCRL) | Core contribution |
| **Critic** | Reset each task | Persistent (default) | Hypothesis: contrastive critic transfers |
| **Reward** | Hand-crafted per task | Self-supervised (hindsight relabeling) | GCRL design |
| **Actor loss** | −Q(s,π(s)) (SAC) | −φ(s,π(s,g))ᵀψ(g) | GCRL design |
| **Batch size** | 128 | 256 | SGCRL default |
| **Learning rate** | 1e-3 (actor and critic) | 3e-4 (actor and critic) | SGCRL default |
| **Hidden layers** | (256, 256) | (256, 256) | ✅ Same |
| **Num SGD steps** | 1 per env step | 64 per env step | SGCRL design (high UTD ratio) |
| **Random actions** | First 10K steps | InitiallyRandomActor until first update | Functionally similar |
| **Pool size** | 9 | 5 | Should match — see below |

### Settings that are ambiguous

| Setting | CKA-RL | Ours | Note |
|---|---|---|---|
| **Pool size** | `pool_size=9` in code, paper says K_max=5-8 | `k_max=5` | Paper Fig 5 shows pool_size sensitivity. Their code default is 9, but Table A5 uses 5. Should verify which is used for main results. |
| **Alpha scale** | Hardcoded in FuseLinear | In `knowledge_pool.py` | Need to verify these match. |
| **Eval protocol** | charts/success (stochastic) at end of training | charts/test_success + cross-task eval | Ours is more rigorous |

### Recommendation

To run a config that is maximally comparable to CKA-RL (aside from the algorithm swap):
```bash
CRITIC_MODE=reset  # match CKA-RL's critic reset behavior
```
This is our "Experiment 1b: Reset critic" — the closest apples-to-apples comparison where only the RL algorithm differs.

---

## 4. Final Evaluation After Full Task Sequence

### Your understanding is correct

After all N tasks are trained, the final evaluation should test each task with its corresponding policy. This is different from what we currently do.

### Current behavior

We evaluate all past tasks using `composed_policy` — the composed policy from the CURRENT task (task k). This policy uses θ_base + Σ α_j v_j + v_k with the alpha and v_k from the most recently trained task. For past task j, this is NOT the optimal policy — v_j from the pool is blended in via alpha, but it's not the primary active vector.

### What should happen for the paper's final evaluation

For each past task j, reconstruct its policy:
$$\theta'^{(j)} = \theta_{\text{base}} + \sum_{i \neq j} \alpha_j^{(j)} v_i + v_j$$

where $\alpha^{(j)}$ are the blending weights that were optimal when task j was being trained.

However, CKA-RL's code does NOT do this either. Their "Performance" metric uses the stochastic success rate from the END of each task's own training run. The cross-task eval in Table 3 uses the single final model on all tasks (NO per-task reconstruction).

### What we should do

For the paper, report both:

1. **Per-task training performance** (P): Success rate at the end of each task's training. We already have this from the `evaluator` logger.
2. **Cross-task evaluation with final model** (Table 3 style): After all tasks, evaluate the final composed policy on every task. We already do this.
3. **(Optional, more principled)** Per-task policy reconstruction: Reconstruct each task's policy and evaluate. This requires saving per-task alpha weights, which we don't currently store. This is a future enhancement.

For now, metrics 1 and 2 are sufficient and match CKA-RL's reported metrics.

---

## 5. Auto-Resume from Latest Checkpoint

### Implementation

When `--start_task=0` (the default), the run now automatically scans for existing checkpoints matching the same config (seed, critic_mode, use_task_id, adapt_heads_only, reset_actor). If a checkpoint for task k is found, training resumes from task k+1.

**Code:** `run_continual_contrastive.py` lines 705–743.

### What is restored

From the checkpoint (`task_{k}.pkl`):

| State | Key | Description |
|---|---|---|
| `theta_base` | `theta_base` | Frozen base policy (with body folded in) |
| Knowledge pool | `pool_vectors` | All stored knowledge vectors (v_1 ... v_k) |
| Critic params | `q_params` | Contrastive critic φ, ψ |
| Target critic | `target_q_params` | Target network for critic |
| Critic optimizer | `q_optimizer_state` | Adam state for critic |
| (if CKA critic) | `q_base`, `critic_pool_vectors` | Critic base + critic knowledge pool |

### What is NOT restored (reinitialized each task)

- `v_k` — initialized to zeros for the new task
- `β_k` (blending weights) — initialized to N(0, 0.01) for the new task
- Replay buffer — fresh per task (no cross-task data leakage)
- Actor optimizer state — fresh per task
- W&B run — new run per task

### How to use

```bash
# First run — starts from task 0
CRITIC_MODE=persistent SEED=6 sbatch draft_3.sh

# Run is preempted after task 4...

# Re-submit the SAME command — auto-detects task 4 checkpoint, resumes from task 5
CRITIC_MODE=persistent SEED=6 sbatch draft_3.sh
```

To force restart from scratch, explicitly set `START_TASK=0` and delete existing checkpoints, or use a different seed.

To force start from a specific task:
```bash
START_TASK=3 CRITIC_MODE=persistent SEED=6 sbatch draft_3.sh
```

---

## 6. Experiment Commands (Seed 6)

### Core experiments

```bash
# Exp 1a: Persistent critic (our main hypothesis)
#   Contrastive critic carried forward across tasks, CKA actor decomposition.
CRITIC_MODE=persistent SEED=6 sbatch draft_3.sh

# Exp 1b: Reset critic (CKA-RL baseline behavior)
#   Critic reinitialized each task. Closest comparison to CKA-RL.
CRITIC_MODE=reset SEED=6 sbatch draft_3.sh

# Exp 1c: CKA critic
#   Knowledge-vector decomposition applied to critic too.
CRITIC_MODE=cka SEED=6 sbatch draft_3.sh
```

### Task ID ablation

```bash
# Exp 2: Persistent critic, no task ID
#   Tests whether the agent can distinguish tasks without explicit ID.
CRITIC_MODE=persistent USE_TASK_ID=false SEED=6 sbatch draft_3.sh
```

### Actor decomposition ablation

```bash
# Exp 3: Full-policy adaptation (not head-only)
#   Entire policy decomposed into base + vectors (not just the head).
ADAPT_HEADS_ONLY=false ENCODER_FROM_BASE=false SEED=6 sbatch draft_3.sh
```

### Baselines

```bash
# Exp 4: Fully independent (reset actor + reset critic)
#   No transfer at all. Each task trains from scratch.
RESET_ACTOR=true CRITIC_MODE=reset SEED=6 sbatch draft_3.sh

# Exp 5: Critic-only transfer (reset actor + persistent critic)
#   Tests whether contrastive critic alone provides forward transfer.
RESET_ACTOR=true CRITIC_MODE=persistent SEED=6 sbatch draft_3.sh
```

### Summary table

| # | Name | `CRITIC_MODE` | `USE_TASK_ID` | `ADAPT_HEADS_ONLY` | `RESET_ACTOR` |
|---|---|---|---|---|---|
| 1a | Persistent critic | persistent | true | true | false |
| 1b | Reset critic | reset | true | true | false |
| 1c | CKA critic | cka | true | true | false |
| 2 | No task ID | persistent | false | true | false |
| 3 | Full-policy | persistent | true | false | false |
| 4 | Fully independent | reset | true | true | true |
| 5 | Critic-only transfer | persistent | true | true | true |

All commands auto-resume if previously interrupted.

---

## Appendix: CKA-RL `eval_agent` vs Our Evaluator

| Aspect | CKA-RL (`eval_agent`) | Our evaluator |
|---|---|---|
| **When** | Once, at end of training | Every `eval_every` env steps |
| **Policy** | Stochastic (`actor.get_action()` samples) | Deterministic (`params.mode()`) |
| **Episodes** | 10 | 10 (configurable via `--eval_episodes`) |
| **Logged as** | `charts/test_success` | `evaluator` logger |
| **Cross-task** | No — only current task | Yes — all tasks seen so far |

Note: CKA-RL's `eval_agent` uses the stochastic policy (calls `actor.get_action()` which samples), but their `compute_performance()` reads `charts/success` from training episodes, not `charts/test_success`. So the "Performance" numbers in their paper are from stochastic training episodes, not from the deterministic test eval.
