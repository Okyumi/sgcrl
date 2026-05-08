# Previous-Replay Negative Bank — April 21, 2026

## Motivation

Continual RL has a natural **offline-to-online structure**: while learning task k (online), the agent has already accumulated replay buffers from tasks 0..k-1 (offline). Rather than discarding that data when starting a new task, we can use its reached goals as *extra negative samples* in the contrastive critic's InfoNCE objective, expanding the pool of negatives beyond the current batch.

This is a natural fit for contrastive GCRL: the critic's training signal is the softmax over the batch's goals, and the quality of that signal depends directly on the diversity of negatives. More (and better) negatives → harder contrastive task → richer representations.

## The Vanilla Variant and Its Problem

The simplest version: after each task finishes, dump its HER-relabeled goals into a bank. At each training step, sample `M` random goals from the bank and append them to the batch as extra negatives. The logits matrix grows from `[B, B]` to `[B, B+M]`, and the extended label is `[I_B, zeros(B, M)]`.

**Problem**: MetaWorld Sawyer tasks occupy different workspace regions. Hammer states live near the nail; push-wall states live near the wall; etc. A goal sampled from a previous task's buffer is **trivially distinguishable** from the current task's goals by raw coordinates alone. The critic can achieve high categorical accuracy without learning semantic features — it just memorises workspace regions. Gradient signal collapses, representation quality degrades, and the critic's learning actually *slows down* relative to the baseline.

This was observed empirically: with vanilla cross-task negatives, `categorical_accuracy` saturates near 1.0 very early but the actor's Q-values stay uninformative, and task performance suffers.

## The Principled Variant: `hard_weighted`

Two key ideas:

### 1. Per-anchor hard-negative mining

For each anchor `(s_i, a_i)` in the batch, score a candidate pool of bank goals:

```
score[i, c] = φ(s_i, a_i)^T ψ(g_cand[c])
```

Keep the top-M by score per anchor. These are the "hard" negatives — goals the critic currently assigns high score, meaning they lie near the current decision boundary. They provide the strongest gradient signal.

Easy negatives (low-score candidates, trivially different from the positive) are discarded. This directly addresses the "too-easy" problem of the vanilla variant: by actively selecting hardest candidates, we force the critic to learn features that distinguish genuinely goal-relevant from genuinely irrelevant states, not just which task region a goal came from.

Implementation: `jax.lax.top_k(scores, M)` is vectorised across the batch.

### 2. Logit down-weighting

Even hard-mined negatives are still cross-task, and some may be *false negatives* — states that happen to be valid goals for the current task too (e.g., workspace-center states). To limit their damage, scale the bank logits by `w_bank < 1` before concatenating:

```
extended_logits = concat([in_batch_logits, w_bank * bank_logits], axis=1)
```

A scalar `w_bank ∈ (0, 1]` reduces the "confidence" of bank negatives in the softmax. With `w_bank = 0.3`:
- Bank logits are 1/3 as strong as in-batch logits
- The softmax gradient through them is correspondingly smaller
- The in-batch contrast remains the dominant signal

## Implementation

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--neg_bank_mode` | `off` | `off`, `vanilla`, `hard_weighted` |
| `--neg_bank_n_per_step` | `256` | M: number of bank negatives per anchor (final count for hard_weighted; shared pool for vanilla) |
| `--neg_bank_candidate_pool` | `1024` | Pool size for hard mining (ignored for vanilla) |
| `--neg_bank_weight` | `0.3` | Logit weight for bank negatives |
| `--neg_bank_per_task_capacity` | `10000` | Goals stored per task |
| `--neg_bank_max_tasks` | `20` | Max tasks retained (FIFO) |

### Lifecycle

- **Task 0**: bank is empty; no bank negatives used. Learner trains exactly as before.
- **End of task k**: driver samples `per_task_capacity` HER-relabeled goals from task k's replay buffer (via the same iterator that feeds training, so goals are already post-`flatten_fn`). These are added to the bank.
- **Task k+1 onwards**: at each learner step, driver samples from the bank (up to `candidate_pool` goals for hard mining, or `n_per_step` for vanilla). These are passed to the JIT-compiled update step. The critic encodes them with `ψ`, computes `[B, bank_size]` logits, applies either vanilla concatenation or per-anchor top-K, then down-weights and appends to the in-batch logits.

### Shape invariance

The bank sampling size is constant across learner steps (determined by `neg_bank_candidate_pool` for hard_weighted or `neg_bank_n_per_step` for vanilla). This keeps the JIT compilation stable — no recompilation at bank-size changes.

When a new task begins, the learner is re-created with `task_id` captured as a closure variable. `neg_bank_active` is `False` at task 0 (using the 2-tuple data path) and `True` at task 1+ (using the 3-tuple data path). This is a one-time compilation switch per task boundary — acceptable overhead.

### Intentional scope

The bank currently only affects the **non-TD, non-twin-Q CPC critic loss** (our default: `alg=contrastive_cpc`, `twin_q=False`). The TD and twin-Q paths silently skip bank negatives. If we later enable those paths, they'd need analogous wiring.

### Metrics

When active, three new metrics appear in W&B:

- `bank/logits_mean`: average bank-negative logit (unweighted). Lower is better — means the critic correctly assigns bank goals low scores.
- `bank/logits_max`: mean-over-batch of the MAX bank logit per anchor. Measures the "hardest" negative the critic saw this step.
- `bank/extended_categorical_accuracy`: fraction of anchors where the positive beats ALL negatives (in-batch + bank).

If `bank/extended_categorical_accuracy` saturates near 1.0 very early, the negatives are too easy — either reduce `neg_bank_weight`, increase `neg_bank_candidate_pool`, or check whether cross-task goals are actually providing useful contrast.

## Files Changed

| File | Change |
|------|--------|
| `contrastive/negative_bank.py` | New — `NegativeBank` class, `pick_hard_negatives` utility |
| `contrastive/continual_learning.py` | `critic_loss_fn` now accepts optional `bank_goals`; `update_step` threads them through; `set_bank_goals()` method on learner |
| `run_continual_contrastive.py` | Flags, bank creation in `main()`, per-step sampling, post-task goal extraction |
| `draft_3.sh` / `draft_4.sh` | `NEG_BANK_*` env vars wired to flags; bank echo in run info |

## Also: `actor_auto_reset` default

The automatic actor reset is now **disabled by default** (`--actor_auto_reset=false`) to ensure the dormancy-triggered reset mechanism cannot interfere with any ablation experiment. To enable: set `ACTOR_AUTO_RESET=true` in the SLURM submission env.
--

# Principled variant (recommended)
SEED=1002 NEG_BANK_MODE=hard_weighted NEG_BANK_N_PER_STEP=256 \
  NEG_BANK_CANDIDATE_POOL=1024 NEG_BANK_WEIGHT=0.3 \
  sbatch draft_3.sh

# Vanilla variant (for comparison — expected to hurt)
SEED=1001 NEG_BANK_MODE=vanilla NEG_BANK_N_PER_STEP=256 NEG_BANK_WEIGHT=1.0 \
  sbatch draft_3.sh

# Disabled (default, current behavior)
sbatch draft_3.sh