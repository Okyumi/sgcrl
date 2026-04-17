# Implementation Plan: Continual Goal-Conditioned RL with Persistent Dual-Encoder Critic

This document lists all to-dos and concerns for implementing the algorithm that combines **goal-conditioned contrastive RL** (SGCRL / “A Single Goal is All You Need”) with **CKA-RL** in the **single-goal (sgcrl) codebase**, applied to the **10-task Meta-World (Sawyer) continual setting** used in CKA-RL. The critic (φ, ψ) is **never reset**; only the actor is adapted via base + knowledge vectors + α.

---

## 1. Algorithm Summary (from your design)

- **Base phase (task τ₁):** Train θ_base, φ, ψ on τ₁ with goal contrastive RL:
  - **Critic:** minimize L_InfoNCE(φ, ψ).
  - **Actor:** maximize E_π [−‖φ(s,a) − ψ(g)‖²].
- **Continual loop (k = 2 … N):**
  - **Critic transfer:** Do not reset; carry forward φ, ψ from task k−1.
  - **Actor construction:** θ' = θ_base + Σ_{j=1}^{k−1} α_j^k v_j + v_k, with α_k = softmax(β_k), v_k init to 0, β_k ~ N(0,1).
  - **Joint optimization on τ_k:** Fine-tune φ, ψ with L_InfoNCE(D_k); optimize v_k and β_k by maximizing E_{π_θ'} [−‖φ(s,a) − ψ(g)‖²].
  - **Knowledge preservation:** V ← V ∪ {v_k}.
  - **Pool management:** If |V| > K_max, merge the most similar pair (e.g. cosine), replace with v_merge (e.g. average), update V.

---

## 2. Environment and Space Consistency

### 2.1 Requirement

- **State, action, and goal dimensions must be consistent across all 10 tasks** so that:
  - One shared observation spec (state || goal) works for every task.
  - One policy network (and one critic) can be used for all tasks.

### 2.2 Current state

- **SGCRL** (`env_utils.py`): Three envs with **different** obs_dim:
  - `sawyer_bin`: obs_dim = 7, full obs = 14 (state 7 + goal 7).
  - `sawyer_box`: obs_dim = 11, full obs = 22.
  - `sawyer_peg`: obs_dim = 7, full obs = 14.
- **CKA-RL Meta-World** (`cka-rl-meta-world/tasks.py`): Uses `ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE`; task list = hammer-v2, push-wall-v2, faucet-close-v2, push-back-v2, stick-pull-v2, handle-press-side-v2, push-v2, shelf-place-v2, window-close-v2, peg-unplug-side-v2. Observation space may differ per task in raw Meta-World.

### 2.3 To-dos: Environment

- [ ] **Define a single (state_dim, goal_dim) for all 10 tasks.** Options:
  - **Option A:** Use the **maximum** state and goal size across the 10 tasks and **pad** smaller envs (e.g. zeros or repeated last dimension) so every env returns vectors of the same length.
  - **Option B:** Define a **common subset** of indices (e.g. end-effector + one object position + goal position) and have each wrapper expose only that subset so all envs have the same state_dim and goal_dim.
- [ ] **Add 10 environment wrappers in `env_utils.py`** that match the CKA-RL task list and the chosen convention (same as `Metaworld_analysis/ADDING_METAWORLD_TASKS_GUIDE.md` pattern):
  - hammer-v2, push-wall-v2, faucet-close-v2, push-back-v2, stick-pull-v2, handle-press-side-v2, push-v2, shelf-place-v2, window-close-v2, peg-unplug-side-v2.
- [ ] Each wrapper must:
  - Subclass the corresponding `ALL_V2_ENVIRONMENTS` (or goal-observable) env.
  - Expose **observation = [state, goal]** with **fixed state_dim and goal_dim** (via padding or projection).
  - Implement `_get_obs()`, `reset()`, `step()`, `observation_space`, and optional `fixed_start_end` for evaluation.
  - Use a **single** `max_episode_steps` (e.g. 150) for all, or document any per-task difference.
- [ ] **`env_utils.load(env_name, fixed_start_end)`:** Extend to accept a **task_id** or **env_name** for each of the 10 tasks and return the same `(gym_env, obs_dim, max_episode_steps)` where `obs_dim` is the **state** dimension (so full observation size = 2 * obs_dim if state and goal have same length, or state_dim + goal_dim otherwise). Ensure `obs_dim` (and goal size) is **identical** for all 10.
- [ ] **Action space:** Confirm all 10 use the same continuous action space (e.g. 4-D). If any task has different action dim, decide on padding or a single action_dim for the whole benchmark.

### 2.4 Concerns

- Meta-World V2 goal-observable envs may have different internal state lengths per task; padding or a common index set is necessary.
- `start_index` and `end_index` in the contrastive code define which part of state is “goal” for the critic; with a unified state/goal layout, ensure `config.start_index` and `config.end_index` are set once and work for all tasks.

---

## 3. Config and Constants

- [ ] **Continual-specific config** (e.g. in `contrastive/config.py` or a new `continual_config.py`):
  - `num_tasks = 10`
  - `K_max` (max knowledge pool size before merging)
  - `steps_per_task` (e.g. 1M per task, to match CKA-RL)
  - Optional: separate `max_number_of_steps` for task 1 (base) vs later tasks.
- [ ] **Single `obs_dim`, `goal_dim`, `action_dim`** used everywhere; set from the first env (or from a fixed constant) and assert in all 10 envs.

---

## 4. Actor: Base + Knowledge Vectors + α (CKA-RL in JAX)

### 4.1 Current state

- SGCRL uses a **single** policy network (Haiku MLP) in `contrastive/networks.py`; params are one flat `policy_params`.
- CKA-RL (PyTorch) uses `CkaRlAgent`: base encoder + FuseLinear heads with base + Σ α_j v_j + v_k; α = softmax(β); vectors stored and merged when |V| > K_max.

### 4.2 To-dos: Actor parameterization

- [ ] **Parameterize policy as** θ' = θ_base + Σ_{j=1}^{k−1} α_j^k v_j + v_k **in JAX**:
  - **Option A (recommended):** Store **θ_base** (one set of policy params), **list of vectors v_1,…,v_{k−1}** (each same structure as policy params or a subset, e.g. only head layers), **v_k** (current task vector), and **β_k** (length k−1 for α over previous tasks). Forward pass: compute α_k = softmax(β_k), then θ' = θ_base + Σ_j α_j v_j + v_k, and run the existing policy network with θ'. Optimizer only updates v_k and β_k (and optionally α_scale if you use it).
  - **Option B:** Implement a **FuseLinear-like** layer in Haiku so that the policy has “base” + “vector” parameters and α; then the learner holds base, list of vectors, current v_k, β_k.
- [ ] **Initialization:**
  - Task 1: v_1 = 0 (or θ_base is the only params); β_1 not used (or α_1 = 1).
  - Task k ≥ 2: v_k ← 0, β_k ~ N(0,1) (or small constant), α_k = softmax(β_k).
- [ ] **Which parameters to adapt:** In CKA-RL only the **actor head** (last layer(s)) are decomposed into base + vectors; the shared body can be frozen or also adapted. Decide for your setting (e.g. only head as in CKA-RL, or full policy). If only head: define a clear split (e.g. “last two linear layers” = head) and store base/vectors only for those.

### 4.3 Concerns

- JAX uses pytrees (dicts of arrays); “vector” params must have the same tree structure as the part of the policy they modify so that θ_base + Σ α_j v_j + v_k is well-defined.
- Gradient flow: only v_k and β_k receive gradients; θ_base and v_1,…,v_{k−1} are fixed during task k (except when merging).

---

## 5. Critic: Persistent φ, ψ (Never Reset)

### 5.1 Current state

- `contrastive/networks.py`: Critic is **q_network** with **sa_encoder** (φ(s,a)) and **g_encoder** (ψ(g)); L_InfoNCE is in `learning.py` (sigmoid BCE / contrastive loss). Learner holds `q_params`, `target_q_params`; both are updated every step.

### 5.2 To-dos: Critic persistence

- [ ] **Do not reinitialize φ, ψ when switching to a new task.** At the start of task k ≥ 2, **load** q_params (and target_q_params) from the end of task k−1. No “reset” or “reinit” of the critic.
- [ ] **Optional actor objective:** Your algorithm uses **−‖φ(s,a) − ψ(g)‖²** for the actor. Current SGCRL actor maximizes **diag(φ(s,a)^T ψ(g))** (inner product). Either:
  - Change actor loss to **−‖φ(s,a) − ψ(g)‖²** (Euclidean distance), or
  - Keep inner product and document that the “persistent dual-encoder” is the same, with the same L_InfoNCE for the critic. (Implement the one you specified.)
- [ ] **Fine-tune φ, ψ on τ_k:** Continue applying L_InfoNCE on data D_k from the current task so the critic adapts to the new task while staying shared.

### 5.3 Concerns

- If the goal or state distribution shifts a lot across tasks, a single (φ, ψ) may be a bottleneck; fine-tuning on each task should mitigate this.

---

## 6. Knowledge Pool and Merging

- [ ] **Maintain pool V = {v_1, …, v_{k−1}}** (list of vectors; each vector is a pytree matching the adapted part of the policy).
- [ ] **When |V| > K_max:**
  - Compute pairwise similarity (e.g. cosine) between all vectors (flatten or per-layer; define clearly).
  - Find (v_m, v_n) = argmax S_{i,j}.
  - Set v_merge = (v_m + v_n) / 2 (or your chosen merge).
  - Set V ← (V \ {v_m, v_n}) ∪ {v_merge}.
- [ ] **Index bookkeeping:** After merging, indices of tasks to vectors change. Either:
  - Keep a mapping “task_id → vector index” and update it when merging, or
  - Store vectors in a fixed-size buffer and manage “which slot is which task” so that α_k still has the right length (number of “active” vectors). The algorithm’s α has k−1 entries for task k; after merging you have fewer vectors, so β_k / α_k must have length = |V| (current pool size) and the sum Σ α_j v_j runs over current V.

### 6.1 Concerns

- Merging changes the number of vectors; the learner state (β_k, list of v’s) must be updated consistently. References from checkpointing (e.g. “task 3’s policy”) may need to point to “current θ_base + current V + that task’s α” rather than a fixed vector set.

---

## 7. Training Loop: Base Phase Then Continual Loop

### 7.1 Current state

- `lp_contrastive.py` builds one **DistributedLayout** and runs one environment until `max_number_of_steps`. There is no notion of “task id” or “switch task.”

### 7.2 To-dos: Orchestration

- [ ] **Two-phase control:**
  - **Phase 1 (base):** Run the existing pipeline for **task 0** only, for `steps_per_task` (or a configured base_steps). At the end, **save** θ_base = policy params, φ, ψ = q_params (and target). Initialize V = {} (or V = {0} as in CKA-RL if you store a placeholder).
  - **Phase 2 (continual):** For k = 2,…,N:
    - Load θ_base, φ, ψ, V from previous run (or in-memory).
    - Initialize v_k, β_k; build θ' = θ_base + Σ α_j v_j + v_k.
    - Run training on **task k** for `steps_per_task`: same replay/learner/actor loop but with env = task k, and learner state = (θ_base, V, v_k, β_k, q_params, target_q_params, optim states). Data collection uses π_θ'.
    - After task k: append v_k to V; run merge step if |V| > K_max; save checkpoint (θ_base, V, q_params, etc.).
- [ ] **Implementation options:**
  - **Option A:** **Single program, sequential tasks:** One LaunchPad program that runs task 0, then task 1, …, task 9. After each task, learner **saves** state and **reloads** with new v_k, β_k and (if merged) updated V; environment factory is switched to the next task; replay can be cleared or kept (see below).
  - **Option B:** **Separate process per task:** A driver script (e.g. `run_continual_contrastive.py`) that for each task_id runs a subprocess or in-process “run one task” that loads checkpoint from task_id−1 and saves checkpoint for task_id. Simpler to debug; less “single program” elegance.

### 7.3 Replay and dataset

- [ ] **Per-task replay:** Clear replay when switching task (only use D_k for task k). This matches “train on τ_k” and avoids mixing old task data unless you explicitly want replay from previous tasks.
- [ ] **Dataset iterator:** `make_dataset_iterator` and `flatten_fn` use `self._config.obs_dim` and `start_index`, `end_index`. With a unified obs_dim, no change needed except ensuring config is set for the continual run.
- [ ] **Adder:** Same EpisodeAdder; only the environment that feeds it changes per task.

### 7.4 Concerns

- If you clear replay each task, the buffer refills from scratch for task k; `learning_starts` / `min_replay_size` must be satisfied again for each task.
- Checkpointing: save both “policy” (θ_base + V + v_k + β_k as one exposed “policy” or as separate pieces) and “critic” (q_params, target_q_params) so you can resume and never throw away the critic.

---

## 8. Variable Source and Actor

### 8.1 Current state

- Learner’s `get_variables(self, names)` returns `policy` and `critic`. The actor’s VariableClient requests **`policy`** only.

### 8.2 To-dos

- [ ] **Continual learner** must expose a **policy** that is the **combined** θ' = θ_base + Σ α_j v_j + v_k (using current α_k and current V). So when the actor calls `get_variables(['policy'])`, it receives the params to run π_θ'. The learner computes θ' from (θ_base, V, v_k, β_k) before returning.
- [ ] **Optional:** Expose also `critic` if you ever need it on the actor side (e.g. for logging); not required for action selection.

### 8.3 Concerns

- Every time the actor pulls policy, it must get the latest θ'. So the learner’s `get_variables` must compute θ' from the current state (no stale cache of θ').

---

## 9. Checkpointing and Saving

- [ ] **What to save at the end of each task k:**
  - θ_base (policy base)
  - V (list of knowledge vectors)
  - For reproducibility: current task’s v_k (already in V after append) and β_k if you need to resume mid-task.
  - q_params, target_q_params (φ, ψ)
  - Optimizer states (policy optimizer for v_k, β_k; q_optimizer for φ, ψ).
- [ ] **Directory layout:** e.g. `logs/continual_goal_crl/task_0`, `task_1`, … or one directory with `checkpoint_task_0.pt`, `checkpoint_task_1.pt`, etc. Reuse the same `CheckpointingRunner` / saver pattern as in `distributed_layout` if you stay in one process.
- [ ] **Loading for next task:** When starting task k, load θ_base, V (after any merge from previous step), q_params, target_q_params; init v_k, β_k; restore optimizer states if you want exact continuation.

---

## 10. Evaluation

- [ ] **Per-task success:** For each task i, evaluate the **current** policy (or the policy at end of task i) on **task i**’s env (fixed goal or small set of goals). Log success rate per task.
- [ ] **Fixed-goal evaluation:** Reuse the same pattern as SGCRL: for each task, have a `fixed_goal_dict` or equivalent (e.g. one fixed goal per task) and run evaluator with that goal.
- [ ] **When to evaluate:** Every `eval_every` steps within a task; and optionally at the end of each task on **all** previous tasks to measure forgetting (e.g. evaluate π_θ_N on tasks 1,…,N).

---

## 11. Entrypoint and Flags

- [ ] **New entrypoint:** e.g. `lp_continual_contrastive.py` or a mode in `lp_contrastive.py` (e.g. `--continual=True`).
- [ ] **Flags:** `--num_tasks=10`, `--K_max=5` (or 8), `--steps_per_task=1000000`, `--task_id` (if running a single task in a driver script), `--load_from_task` (for loading checkpoint from a previous task).

---

## 12. Testing Strategy (step by step)

- [ ] **Step 1 – Envs:** Implement one extra task (e.g. hammer) in `env_utils.py` with the same obs_dim as one existing (e.g. sawyer_bin 7), run current SGCRL on it; confirm training and eval work.
- [ ] **Step 2 – Unified dims:** Decide and implement unified (state_dim, goal_dim) for all 10; add all 10 wrappers; run a quick loop that resets each env and checks `observation_space.shape` and `action_space.shape` are identical.
- [ ] **Step 3 – Single-task base:** Run base phase (task 0) only with the unified env; save checkpoint; confirm critic and policy are saved.
- [ ] **Step 4 – Actor CKA in JAX:** Implement θ' = θ_base + α v_1 + v_2 (two tasks only), train task 2 with v_2 and β_2 only; no merging yet. Check that gradients flow only to v_2 and β_2.
- [ ] **Step 5 – Persistent critic:** Run task 0, then task 1 without reinitializing critic; confirm q_params are loaded and fine-tuned.
- [ ] **Step 6 – Knowledge pool and merge:** Implement V and merge when |V| > K_max; test with K_max=2 and 3 tasks so that one merge happens.
- [ ] **Step 7 – Full loop:** Run tasks 0→1→2 sequentially in one script; then extend to all 10 and compare with CKA-RL (SAC) results if available.

---

## 13. File-Level To-Do Summary

| Area | Files to touch |
|------|-----------------|
| Envs | `env_utils.py` (add 10 task wrappers, unified dims), `contrastive/utils.py` (`make_environment` if env_name → task_id) |
| Config | `contrastive/config.py` or new continual config (num_tasks, K_max, steps_per_task) |
| Networks | `contrastive/networks.py` (optional: actor with base+vectors if you do Fuse-style in JAX) |
| Learner | New `contrastive/continual_learning.py` or extend `learning.py`: state = (θ_base, V, v_k, β_k, q_params, target_q_params, optim states); update step for task k; merge step; get_variables that return θ' and critic |
| Builder | New continual builder or extend `builder.py`: make_learner that builds continual learner; make_replay possibly per-task |
| Layout | New `continual_distributed_layout.py` or extend: orchestrate task sequence, swap env factory per task, reload learner state |
| Entrypoint | `lp_continual_contrastive.py` or `lp_contrastive.py` with `--continual` |
| Eval | Reuse evaluator with per-task env and fixed goals; log by task_id |
| Checkpointing | Save/load θ_base, V, q_params, v_k, β_k, optimizers; directory per task or single dir with task suffix |

---

## 14. Additional Concerns

- **JAX vs PyTorch:** CKA-RL reference is PyTorch; your base is JAX/Haiku. All “base + vectors” logic must be in JAX; no mixing of frameworks in the same learner.
- **Success metric:** Meta-World tasks use different success criteria (e.g. distance threshold). Each wrapper’s `step()` should return a reward or info that matches the task’s success definition; observers (e.g. `SuccessObserver`) should work with that.
- **Seeds:** Use a fixed seed per task (e.g. base_seed + task_id) for reproducibility.
- **Compute:** 10 tasks × 1M steps each = 10M steps total; ensure cluster/job limits and checkpointing allow long runs.

---

This plan should be enough to implement continual goal-conditioned RL with a persistent dual-encoder critic in the sgcrl codebase step by step and test as you go.
