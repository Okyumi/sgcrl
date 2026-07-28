# Sparse Goal-Conditioned SAC — Integration Notes and Run Guide

*2026-07-28.* Integrates the collaborator's sparse goal-conditioned SAC+HER
implementation as a continual baseline alongside the contrastive (CRL) driver.
A paired CRL/SAC run at the same seed and transfer mode isolates **the effect of
the critic**: everything else — Sawyer environments, wrappers, observation/goal
layout, HER relabeling, CKA actor decomposition, knowledge pool, checkpointing,
cross-task evaluation, W&B conventions — is shared.

The collaborator's own explanatory Markdown is preserved verbatim at
[`doc/reference/SAC_README.md`](../doc/reference/SAC_README.md).

---

## 1. What was added

Everything lives in the `sac/` package; nothing was dumped at the repo root
except the one entry point and its config enumerator/launcher, which mirror the
existing `run_continual_contrastive.py` / `experiment_configs.py` / `draft_3.sh`
trio.

| File | Contents | Heavy imports |
|---|---|---|
| `run_continual_sac.py` | **The supported entry point.** Thin: defines nothing, defers `sac.training` into `main()` so `--help` works bare. | no |
| `sac/flags.py` | Every flag with an explicit default; `resolve_tasks`, `build_continual_config`, `build_contrastive_params`, `wandb_run_config`. | no |
| `sac/tasks.py` | `FIXED_GOALS`, sequence resolution/validation, per-task step budget, and the actor transfer rules at task boundaries. | no |
| `sac/her.py` | The sparse reward + terminal-discount rule, backend-parameterized (numpy for tests, `tf` ops for the reverb pipeline). | no |
| `sac/checkpointing.py` | Path keying, save/load, auto-resume probe. | jax only |
| `sac/metrics.py` | SAC representation metrics for the actor trunk and both Q heads. | jax/haiku |
| `sac/networks.py` | `make_sac_networks`: twin scalar Q + the CRL-identical actor. | jax/haiku/acme |
| `sac/learning.py` | `ContinualSACLearner`: SAC TD loss, actor loss against min-Q, adaptive alpha, CKA composition. | jax/haiku/acme |
| `sac/training.py` | The training loop: env, reverb, `tf.data` pipeline, actors, evaluator, W&B, checkpoints. | reverb + TF + acme |
| `sac/reference/` | Vendored Apache-2.0 brax/JaxGCRL SAC for provenance. **Nothing imports it.** | — |
| `experiment_configs_sac.py` | Enumerates the sweep grid for the launcher. | no |
| `draft_sac.sh` | SLURM launcher (R/R, P/P, smoke, two-cell sweep). | — |
| `tests/test_sac_*.py` | 197 tests, all runnable without reverb/TF. | no |

Modified: `contrastive/__init__.py` (made lazy — see §7), `default.py` (opt-in
W&B auto-step — see §7). No behavior change to any contrastive/DCC path.

**No dependency changes.** Everything needed is already pinned in
`requirements.txt`.

---

## 2. Environment setup

Same environment as the contrastive driver — no extra packages.

```bash
# HPC (NYU Greene): the launcher does this for you.
module purge && module load cuda/11.8.0 conda-gcc/11.2.0
eval "$(conda shell.bash hook)" && conda activate contrastive_rl
export MUJOCO_GL=egl                                   # headless rendering
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python   # protobuf 4.x compat

# Local, from scratch:
pip install -r requirements.txt
```

W&B credentials are **never** stored in the repo. Either run `wandb login` once
on the login node, or use `--wandb_mode=offline` and `wandb sync` later.

---

## 3. Commands

### Smoke test (minutes, no W&B)

```bash
python run_continual_sac.py \
    --seed=42 --num_tasks=2 --steps_per_task=10000 --base_steps=10000 \
    --k_max=2 --eval_every=5000 --eval_episodes=2 --nouse_wandb
```

Two tasks, an explicit pair, offline W&B:

```bash
python run_continual_sac.py \
    --task_sequence=sawyer_push,sawyer_hammer \
    --steps_per_task=10000 --base_steps=10000 --wandb_mode=offline
```

### Full runs, 10-task sequence

```bash
# R/R — fully independent per task (no transfer at all).
python run_continual_sac.py --seed=1 --actor_mode=reset --critic_mode=reset

# P/P — both actor and critic carry over.
python run_continual_sac.py --seed=1 --actor_mode=persistent \
    --critic_mode=persistent
```

`--actor_mode=cka --critic_mode=persistent` also works (identical pool
machinery to CRL), but the SAC baseline exists to measure the critic, so the
default sweep is the R/R and P/P diagonal.

### W&B

```bash
--wandb_project=continual_sac --wandb_group=sac_rr_seed1   # one group per cell
--wandb_entity=my-team                                     # empty = local login
--wandb_mode=online | offline | disabled
--nouse_wandb                                              # skip entirely
```

Offline runs land in `wandb/`; sync with `wandb sync wandb/offline-run-*`.

### HPC (SLURM)

```bash
# The whole default grid: 6 runs (R/R + P/P x seeds 1,2,3) over 3 GPUs.
sbatch draft_sac.sh

# Size the array yourself:
total=$(python experiment_configs_sac.py --total)   # 6
python experiment_configs_sac.py --list            # the table

# One run, no config file — override anything via env vars.
ACTOR_MODE=reset CRITIC_MODE=reset SEED=1 SINGLE_RUN=true sbatch draft_sac.sh
ACTOR_MODE=persistent CRITIC_MODE=persistent SEED=1 SINGLE_RUN=true \
    sbatch draft_sac.sh

# Smoke test through the launcher (2 tasks x 10k steps, W&B forced offline).
SMOKE=true SINGLE_RUN=true sbatch draft_sac.sh
```

Two runs share each GPU (`TASKS_PER_GPU=2`,
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`). `TASKS_PER_GPU * fraction` must stay
below 1.

To change the sweep, edit `ACTOR_CRITIC_PAIRS` / `SEEDS` in
`experiment_configs_sac.py`, or append a dict to `CELLS` for anything outside
the Cartesian product (there are commented reward-shape and goal-threshold
ablation examples in the file).

---

## 4. Key flags

`python run_continual_sac.py --help` is the full reference. The ones that
matter:

| Flag | Default | Notes |
|---|---|---|
| `--seed` | 42 | Part of the checkpoint path. |
| `--actor_mode` | `cka` | `cka` / `reset` / `persistent`. |
| `--critic_mode` | `persistent` | `persistent` / `reset` / `cka`. |
| `--num_tasks` | 10 | Clamped to the sequence length. |
| `--task_sequence` | `''` | **SAC-only.** Comma-separated names, in order. Beats `--use_20_tasks`. |
| `--single_task` | `''` | Overrides everything; forces one task. |
| `--use_20_tasks` | false | Two passes of the 10-task sequence. |
| `--base_steps` / `--steps_per_task` | 8e6 | Task 0 vs. tasks 1..N-1. |
| `--her_reward_threshold` | 0.05 | **SAC-only.** Goal radius τ. Looser tasks (`stick_pull`, `hammer`) may want 0.07–0.12. |
| `--step_penalty_reward` | true | **SAC-only.** `0/-1` step penalty; `--nostep_penalty_reward` gives `+1/0`. Part of the checkpoint path. |
| `--k_max` | 10 | Knowledge-pool size before merging. |
| `--adapt_heads_only` | true | Pool only actor head deltas; body deltas fold into `theta_base`. |
| `--use_task_id` | false | One-hot task ID appended to state *and* goal. |
| `--network_width` / `--critic_depth` / `--actor_depth` | 256 / 4 / 4 | `ResidualMLP`; `--nouse_residual` for a plain MLP. |
| `--eval_every` / `--eval_episodes` | 50000 / 10 | `--eval_episodes=0` disables both evaluators. |
| `--k_sample_k` | 0 | K-sample-argmax eval scored by `min(Q1,Q2)`; 0 = deterministic mean. |
| `--log_rl_metrics` | true | Representation metrics (§6). |
| `--checkpoint_dir` | `logs/continual_sac_checkpoints` | See §5. |
| `--start_task` / `--auto_resume` | 0 / true | See §5. |
| `--alg` | `sac_her` | Log/checkpoint subtree tag. Leave it alone. |

---

## 5. Outputs, checkpoints and resume

```
{checkpoint_dir}/actor_{mode}_critic_{mode}_tid_{bool}_heads_{bool}_rew_{steppen|sparse01}/
    seed_{seed}/task_{id}.pkl

{log_dir}/continual_sac_her/<same config key>/task{id}_{env}_s{seed}/
    learner/logs.csv, actor/logs.csv, evaluator/logs.csv
```

The `_rew_*` suffix is the one addition over the CRL path format: the critic's
TD target is trained against **one** reward shape, so a `steppen` checkpoint
must never be loadable by a `sparse01` run. The path key enforces that
structurally rather than by a runtime check. (The CRL `_dyn*_pt*` decomposed-
critic suffix has no SAC analogue and is absent.)

Checkpoint contents: `theta_base`, `pool_vectors`, `q_params`,
`target_q_params`, `q_optimizer_state`, `composed_policy`, `task_id`,
`env_name`, plus `q_base` and `critic_pool_vectors` when `critic_mode=cka`.
Plain pickles of numpy pytrees, same as CRL, so `results/scripts/` reads them
unchanged.

**Resume.** With `--start_task=0` (default) and `--auto_resume`, the driver
probes the checkpoint directory backwards for the newest task matching *this
exact config key and seed* and restarts after it. A completed sequence resumes
at `num_tasks`, i.e. re-running a finished cell is a safe no-op. Pass
`--noauto_resume` to always start at task 0 (what you want when reusing a
`checkpoint_dir` for a fresh sweep), or `--start_task=N` to force a specific
task, which loads `task_{N-1}.pkl`.

Changing *any* of seed, `actor_mode`, `critic_mode`, `use_task_id`,
`adapt_heads_only`, or `step_penalty_reward` changes the path, so cells never
collide and never accidentally resume from each other.

---

## 6. Expected metric names

W&B families and the x-axis each is plotted against (declared with
`define_metric`, see §7):

| Family | x-axis | Keys |
|---|---|---|
| `learner/*` | `learner/steps` | `critic_loss`, `actor_loss`, `alpha`, `alpha_loss`, `q_mean`, `q_std`, `target_mean`, `td_error_abs`, `reward_mean`, `reward_pos_rate`, `discount_mean`, `entropy_mean`, `q_pi_mean`, `alpha_weights_max`, `alpha_scale` |
| `actor/*` | `actor/steps` | `episode_return`, `episode_length`, `success`, `final_dist`, `steps_per_second` |
| `evaluator/*` | `evaluator/steps` | same keys, deterministic policy |
| `rl_metrics/*` | `rl_metrics/env_steps` | `actor/{weight_norm,final_layer_norm,entropy,gini,feature_rank,nrc1,nrc2,dormant_ratio}`, `critic/weight_norm`, `critic_q1/*`, `critic_q2/*` |
| `intra_eval/*` | `intra_eval/env_steps` | per-task success + `mean_success` (needs `--intra_eval_previous_tasks`) |
| `eval/*` | `eval/num_tasks_seen` | post-task sweep over every task seen + `mean_success` — **this is the forgetting/transfer plot** |
| `actor_reset/*` | `actor_reset/env_steps` | `triggered`, `dormant_ratio_at_reset`, `count` (needs `--actor_auto_reset`) |

Sanity checks on a healthy run: `learner/reward_pos_rate` should climb off zero
once HER starts producing reached goals; `learner/discount_mean` should fall
below γ=0.99 as terminal transitions appear; `learner/alpha` should decay from
1.0; `actor/dormant_ratio` should stay low.

---

## 7. Deliberate deviations from the archive

Where the archive and the current repo disagreed, **current sgcrl interfaces
won**. The archive was cut from an older snapshot, so these current files were
kept as-is and *not* overwritten: `contrastive/continual_config.py` (the
archive lacks every DCC/CKA-diagnostic field), `contrastive/networks.py` (lacks
`ACTOR_HEAD_PATH_TAGS` / `is_actor_head_path`), `contrastive/knowledge_pool.py`
(pre-`CKAPool`), plus `env_utils.py`, `distributional.py`, `point_env.py`,
`contrastive/config.py`, `contrastive/utils.py`, `requirements.txt`.

The five deliberate departures from the collaborator's code:

1. **`default.py` W&B stepping is opt-in.** The archive deleted the explicit
   `step=` from `WandbLogger.write`, which is *required* here — several loggers
   share one run and their counters advance at different rates, so wandb drops
   the family whose step went backwards. Removing it outright would have
   changed CRL behavior, so it is now a `wandb_auto_step` parameter defaulting
   to `False`. Only the SAC driver passes `True`, and it pairs that with
   `wandb.define_metric` declarations so each family keeps its own x-axis.
2. **SAC metrics live in `sac/metrics.py`**, not appended to the shared
   `contrastive/rl_metrics.py`. It imports the primitives from there, so each
   metric still has exactly one implementation, but the shared module is
   untouched — zero risk to contrastive/DCC.
3. **`contrastive/__init__.py` is lazy** (PEP 562). It previously ran 18 eager
   imports, so `import contrastive.rl_metrics` dragged in
   `agents → builder → distributed_layout → launchpad`. The same 14 public
   names are still exported; they now resolve on first attribute access. This is
   what lets `sac/*` reuse `contrastive.rl_metrics`, `contrastive.networks` and
   `contrastive.knowledge_pool` without pulling launchpad into a `--help` call.
4. **`FIXED_GOALS` is duplicated** in `sac/tasks.py` rather than imported from
   `run_continual_contrastive.py` (which imports reverb at module level, so the
   light modules cannot touch it). `tests/test_sac_tasks.py` AST-parses the CRL
   driver and asserts the two tables are identical, so drift fails the suite.
5. **Credentials and account details removed.** The archive hard-coded a W&B
   entity/project/group and another user's netID, SLURM account and email; all
   are now flags / env-overridable launcher variables. No API key anywhere.

SAC-specific behavior differences from the CRL driver, by design:

* The reverb pipeline **recomputes reward and discount** from the sampled
  relabeled goal (`sac.her`). SAC needs a TD signal consistent with that goal;
  InfoNCE never reads the reward at all.
* Contrastive-only machinery is absent: negative bank, `energy_fn`,
  `logsumexp_penalty`, and the dual-encoder representation metrics.
* `--alg` defaults to `sac_her` so logs and checkpoints never collide with CRL.
* `target_entropy=-2.0` (= `-0.5 * |A|` for the 4-D Sawyer action space, the
  brax/JaxGCRL convention) with adaptive alpha; twin Q always on.

Everything else is shared with CRL: the same direct Sawyer environments and
wrappers via `contrastive.utils.make_environment`, the same unified
state/goal layout and `start_index`/`end_index` semantics, the same geometric
future-state HER sampling, the same fixed per-task goals, the same actor
architecture.

---

## 8. Tests

```bash
python -m pytest tests/ -q                    # 197 passed
python run_continual_sac.py --help            # full flag reference
python -m compileall -q sac/ run_continual_sac.py experiment_configs_sac.py
bash -n draft_sac.sh
python experiment_configs_sac.py --list
```

| File | Covers |
|---|---|
| `tests/test_sac_her.py` | Both reward shapes, terminal-discount zeroing, strict-inequality τ, Euclidean norm over all goal dims, dtypes/shapes, `sparse01 == steppen + 1`. |
| `tests/test_sac_tasks.py` | `FIXED_GOALS` drift guard vs. the CRL driver, sequence resolution and precedence, step budget, transfer/reset at task boundaries incl. full R/R and P/P round trips. |
| `tests/test_sac_flags.py` | Every flag default, enum validation, the four config builders, and that every flag `draft_sac.sh` passes actually exists. |
| `tests/test_sac_checkpointing.py` | Path keying (every ablation axis separates), save/load round trip, numpy-on-disk, resume incl. gaps, other seeds, other cells, and the finished-sequence case. |
| `tests/test_sac_training_wiring.py` | AST checks on the un-importable driver: W&B step axes, auto-step opt-in (on for SAC, off for CRL), entry-point import hygiene. |

---

## 9. Known limitations

* **`sac/training.py`, `sac/learning.py` and `sac/networks.py` were never
  executed.** This sandbox has `jax 0.10` while the repo pins `jax 0.3.13`, and
  `acme 0.4.0` uses the long-removed `jax.xla.Device` — so *every* module that
  imports acme fails here, `contrastive/learning.py` and
  `contrastive/networks.py` included. Verification was therefore static:
  byte-compilation, plus an AST cross-reference confirming all 37 cross-module
  imports and all 25 `module.attr` accesses in `sac/` resolve to real
  definitions. **Run the §3 smoke test on the pinned environment before
  launching a sweep.**
* `reverb` and `tensorflow` are not installed here either, which is why the
  light/heavy module split exists at all.
* The default `--her_reward_threshold=0.05` is tuned for
  `push` / `window_close` / `faucet_close`. `stick_pull` and `hammer` are
  looser; if their success rate stays pinned at zero, raise τ (0.07–0.12) —
  but note that changes the checkpoint path, so it cannot resume from a τ=0.05
  run.
* No SAC entry exists in `results/scripts/`; the analysis scripts read the
  checkpoints unchanged, but any SAC-vs-CRL comparison plot still has to be
  written.
