> **Provenance.** Collaborator-supplied notes, shipped with
> `sparse_sac_continual_rl.zip` and reproduced here **verbatim** (apart from
> this banner). They describe the archive as delivered, which was cut from an
> older sgcrl snapshot, so file paths and a few details no longer match the
> integrated code — the integrated layout, the flags, and every deliberate
> departure from these notes are documented in
> [`docs/2026-07-28_sparse_sac_integration_and_run_guide.md`](../../docs/2026-07-28_sparse_sac_integration_and_run_guide.md).
> Where the two disagree, that guide is authoritative. Kept for the
> implementation rationale, which is not repeated elsewhere.

# Sparse-reward SAC + HER for Continual RL

A goal-conditioned SAC baseline for the continual Meta-World setting. It is a
drop-in mirror of our continual contrastive-RL (CRL) driver: **only the critic
and the losses were swapped**. The data pipeline, environments, replay, task
sequence, actor architecture, continual machinery, evaluation and logging are
shared with CRL so the two are directly comparable.

Implementation details are documented inline: **every deviation from the CRL
code you passed me is marked with a `# Modified` comment.** Since you know that
pipeline already, grepping for `# Modified` is the fastest way to see exactly
what is different and why.

---

## 1. What's in the bundle

**SAC-specific (new code)**

| File | What it is |
|---|---|
| `run_continual_sac.py` | Driver: task loop, replay, HER relabeling, eval, checkpointing |
| `contrastive/continual_sac_learning.py` | `ContinualSACLearner` — the three SAC losses + continual adaptation |
| `contrastive/sac_networks.py` | `make_sac_networks` — twin scalar Q + Gaussian-tanh actor |
| `experiment_configs_sac.py` | Variant × seed grid |
| `draft_sac.sh` | SLURM launcher |

**Shared with the CRL pipeline (unchanged unless noted in §3)**

| File | What it provides |
|---|---|
| `contrastive/config.py`, `contrastive/continual_config.py` | Hyperparameters, task sequence |
| `contrastive/networks.py` | `ResidualMLP` (LayerNorm + Swish + skip) |
| `contrastive/knowledge_pool.py` | Continual knowledge pool |
| `contrastive/utils.py` | Env construction, goal projection, observers |
| `contrastive/rl_metrics.py` | Representation metrics **(modified — see §3)** |
| `default.py` | Loggers **(modified — see §3)** |
| `contrastive/__init__.py` | Package exports **(modified — see §3)** |
| `env_utils.py`, `point_env.py` | Meta-World Sawyer envs |
| `distributional.py` | `NormalTanhDistribution` |

**Reference only** — `jaxgcrl_sac.py`, `jaxgcrl_sac_networks.py`: verbatim copies
of the JaxGCRL SAC the losses were ported from (it wraps
`brax.training.agents.sac`). Nothing imports them.

---

## 2. The reward

Standard SAC otherwise; check the algorithm box in the paper.

This is the one genuine data-pipeline change. CRL never reads the reward (InfoNCE
doesn't need it), so its relabeling function passes the env reward straight
through. SAC needs a TD signal consistent with the *relabeled* goal, so
`flatten_fn` in `run_continual_sac.py` recomputes it:

```
r_t        = 1[ ‖achieved_goal(s_{t+1}) − g_relabeled‖ < τ ]   (τ = --her_reward_threshold)
discount_t = (1 − r_t) · γ_env                                  (terminal bootstrap)
```

Two decisions inside this:

**Where the relabeled goal comes from.** I kept CRL's sampler — geometric
sampling over in-trajectory *future* states. JaxGCRL's SAC does something
different: it uses the in-trajectory *terminal* state as the goal source, falling
back to the originally commanded goal when no terminal is detected in the sampled
window. I went with the CRL sampler so the SAC and CRL runs see an identical
goal distribution and the comparison isolates the critic.

**Which state supplies the achieved goal.** I use `s_{t+1}`; JaxGCRL's
`flatten_batch` uses `s_t`. The next state is what `env.step` actually reports on
the true trajectory (reward delivered on arrival at the goal).

---

## 3. Changes to files shared with the CRL pipeline

Three shared files were touched. Two are additive; one changes behavior for CRL
runs too, so it is worth knowing about.

**`contrastive/rl_metrics.py` — additive, safe.** Adds `compute_sac_metrics`,
`extract_q_hidden_features`, `_get_q_head_final_weights`. Nothing existing was
modified. The SAC critic has no `φ/ψ` encoders, so the metrics are computed on
each Q head's penultimate-layer activations (the features the final `Dense(1)`
consumes), reported under `critic_q1/*` and `critic_q2/*`. All the primitives
(entropy, gini, feature rank, NRC1/NRC2, dormant ratio) are reused as-is, and
`actor/*` is identical to CRL since the actor architecture is shared. NRC1 uses
`target_dim=1` because Q is a scalar regressor.

**`default.py` — behavioral, affects CRL too.** `WandbLogger.write` no longer
passes an explicit `step=`. When several loggers (actor / learner / evaluator /
rl_metrics) share one W&B run their counters advance at very different rates, so
the explicit steps were non-monotonic and W&B silently *dropped* entries
("user provided step ... is less than current step"). Instead each driver
declares `wandb.define_metric(<family>/*, step_metric=<family>/<axis>)` and W&B
auto-increments the global step. **If you run the CRL driver from this bundle,
it needs those `define_metric` declarations too, or its x-axes shift.**

**`contrastive/__init__.py` — import hygiene, behavior-preserving.** It used to
eagerly import the whole CRL agent stack (`agents → builder →
distributed_layout`, `learning`, `continual_learning`, …). Since Python runs a
package's `__init__` before any submodule, that made even
`from contrastive.sac_networks import ...` pull in every contrastive-specific
module — SAC could not run without the entire CRL codebase. Exports are now
resolved lazily (PEP 562). Attribute access is unchanged:
`contrastive.ContrastiveConfig`, `from contrastive import KnowledgePool` and the
rest all still work.

---

## 4. Key design decisions

These are deliberate choices, not oversights — flagging them because they are the
things most likely to matter to you.

1. **Single global goal threshold.** `--her_reward_threshold` defaults to `0.05`
   for every task, whereas the envs themselves use per-task success thresholds
   (e.g. `stick_pull` 0.12, `hammer` 0.07). On those tasks the HER relabeled
   reward is therefore *stricter* than the env's own success rule. Kept uniform
   so the reward definition is one number rather than a per-task table; raise it
   if you see those tasks stalling.

2. **Update-to-data ratio ≈ 0.43.** The learner does 64 SGD steps once per
   150-step episode, inherited verbatim from the CRL pipeline. That is well below
   standard SAC's UTD of 1.0. This is intentional — the point of the baseline is
   that hyperparameters are held fixed across the two algorithms — but it does
   mean the SAC numbers are not a tuned-SAC upper bound.

3. **Step-penalty reward by default.** `r = 0` on goal reach, `−1` otherwise;
   `--nostep_penalty_reward` gives `+1 / 0` instead. I observed better
   performance with the step penalty turned on when tuning it. The reward shape
   is part of the checkpoint path key, so the two shapes can never cross-load a
   critic.

4. **Network width 256.** From memory, width 256 performed slightly better than
   1024 — so the launcher is set to 256 rather than the wider config.

5. **Target entropy `−0.5 · |A| = −2.0`**, following brax/JaxGCRL's convention
   rather than the more common `−|A|`. α is learned (adaptive entropy).

---

## 5. Running it

**Full runs.** How I actually do it: set everything in `draft_sac.sh` — it holds
all the flags in one block near the top — then just `sbatch` it. Before the first
submit, change the `#SBATCH --account` / `--mail-user`, the netID in every path
(there's a `>>> Replace netID <<<` marker), and the `source activate` line.

```bash
sbatch draft_sac.sh
```

The script runs one variant per array task (`--array=0-1`: `reset/reset` and
`persistent/persistent`) and fans 3 seeds out in parallel on that GPU.
Auto-resume probes backwards for the newest checkpoint matching the exact config
key (`actor_*_critic_*_tid_*_heads_*_rew_*`), so re-running a completed task is a
safe no-op and configurations never share checkpoints.

**Adjust the log / checkpoint names.** `LOG_DIR` and `CHECKPOINT_DIR` in
`draft_sac.sh` are named after my own sweeps — `CHECKPOINT_DIR` still ends in
`_W1024` even though the width is 256 now. Rename them to whatever tags the run
you are doing, otherwise separate experiments land in confusingly-labelled
directories.

Same for W&B: it is on by default and the `wandb.init` call in
`run_continual_sac.py` has my entity / project / group hardcoded
(`d_konoki` / `continual_sac` / `sac_test`) — point those at your own account
before running, or pass `--nouse_wandb`.
