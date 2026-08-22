# Bridge-DCC action-credit experiments

Date: 2026-08-22
Branch: `section3_done`
Initial implementation commit: `0678a10`

## Motivation and diagnosis

The completed Task 5 (`sawyer_handle_press_side`) and Task 8
(`sawyer_window_close`) diagnostics support a critic-landscape failure rather
than actor exploitation as the primary cause:

- DCC retained almost all categorical goal-retrieval accuracy when replay
  actions were shuffled or replaced with zero actions.
- Same-state DCC action rankings had approximately zero or negative
  correlation with measured mechanism progress.
- Policy actions did not exhibit the predicted high-score, poor-outcome, and
  far-from-replay-support signature expected under critic exploitation.
- CRTR/in-trajectory negatives with repetition factor `r=12` improved success
  AUC, but did not solve either task.

The remaining hypothesis is that two problems interact:

1. Contact/interaction states form a sparse bridge in the replay trajectory.
2. The contrastive retrieval objective does not directly train local,
   goal-conditioned action ordering.

## Implemented variants

All new variants retain DCC, the dynamics auxiliary, and CRTR `r=12`.

### IWR-DCC

Interaction-weighted relabeling changes the future-goal sampling probability
to

$$
p(t'\mid t) \propto
\mathbf{1}[t'>t]\gamma^{t'-t}
\left[
\epsilon + \exp\left(
-\frac{1}{2}\left(\frac{d_{t'}-c}{\sigma}\right)^2
\right)
\right],
$$

where $d_{t'}$ is the hand-to-mechanism distance, $c=0.09$,
$\sigma=0.03$, and $\epsilon=0.05$. The floor preserves support over all
future states while increasing the frequency of interaction-boundary goals.

### Advantage-DCC

A task-local forward action-effect head predicts the change in normalized DCC
goal geometry:

$$
y_t = \operatorname{sg}\left[
\gamma d_t\bar\psi(s_{t+1})-\bar\psi(s_t)
\right],
$$

$$
\mathcal{L}_{\mathrm{effect}}
=\mathbb{E}_t\left[
\operatorname{Huber}(u_k(s_t,a_t),y_t)
\right].
$$

The local action-comparative score is

$$
A_k(s,a,g)=\tanh\left(
\frac{u_k(s,a)^\top\bar\psi(g)}{\tau}
\right).
$$

This is a forward-effect objective: it learns what the action changes in goal
space. It is not AC-DCC's inverse-dynamics objective of identifying which
replay action explains an already observed next state.

The actor maximizes

$$
C(s,a,g)=
\frac{f_{\mathrm{DCC}}(s,a,g)}{\max(m,10^{-3})}
+\beta A_k(s,a,g),
$$

where $m$ is an EMA of the mean absolute DCC score. Defaults are
$\beta=1$, $\tau=1$, and EMA decay $0.99$.

### Bridge-DCC

Bridge-DCC combines IWR-DCC and Advantage-DCC. It also interaction-weights the
forward-effect loss so rare contact transitions have adequate influence on the
local control head.

## Experiment matrix

| Config indices | Variant | Task | Seeds |
|---:|---|---|---|
| 18–20 | IWR-DCC | Task 5 | 5, 6, 7 |
| 21–23 | IWR-DCC | Task 8 | 5, 6, 7 |
| 24–26 | Advantage-DCC | Task 5 | 5, 6, 7 |
| 27–29 | Advantage-DCC | Task 8 | 5, 6, 7 |
| 30–32 | Bridge-DCC | Task 5 | 5, 6, 7 |
| 33–35 | Bridge-DCC | Task 8 | 5, 6, 7 |

Matched CRTR controls are indices 6–11.

## Causal acceptance diagnostics

The action-landscape probe now:

- searches up to 200 policy steps for the closest interaction state;
- records whether the anchor reached the `0.09` interaction threshold;
- compares policy, local, replay-neighbour, and uniform actions from the same
  restored MuJoCo state;
- uses a 100-step common-policy continuation rollout;
- records score/outcome Spearman correlation, top-score regret, outcome
  variance, policy score–outcome percentile gap, and replay-support distance.

Important W&B metrics include:

- `iwr/selected_near_bridge_fraction`
- `iwr/selected_interaction_distance`
- `action_effect/loss`
- `action_effect/cosine`
- `action_effect/advantage_std`
- `action_landscape/anchor_near_interaction`
- `action_landscape/score_vs_rollout_mechanism_progress_spearman`
- `action_landscape/top_score_rollout_mechanism_progress_regret`
- `action_landscape/policy_score_outcome_percentile_gap`

Success AUC on Tasks 5 and 8 remains the primary outcome. Improved auxiliary
metrics without improved success are not sufficient evidence of a solution.

## Code locations

- `contrastive/decomposed_networks.py`: task-local action-effect network.
- `contrastive/continual_learning_decomposed.py`: effect loss, combined actor
  objective, score EMA, and IWR/effect metrics.
- `run_continual_contrastive.py`: new modes, IWR sampler, manifests, guards,
  and combined-score diagnostics.
- `contrastive/action_ranking_diagnostics.py`: interaction-aware anchors and
  longer counterfactual outcomes.
- `experiment_configs.py`: indices 18–35.
- `DRAFT.sh`: canonical flag and runtime forwarding.
- `DRAFT_bridge_dcc.sh`: Torch batch wrapper.
- `tests/test_bridge_dcc.py`: dependency-light matrix and wiring checks.

## Torch launch

```bash
cd /scratch/yd2247/sgcrl
git pull --ff-only origin section3_done
sbatch DRAFT_bridge_dcc.sh
```

The wrapper selects indices 18–35 and asks `DRAFT.sh` to run the MuJoCo
restore-step preflight. `DRAFT.sh` performs the preflight only **after** it
activates the `contrastive_rl` Conda environment and sources
`set_up/torch_hpc_env.sh`.

## Launcher failure and fix

The initial wrapper directly executed

```bash
python -m contrastive.action_ranking_diagnostics ...
```

before the canonical launcher activated Conda. On Torch this resolved to
`/usr/lib64/python3.9`, which lacked NumPy and failed with
`ModuleNotFoundError: No module named 'numpy'`.

The fix moves the preflight into `DRAFT.sh` after environment setup and gates
it with `ACTION_LANDSCAPE_SELF_TEST=true`. Consequently, the preflight and
training now use the same Python environment and runtime variables.

## Validation

- `bash -n DRAFT.sh DRAFT_bridge_dcc.sh`
- Python compilation for all Bridge-DCC implementation files
- 36-config expansion with indices 18–35 matched as above
- dependency-light Bridge-DCC, action-landscape, and persistent-actor checks
- `git diff --check`

The full MuJoCo restore test must run on Torch because the local Codex runtime
does not contain the project RL/MuJoCo environment.
