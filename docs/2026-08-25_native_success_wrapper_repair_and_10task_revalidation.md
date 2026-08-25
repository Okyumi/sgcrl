# Native-success wrapper repair and efficient 10-task revalidation

Date: 2026-08-25  
Status: implemented; V5 positive controls and the 100k/task smoke stage must
pass before full-horizon promotion.

## Motivation

The V2 positive controls showed that the official MetaWorld expert succeeds
on Tasks 5 and 8, while the custom wrapper reports zero reward. The V2 audit
then incorrectly treated full three-dimensional mechanism distance as an
independent copy of MetaWorld success. Inspection of the MetaWorld v2.0.0
task code establishes the actual predicates:

$$
\operatorname{success}_5 =
\mathbb{1}\left[|z_{\mathrm{handle}}-z_{\mathrm{target}}|\le 0.02\right],
$$

$$
\operatorname{success}_8 =
\mathbb{1}\left[|x_{\mathrm{window}}-x_{\mathrm{target}}|\le 0.05\right].
$$

The custom wrappers instead used full-vector Euclidean distance. They also
discarded the reward and `info` returned by the parent MetaWorld `step`. The
primary repair is therefore to use MetaWorld's own success result, rather than
maintaining a second hand-written approximation in the continual wrapper.

## Versioned reward semantics

Every custom Sawyer environment now supports two explicit modes:

- `legacy_distance`: preserves the historical hand-written predicates and is
  the default. Existing paper configurations and checkpoints therefore do not
  change silently.
- `native_info`: executes the parent transition once and defines the sparse
  reward as

$$
r_t^{\mathrm{native}}=
\mathbb{1}\left[\mathrm{info}_t[\text{success}]=1\right].
$$

The corrected transition retains MetaWorld's shaped reward as
`info["native_reward"]`, retains all other native diagnostic fields, and keeps
`done=False` because the existing outer `StepLimitWrapper` remains responsible
for episode boundaries. The native mode adds neither a second simulator step
nor a second `evaluate_state` call.

The adapter accepts both standard step signatures used by supported
installations:

```text
(observation, reward, done, info)
(observation, reward, terminated, truncated, info)
```

It normalizes either form to the four-item API expected by the existing Acme
stack and records `native_terminated`, `native_truncated`, and
`native_step_api` in `info`. The warning that legacy Gym is unmaintained is not
the runtime failure and does not justify a one-line import replacement here:
the current Acme and wrapper stack still consumes Gym's four-item interface.

The pinned MetaWorld fork may also advance the simulator while returning
`None`, because its task subclasses historically recomputed their own reward.
For that case the adapter calls the inherited MetaWorld `evaluate_state` on the
already-advanced observation and executed action. This recomputes only reward
and success; it does **not** take another simulator step. A direct
`(reward, info)` return is supported as well, and `native_step_api` records
which path was used.

The V3 controls exposed a subtle bug in that fallback. It passed the custom
22-dimensional goal-conditioned observation to MetaWorld's
`evaluate_state`, although MetaWorld expects its own raw observation. That
made the physical replay and exact success-axis proxy succeed while the
wrapper reward remained zero. V4 explicitly calls the first MetaWorld parent
class's `_get_obs` and `evaluate_state`, so method overriding cannot substitute
the wrapper observation at this boundary.

V4 then isolated a second compatibility defect. MetaWorld sets `isV2` using
the runtime-class-name test `"V2" in type(self).__name__`. The local classes
are genuine V2 subclasses but are named `SawyerHandlePressSide`,
`SawyerWindowClose`, and so on, so MetaWorld marks them as V1. The V4 reward,
native-info, and exact-axis signals agree, and exact native-action replay has
zero trajectory error on both tasks; however, the regenerated expert-policy
trajectory diverges because the parent observation is laid out as V1. V5
constructs the native 39-dimensional V2 observation under temporary V2
bookkeeping while preserving the historical wrapper transition path. It also
re-evaluates any nominal four- or five-item parent result when the local class
is detected as a misclassified V2 wrapper, preventing a different MetaWorld
fork from silently computing success from the custom 22-dimensional layout.

The three-seed V4 summary gives the diagnostic signature directly:

| Task | Native-action replay R/I/A | Regenerated policy R/I/A | Policy trajectory error | Fallback fraction |
|---|---:|---:|---:|---:|
| Task 5 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.170 | 1.000 |
| Task 8 | 1.000/1.000/1.000 | 0.000/0.000/1.000 | 0.204 | 1.000 |

Here `R/I/A` means sparse reward, MetaWorld `info["success"]`, and the
success-axis proxy. Task 8's apparent policy-axis success came from including
the reset state although `info` exists only after a step; V5 excludes the
reset state from all success-proxy gates.

Acme's interface is already used at the correct boundary:
`contrastive/utils.py` wraps the raw Gym/MetaWorld environment with Acme's
`GymWrapper`, then applies `StepLimitWrapper`. Moving the custom MuJoCo
subclasses themselves to `dm_env.Environment` would require rewriting reset,
step, spaces/specs, scripted-policy access, and audit state restoration. It
would not remove the need to adapt the installed MetaWorld parent API, so the
small raw-environment compatibility layer is the safer fix.

The flag `--sawyer_success_mode` is propagated through training, evaluation,
action-landscape environments, counterfactual environments, W&B configs, and
checkpoint paths. Corrected checkpoints receive the suffix
`_success_native_info`; legacy paths remain byte-for-byte unchanged.

## Exact goal-success audit

`contrastive/goal_semantics.py` now records both the three-coordinate mechanism
block and the exact official success coordinate:

| Task | Mechanism state block | Official success coordinate | Threshold |
|---|---|---|---:|
| Task 5: handle press side | `state[4:7]` | `state[6]` (handle z) | 0.02 |
| Task 8: window close | `state[4:7]` | `state[4]` (window x) | 0.05 |

The earlier `success_mechanism` mode remains available as a historical 3-D
ablation. A new `native_success_axis` mode is available for single-task
representation experiments. It is deliberately not used in the 10-task paper
revalidation because its semantic index differs across tasks; those runs keep
the original `full_state` input architecture and change only reward semantics.

The positive-control audit is now version 5. It logs:

- authoritative native `info["success"]`;
- corrected wrapper positive reward;
- exact native-axis success and distance;
- full 3-D distance, labeled descriptive rather than authoritative;
- native/fixed target distance on the official success axis;
- paired reset, random-vector, and replay-trajectory errors.
- per-step disagreement between MetaWorld `info["success"]` and the exact
  success-axis proxy;
- per-step disagreement between wrapper positive reward and native success;
- whether the `evaluate_state` compatibility fallback was exercised;
- policy-trajectory error as well as exact-action replay error.
- the maximum difference in the scripted policy's seven relevant input
  coordinates and resulting action, so policy-adapter failures cannot be
  mislabeled as dynamics failures.

The evaluator refuses V2, V3, and V4 files because none can authorize promotion:
V2 used a full-3D proxy and V3 evaluated the wrong observation in the
non-standard parent-step fallback; V4 still constructed the parent observation
under the erroneous V1 identity. It distinguishes `audit_metric_inconsistent`,
`native_positive_control_failed`, `custom_wrapper_invalid`,
`fixed_global_target_misaligned`, `fixed_target_valid`, and `inconclusive`.

The submitted V3 results nevertheless contain useful physical evidence. On
all three seeds, replay trajectories match the native environment exactly.
The historical fixed target differs from the native target by only about
$0.00495$ on Task 5's success coordinate (threshold $0.02$) and by $0$ on
Task 8's success coordinate (threshold $0.05$). Thus V3 does **not** confirm
that the fixed target is physically invalid; it identifies an audit/reward
adapter inconsistency. V4 subsequently verified reward consistency and exact
replay, while isolating the remaining policy-observation mismatch. V5 is the
first audit version that can promote the wrapper.

## Experiment sequence

### V0: corrected positive controls

Run three CPU audit seeds:

```bash
sbatch DRAFT_goal_wrapper_positive_controls.sh
```

Then aggregate them:

```bash
python scripts/evaluate_goal_wrapper_positive_controls.py \
  logs/goal_validity/positive_controls_v5_seed{5,6,7}.json \
  --strict-promotion
```

Promotion requires `fixed_target_valid` for both Tasks 5 and 8 on all three
seeds, with native `info` and the exact-axis proxy consistent.

### V1: cheap 10-task runtime and semantics smoke

```bash
sbatch DRAFT_native_success_wrapper_smoke.sh
```

This is a paired 3-method by 3-seed matrix:

| Method | Actor | Critic | Seeds |
|---|---|---|---|
| DCC | reset | decomposed | 5, 6, 7 |
| Reset/reset | reset | reset | 5, 6, 7 |
| Persistent/persistent | persistent | persistent | 5, 6, 7 |

Each run executes 100,000 steps on each of Tasks 0–9: one million environment
steps total. Evaluation occurs only once per task, on the current task, with
five episodes. Counterfactual probes, oracle rollouts, action-landscape probes,
shortcut diagnostics, representation sweeps, pool-cosine dumps, and probe-data
dumps are all disabled. Runtime profiling remains enabled.

Three CPU-actor/JAX-learner processes share each L40S. This addresses the low
single-run GPU utilization without adding simulator work. Based on the prior
streamlined throughput of roughly 245–256 environment steps/s per run, the
environment-step floor is about 65–70 minutes per smoke run before task/W&B
overhead. The three-array layout is intended to keep total wall time within a
few hours rather than running nine cells serially.

V1 passes only if all cells:

1. report `sawyer_success_mode=native_info` on all ten tasks;
2. complete one million total steps without checkpoint collisions;
3. log finite success and runtime metrics on every task;
4. spend negligible time in disabled diagnostics;
5. avoid a severe per-process throughput regression relative to the previous
   diagnostic-free baseline.

V1 is a validity and performance gate, not a learning conclusion: 100k steps
per task is intentionally too short to reproduce the paper's final AUC.

### V2: guarded full-horizon paper revalidation

Only after V0 and V1 pass:

```bash
NATIVE_SUCCESS_WRAPPER_PROMOTED=true \
  sbatch DRAFT_native_success_wrapper_promotion.sh
```

The promotion matrix uses the same methods and paired seeds at 8M steps per
task. Evaluation is every 200k steps instead of every 50k, reducing evaluation
frequency by 4x while retaining 40 points per task. Heavy diagnostics remain
disabled and three processes share each underutilized GPU. The full sequence
is still 80M steps per run and cannot be made intrinsically short; the launcher
therefore uses separate task-boundary checkpoints and supports safe resubmission
without mixing corrected and legacy rewards.

## Code and configuration changes

- `env_utils.py`: versioned success mode for all 13 custom Sawyer wrappers.
- `contrastive/sawyer_success.py`: dependency-light native-success adapter.
- `contrastive/utils.py`: environment factory propagation.
- `run_continual_contrastive.py`: flag, all environment sites, W&B manifest,
  console manifest, checkpoint identity, and checkpoint metadata.
- `contrastive/goal_semantics.py`: official success-axis metadata and helpers.
- `scripts/audit_sawyer_goal_positive_controls.py`: V5 authoritative metrics.
- `scripts/evaluate_goal_wrapper_positive_controls.py`: V5-only aggregation
  and corrected classifications.
- `experiment_configs_native_success_wrapper.py`: 100k/task smoke matrix.
- `experiment_configs_native_success_wrapper_promotion.py`: guarded 8M/task
  matrix.
- `DRAFT_native_success_wrapper_smoke.sh` and
  `DRAFT_native_success_wrapper_promotion.sh`: Torch launchers.
- Dependency-light tests cover the adapter, all wrapper insertion points,
  exact task axes, audit classifications, runtime propagation, checkpoint
  separation, config matrix, promotion guard, and launcher resources.

## Interpretation and limitations

This repair can establish whether Tasks 5 and 8 failed partly because the
continual wrapper supplied the wrong sparse reward. V3 does not yet establish
that claim, because its fallback computed MetaWorld success from the wrong
observation. The V5 positive controls are the promotion boundary. This work
does not retroactively
invalidate comparisons made entirely under the legacy wrapper: those runs
still compare algorithms under the same historical contract. However, if the
corrected full-horizon ranking changes materially, benchmark-faithful claims
should use the guarded full-horizon revalidation and describe the legacy result as a controlled
custom-goal benchmark rather than native MetaWorld success.

MetaWorld's native success is authoritative for reward, while the exposed
full-state goal remains a representation choice. A future unified goal redesign
would need task masks or a common semantic schema before using exact scalar
success axes throughout a continual 10-task critic.
