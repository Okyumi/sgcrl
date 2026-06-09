# DCC ablation flags + rl_metrics logging fix (sgcrl side)

Date: 2026-06-09

## 1. Why

The user is preparing a main-conference push that needs DCC ablation
support and per-step representation metrics for the decomposed-critic
learner. Companion change adds DCC to BuilderBench
(see `builderbench/doc/2026-06-09_dcc_port.md`).

## 2. Ablation flags added

| Flag                  | Default | Effect |
|-----------------------|---------|--------|
| `--combine_mode`      | `add`   | `add` (z_sa = h_phi(b_shared) + phi_task) or `concat` (z_sa = [h_phi(b_shared); phi_task]). When `concat`, a learnable goal projection is automatically attached so the contrastive score uses matching `2*repr_dim` space. |
| `--goal_encoder_mode` | `shared`| `shared` (single psi, reused across tasks) or `projected` (adds the goal projection regardless of combine mode). |
| `--dyn_aux_weight`    | `1.0`   | μ on `L_dyn`. Set to `0.0` to disable the dynamic loss entirely. Existing flag, mentioned here for completeness. |

These propagate via `continual_config.ContinualConfig` →
`make_decomposed_networks`. They are zero-cost when left at default
(the projector is only created when needed, so checkpoints from
previous runs continue to load).

### 2.1 Implementation note: goal-encoder param bundle

When the projection is active, `psi_params` becomes the dict
`{'psi': ..., 'psi_proj': ...}` instead of the bare psi pytree. This
avoids adding a new field to `DecomposedTrainingState` (and a new
optimiser-state slot threaded through `__init__` / `update_step` /
checkpoints). Optax operates on the dict transparently; only
`decomp_nets.apply_psi` and the rl-metrics shim peek inside.

### 2.2 What was not implemented in sgcrl

The richer goal-encoder modes (`task_specific`, `partial_shared`,
`decomposed`) are exposed only on the BuilderBench port for now. The
reason is the cross-task resume protocol: each new psi variant adds
an optimiser-state field that has to migrate through the
configuration-keyed checkpoint path
(`_ckpt_path(..., dyn_aux_weight=..., phi_task_width=..., phi_task_depth=...)`).
Extending that path is straightforward but breaks bit-identical resume
for in-flight runs, so I deferred. The BuilderBench port uses simple
per-task pickle checkpoints that already carry the optional groups.

## 3. rl_metrics fix for the decomposed critic

### Problem

`run_continual_contrastive.py:902-920` previously short-circuited the
metrics block whenever `learner.q_params is None` (which is the DCC
learner's intentional signal that the critic is not a monolithic
pytree). The result: zero representation metrics during DCC runs,
including `actor/weight_norm`, `critic_sa/feature_rank`, `NRC1`,
`dormant_ratio`, and the actor-auto-reset signal.

### What I checked

1. `ContinualDecomposedLearner.get_variables(['critic'])[0]` returns the
   bundle dict `{b_shared, h_phi, h_dyn, phi_task, psi}`.
2. `rl_metrics.compute_all_metrics` only needs three things from the
   `networks` object: `repr_fn(params, obs, action) -> (sa, g, _)`,
   `critic_hidden_repr_fn` (may be `None`; the function tolerates it),
   and `actor_repr_fn`. The DCC actor architecture is identical to the
   persistent-critic actor architecture, so `networks.actor_repr_fn`
   works as-is.
3. NRC2 on critic hidden features is the only metric that genuinely
   needs the encoder's final projection weight. DCC has two heads
   (`h_phi` and `h_dyn`) on the same hidden representation, so picking
   one is ambiguous. The function silently skips this metric when
   `critic_hidden_repr_fn=None`.

### Fix

In the rl-metrics block, when `learner.q_params is None`, build (and
cache on the learner) a `SimpleNamespace` shim:

```python
shim.repr_fn         = lambda params, o, a, hidden=None: (
    decomp.apply_sa_repr(params['b_shared'], params['h_phi'],
                         params['phi_task'], o, a),
    decomp.apply_psi(params['psi'], o),
    None,
)
shim.critic_hidden_repr_fn = None
shim.actor_repr_fn   = networks.actor_repr_fn
```

and pass `learner.get_variables(['critic'])[0]` as the critic params.
Persistent / CKA / reset runs are bit-identical (the new path is
guarded by `learner.q_params is None`).

### Limitations remaining

- `critic_sa/nrc2` and the hidden-layer dormancy metric are not
  available for DCC. Cheap follow-up: compute NRC2 against
  `h_phi.weight` and log it as `critic_sa/nrc2_phi`.

## 4. Files touched

- `contrastive/decomposed_networks.py` — new args, optional psi
  projector, bundle-dict for goal params.
- `contrastive/continual_config.py` — `combine_mode`,
  `goal_encoder_mode` fields with docstrings.
- `run_continual_contrastive.py` —
  - new `--combine_mode`, `--goal_encoder_mode` flags;
  - plumbed into `ContinualConfig` construction (line ~1356);
  - rl-metrics shim path (line ~919).

No flags or behaviour change when `critic_mode != 'decomposed'`.
