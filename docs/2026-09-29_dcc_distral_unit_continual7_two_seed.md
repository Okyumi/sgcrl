# From-scratch seven-task Distral alpha pilot

Date: 2026-09-29
Status: launcher and tests are in place. Submit from a Torch login node.

## Motivation

The Task-6 branch-from-checkpoint pilot asks how strongly an already
learned shared representation should be used on one new task. This run
asks a different question: if alpha is held fixed from random
initialization, does it change the shared representation that is learned
and how later tasks use it?

The pilot is human Tasks 1–7 (code IDs 0–6), one million steps each.
Alphas are `0, 0.25, 0.5, 0.75, 1, 1.5`. Seeds are 5 and 6 only, twelve
curricula instead of the eighteen-run three-seed grid. Seed 7 stays
available through `ALPHA_SEEDS=5,6,7`. Nine- and ten-task follow-ups stay
behind `ALPHA_NUM_TASKS` and are not part of this submit.

Plain DCC through tasks 0–9 is not rerun. Those runs are an external
reference, not a matched normalization control: this pilot uses
`unit_distral`, one million steps, and `native_info`. A large gap between
`unit_distral` at alpha 1 and those older runs is not, by itself, evidence
about prior strength.

## Mathematical objective

`unit_distral` scores a fixed critic as

```text
f_alpha = alpha * <unit(z_shared), unit(z_goal)>
          +       <unit(z_task),   unit(z_goal)>
```

with `unit(x) = x / max(||x||_2, epsilon)` per sample. The score is not
divided by `alpha + 1`. For a fixed critic and an unrestricted
maximum-entropy actor with temperature `tau`,

```text
pi*(a|s,g) proportional to
    pi_0(a|s,g)^alpha * exp(f_task_hat(s,a,g) / tau),
```

where `pi_0` is the softmax policy of the normalized shared score. Larger
alpha gives that shared policy more influence. Alpha 1.5 is outside the
usual Distral range `0 <= alpha <= 1` and is kept as a stronger prior.

This is not an identity between DCC and Distral. The critic and the actor
train together, and the shared object is a contrastive representation with
a dynamics auxiliary, not Distral's distilled policy. The logged
manipulation check is whether the effective shared-to-task norm ratio
tracks alpha. The behavioral question is whether that ratio changes
learning across matched seeds.

## Code and configuration

No change to the score. `shared_repr_normalization` still defaults to
`none`. New files:

- `experiment_configs_dcc_distral_unit_shared_scale_continual.py`
- `DRAFT_dcc_distral_unit_shared_scale_continual.sh`
- `tests/test_dcc_distral_unit_shared_scale_continual.py`

Each alpha/seed pair has its own run, log, and checkpoint directory under

```text
/scratch/yd2247/sgcrl/logs/dcc_distral_unit_continual/
  continual7_alpha{tag}_seed{S}/
```

The launcher does not pass a prefix checkpoint. The runner's default
`start_task=0` still auto-resumes from that run's own completed
`task_k.pkl`, which is what the 48-hour continuation hop uses.

Fixed plain-DCC settings: reset actor, decomposed critic, width 1024,
depth 4, `dyn_aux_weight=1`, `phi_task` 256×4, additive inner-product
score, eval every 50k for 10 episodes, `native_info`, full-state goals,
no task id, current-task post-eval, and no behavior cloning, action-effect
head, counterfactual probes, extra negatives, or interaction-weighted
relabeling.

W&B project `continual-contrastive-rl`, group
`DCC-DISTRAL-UNIT-ALPHA-CONTINUAL7-1M`.

## Launch command

On a Torch login node, from `/scratch/yd2247/sgcrl` after this revision
is checked out:

```bash
python tests/test_dcc_distral_unit_shared_scale.py
python tests/test_dcc_distral_unit_shared_scale_continual.py
mkdir -p /scratch/yd2247/sgcrl/logs/dcc_distral_unit_continual
sbatch DRAFT_dcc_distral_unit_shared_scale_continual.sh
```

The array is `0-5`: one L40S per alpha on `l40s_public` / QOS `gpu48`,
with seeds 5 and 6 sharing that GPU. One `afterany` hop is scheduled and
cancelled if both seeds have `task_6.pkl`. Six running jobs plus six
pending hops stay under the 16-GPU QOS cap.

## Logged metrics

The learner already records the manipulation check. Use these keys rather
than new aliases:

- `decomp/shared_scale`
- `decomp/shared_coefficient` (equals alpha under `unit_distral`)
- `decomp/task_coefficient` (equals 1)
- `decomp/shared_norm`, `decomp/task_norm` (raw branch norms)
- `decomp/scaled_shared_norm`, `decomp/effective_task_norm`
- `decomp/scaled_shared_to_task_norm` (must track alpha)
- `decomp/shared_goal_score_abs`, `decomp/task_goal_score_abs`
- `decomp/shared_score_fraction`
- `decomp/shared_task_cosine`

Also keep critic loss and accuracy, `decomp/L_dyn`, actor loss, and
`entropy_mean`, plus `evaluator/success_rate` every 50k steps. Per-task
summaries (AUC, success at 250k/500k/750k/1M, best success, mean of the
last three evals, and time to 50/80/90 percent) can be computed from that
curve after the runs finish. Compare seeds 5 and 6 separately as well as
together.

## Known limitations

- Two seeds can show a lucky seed, but they are not the three-seed grid.
- Tasks 8–10 are not in this submit.
- The existing plain-DCC task 0–9 curves are not an alpha-1
  `normalization=none` twin of this protocol.
- A wall-clock kill mid-task does not resume inside that task. The next
  hop restarts the unfinished task from its beginning.
- Alpha still does not prove the Distral derivation once the critic is
  learned. The norm-ratio check has to pass before a performance
  difference is read as prior strength.
