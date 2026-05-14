# C1 task-1 crashes + checkpoint-collision risk

Date: 2026-05-14

User hypothesis: the C1 task-1 runs crashed because the checkpoint
folder name does not include `dyn_aux_weight`, so a `dyn_aux_weight=0`
C1 task-1 ran loaded a checkpoint that had been saved by a different
`dyn_aux_weight` run.

**Verdict: structurally correct (checkpoint paths are indeed missing
`dyn_aux_weight` and other fields), but not what caused these
specific crashes. The C1 task-1 crashes were a startup-time
infrastructure failure, distinct from the checkpoint-collision
issue. The collision issue is a real bug that will affect the C3
(N7) ablation grid if not fixed before launch.**

---

## What `_ckpt_path` keys on today

In `run_continual_contrastive.py:235-245`:

```python
def _ckpt_path(ckpt_dir, task_id, seed, critic_mode='persistent',
               use_task_id=True, adapt_heads_only=True, actor_mode='cka'):
  config_key = (f'actor_{actor_mode}_critic_{critic_mode}'
                f'_tid_{use_task_id}_heads_{adapt_heads_only}')
  return os.path.join(ckpt_dir, config_key, f'seed_{seed}',
                      f'task_{task_id}.pkl')
```

The directory key contains: `actor_mode`, `critic_mode`,
`use_task_id`, `adapt_heads_only`, `seed`. It does **not** contain:

- `dyn_aux_weight` (the C3 / N7 ablation variable)
- `phi_task_width`, `phi_task_depth` (network shape)
- `network_width`, `critic_depth`, `actor_depth` (architecture)
- `k_max`, `steps_per_task` (training schedule)

Any two cells that share the actor/critic mode and the same seed
will write to the same `task_<k>.pkl` file. Specifically:

- C1 (`dyn_aux_weight=0`) and C2 (`dyn_aux_weight=1`) both at
  `actor=reset, critic=decomposed, seed=5` → same path.
- The C3 grid cells G2 (w=0), G3 (w=0.1), G4 (w=1.0) all at
  `actor=reset, critic=decomposed, seed=5` → same path.

When run sequentially in the same job array, **later cells
overwrite earlier cells' checkpoints silently.**

## Timeline of the C1 seed=5 case (from W&B)

| Time (UTC)          | Group | Run                                      | State    | Critic    | Notes                                                                                    |
|---------------------|-------|------------------------------------------|----------|-----------|------------------------------------------------------------------------------------------|
| 2026-05-10 13:38:45 | C1    | task0_sawyer_hammer_s5 (`kasyumio`)      | finished | decomposed| writes `…_decomposed_…/seed_5/task_0.pkl` (C1's dyn_aux=0 ckpt)                            |
| 2026-05-10 13:38:46 | C1    | task0_sawyer_hammer_s5 (`yya4w369`)      | finished | persistent| writes `…_persistent_…/seed_5/task_0.pkl`                                                 |
| **2026-05-10 13:47:06** | **C2**| **task0_sawyer_hammer_s5 (`2t9lwiy8`)** | finished | decomposed| **OVERWRITES `…_decomposed_…/seed_5/task_0.pkl` with the C2 dyn_aux=1 ckpt**             |
| 2026-05-10 17:38:21 | C1    | task1_sawyer_push_wall_s5 (`18aajcpb`)   | CRASHED  | persistent| loads `…_persistent_…/seed_5/task_0.pkl` (its own); zero history rows; no metadata synced |
| 2026-05-10 19:50:06 | C1    | task1_sawyer_push_wall_s5 (`zvdgfgu6`)   | CRASHED  | decomposed| loads `…_decomposed_…/seed_5/task_0.pkl` (now C2's data); zero history rows               |
| 2026-05-10 19:57:18 | C2    | task1_sawyer_push_wall_s5 (`allszom8`)   | finished | decomposed| loads same path; works fine                                                              |

## Why this isn't the cause of the crashes (but it's still a bug)

If the collision were the cause:

- The persistent C1 task-1 (`18aajcpb`) should NOT have crashed,
  because the persistent path is entirely separate from the
  decomposed path. It loads from `…_persistent_…/seed_5/task_0.pkl`,
  which was written by C1 only and never overwritten by C2.
- But it did crash, with the same signature as the decomposed
  crash (created_at == heartbeat_at, zero history rows, no
  wandb metadata visible).
- And then the C2 decomposed task-1 (`allszom8`), launched only
  7 minutes later from the same git commit (`dbc8935`) on the same
  host (`gl017.hpc.nyu.edu`), loaded the same `…_decomposed_…/seed_5/task_0.pkl`
  and ran fine.

So the cause cannot be checkpoint contents. The most likely
explanation is the same as for the C2 task-8 crashes: a transient
infrastructure issue (cluster scheduler problem, node startup
failure, file-system unmount). Both modes crashed at the same time
because the same SLURM array submitted them on the same node and
hit the same transient.

The fact that the C2 task-1 worked 7 minutes later supports this:
the same code, same config (modulo `dyn_aux_weight`), same host,
no source-code changes — it succeeded on the second try.

## The real bug worth fixing: ckpt path needs more keys

Even though it didn't cause this crash, the missing keys are
guaranteed to cause silent failures when running the C3 grid:

- G1 (persistent) and the C0 / G0 reset-persistent cells will
  collide on the same path if they share seeds.
- G2 (decomposed, dyn_aux=0), G3 (dyn_aux=0.1), G4 (dyn_aux=1.0)
  all share `actor=reset, critic=decomposed, seed=…` and will
  overwrite each other in any concurrent or sequential job
  array.

**Proposed fix (small):** extend `config_key` to include
`dyn_aux_weight` (when `critic_mode == 'decomposed'`) and
`phi_task_width` / `phi_task_depth` (likewise). Keep the format
backward-compatible by only adding fields when they differ from
defaults, OR by always emitting them but with sensible
filename-safe encoding.

The simplest robust change is to always include the relevant
fields, gated on `critic_mode`:

```python
def _ckpt_path(ckpt_dir, task_id, seed, critic_mode='persistent',
               use_task_id=True, adapt_heads_only=True, actor_mode='cka',
               dyn_aux_weight=1.0, phi_task_width=256, phi_task_depth=4):
  config_key = (f'actor_{actor_mode}_critic_{critic_mode}'
                f'_tid_{use_task_id}_heads_{adapt_heads_only}')
  if critic_mode == 'decomposed':
    config_key += (f'_dyn{dyn_aux_weight:.3f}'
                   f'_pt{phi_task_width}x{phi_task_depth}')
  return os.path.join(ckpt_dir, config_key, f'seed_{seed}',
                      f'task_{task_id}.pkl')
```

This changes the directory name for decomposed runs but keeps the
persistent path unchanged. Backward compatibility: existing C2
checkpoints would not be auto-found because the path key changes;
the user would need to either move them or run from task 0. This is
a one-time cost worth paying before C3 launches.

I will not make this change yet — proposing it here for review
because it changes the on-disk filenames the user has been working
with. If you approve I'll ship it in a follow-up commit along
with a path-migration note.

## What about the actual crash?

We cannot diagnose the actual C1 task-1 crash without the SLURM
stderr / cluster output log. The recipe from
`docs/2026-05-13_wandb_findings.md` §4.1 still applies:

```bash
find /scratch/yd2247/sgcrl/logs/continual/continual_contrastive_cpc/ \
    -name '*task1_sawyer_push_wall_s5*' -type d 2>/dev/null
grep -nE 'Traceback|Error|OOM|CUDA' /scratch/yd2247/sgcrl/logs/.../output.log
```

If the traceback shows pickle-deserialization error, then the
checkpoint-collision is in fact involved. If it shows
`CUDA_ERROR_OUT_OF_MEMORY`, `socket timed out`, or `Connection
refused`, then it's infrastructure.

Given the persistent C1 task-1 crashed with no collision possible,
the dominant prior is infrastructure. But if logs show otherwise
we can refine.

## Files produced

This doc only.

Related artifacts referenced:
- `docs/2026-05-13_wandb_findings.md` §4.1 (original crash diagnosis).
- W&B runs queried: `zvdgfgu6` (C1 decomp crashed), `18aajcpb` (C1
  persist crashed), `allszom8` (C2 decomp finished).
