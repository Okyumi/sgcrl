#!/usr/bin/env python3
"""Enumerate all experiment configurations for the continual CRL grid.

Two sources are concatenated to produce the final list of runs:

1. **Cartesian grid** (`ACTOR_MODES` × `CRITIC_MODES` × `SEEDS`) — the
   classic 9-cell ablation table. All non-grid fields fall back to
   `EXTRA_OVERRIDES` and the `run_continual_contrastive.py` flag
   defaults.
2. **Explicit CELLS list** — one dict per run, used for diagnostic and
   decomposed-critic cells from the runbook
   (`docs/2026-05-08_runbook_what_to_launch_next.md`). Each dict can
   set any subset of:
       actor_mode, critic_mode, seed,
       dyn_aux_weight, phi_task_width, phi_task_depth,
       combine_mode, goal_encoder_mode,
       bellman_loss_weight, bellman_residual_l2_weight,
       bellman_discount, bellman_tau, bellman_hidden_dim,
       her_reward_threshold, step_penalty_reward,
       log_pool_cosine, log_mixture_norm, log_probe_data,
       (and any other future field that submit scripts know how to
       forward).

Cell dicts win over EXTRA_OVERRIDES; EXTRA_OVERRIDES wins over
script-level shell defaults.

Usage:
  # Total number of configs (used by submit scripts to size the SLURM array):
  python experiment_configs.py --total

  # Print a specific configuration as KEY=VALUE pairs (sourced by the
  # submit scripts via `eval`):
  python experiment_configs.py --setting 0

  # Print all configs as a human-readable table:
  python experiment_configs.py --list
"""
import argparse
import itertools
import shlex
import sys


# =====================================================================
# Cartesian grid — edit ACTOR_MODES / CRITIC_MODES / SEEDS to change
# the 9-cell ablation sweep.
# =====================================================================

ACTOR_MODES = []
CRITIC_MODES = []
SEEDS = []

# Overrides applied to every Cartesian-grid cell. The CELLS list below
# can additionally override these on a per-cell basis.
# Example: {'steps_per_task': 4_000_000, 'k_max': 5}
EXTRA_OVERRIDES: dict = {}


# =====================================================================
# Explicit cells — one dict per run.
#
# Each dict's keys correspond to env-var names that the submit scripts
# (draft_3.sh / draft_4.sh / DRAFT.sh / submit_continual_torch.sh)
# already understand. Use SNAKE_lowercase here; submit scripts
# uppercase them.
#
# Empty by default. Uncomment / append rows for the runbook cells:
#   - C0 (D7 CKA-failure diagnostic): actor_mode=cka, critic_mode=cka,
#     log_mixture_norm=True, log_pool_cosine=True, three seeds.
#   - C1 (N5 regression check): actor_mode=reset, critic_mode=decomposed,
#     dyn_aux_weight=0.0, log_probe_data=True, three seeds; plus the
#     persistent baseline at the same seeds.
#   - C2 (N6 sanity): actor_mode=reset, critic_mode=decomposed,
#     dyn_aux_weight=1.0, log_probe_data=True, three seeds.
#   - C3 (N7 ablation grid): five cells x five seeds (G1..G5; G5
#     deferred until the 'decomposed body + reset carry' plumbing
#     lands).
# =====================================================================

CELLS: list = [
    # ---- C0: CKA-failure diagnostic (D7) ----------------------------
    # {'actor_mode': 'cka', 'critic_mode': 'cka', 'seed': 5,
    #  'log_mixture_norm': True, 'log_pool_cosine': True},
    # {'actor_mode': 'cka', 'critic_mode': 'cka', 'seed': 6,
    #  'log_mixture_norm': True, 'log_pool_cosine': True},
    # {'actor_mode': 'cka', 'critic_mode': 'cka', 'seed': 7,
    #  'log_mixture_norm': True, 'log_pool_cosine': True},

    # ---- C1: decomposed regression check, dyn_aux_weight=0 (N5) -----
    # Resume unfinished seeds 5/6/7 (prior runs stopped around task 8).
    # {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 5,
    #  'dyn_aux_weight': 0.0, 'log_probe_data': True,
    #  'wandb_group': 'C1: decomposed regression check and baseline'},
    # {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 6,
    #  'dyn_aux_weight': 0.0, 'log_probe_data': True,
    #  'wandb_group': 'C1: decomposed regression check and baseline'},
    # {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 7,
    #  'dyn_aux_weight': 0.0, 'log_probe_data': True,
    #  'wandb_group': 'C1: decomposed regression check and baseline'},
    # ---- C1 baseline: persistent at the same seeds ------------------
    # {'actor_mode': 'reset', 'critic_mode': 'persistent', 'seed': 5,
    #  'log_probe_data': True},
    # {'actor_mode': 'reset', 'critic_mode': 'persistent', 'seed': 6,
    #  'log_probe_data': True},
    # {'actor_mode': 'reset', 'critic_mode': 'persistent', 'seed': 7,
    #  'log_probe_data': True},

    # ---- C2: decomposed single-cell sanity, dyn_aux_weight=1 (N6) --
    # {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 97,
    #  'dyn_aux_weight': 1.0, 'log_probe_data': True,
    #  'wandb_group': 'C2: decomposed single-cell sanity'},
    # {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 98,
    #  'dyn_aux_weight': 1.0, 'log_probe_data': True,
    #  'wandb_group': 'C2: decomposed single-cell sanity'},
    # {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 99,
    #  'dyn_aux_weight': 1.0, 'log_probe_data': True,
    #  'wandb_group': 'C2: decomposed single-cell sanity'},
    # {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 100,
    #  'dyn_aux_weight': 1.0, 'log_probe_data': True,
    #  'wandb_group': 'C2: decomposed single-cell sanity'},
    # {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 101,
    #  'dyn_aux_weight': 1.0, 'log_probe_data': True,
    #  'wandb_group': 'C2: decomposed single-cell sanity'},

    # ---- C2b: dyn-aux only at task 0; off afterward -----------------
    # Tests whether the dynamics auxiliary is doing real work during
    # tasks 1..9 or just acting as a task-0 initialiser for b_shared.
    # See docs/2026-05-14_c2_ldyn_interpretation.md for the rationale.
    # The new --dyn_aux_after_task0=0.0 flag overrides dyn_aux_weight
    # starting at task 1. If C2b matches C2, the aux is purely a
    # task-0 init and we can simplify the algorithm description in
    # the paper. If C2b is worse, the aux is providing a continual
    # constraint we underestimated.
    # {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 5,
    #  'dyn_aux_weight': 1.0, 'dyn_aux_after_task0': 0.0,
    #  'log_probe_data': True},
    # {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 6,
    #  'dyn_aux_weight': 1.0, 'dyn_aux_after_task0': 0.0,
    #  'log_probe_data': True},
    # {'actor_mode': 'reset', 'critic_mode': 'decomposed', 'seed': 7,
    #  'dyn_aux_weight': 1.0, 'dyn_aux_after_task0': 0.0,
    #  'log_probe_data': True},

    # ---- C3: full ablation grid (N7) --------------------------------
    # G1 baseline (actor_mode=reset, critic_mode=persistent), 5 seeds
    # G2 dyn-aux off  (dyn_aux_weight=0.0, decomposed), 5 seeds
    # G3 dyn-aux weak (dyn_aux_weight=0.1, decomposed), 5 seeds
    # G4 dyn-aux full (dyn_aux_weight=1.0, decomposed), 5 seeds
    # G5 (decomposed body + reset carry) — deferred (N7b plumbing)

    # ---- RBC-DCC: 10-task continual comparison vs DCC (seeds 5,6,7) ----
    {'actor_mode': 'reset', 'critic_mode': 'rbc_decomposed', 'seed': 5,
     'dyn_aux_weight': 1.0, 'combine_mode': 'add',
     'goal_encoder_mode': 'shared', 'bellman_loss_weight': 1.0,
     'bellman_residual_l2_weight': 0.0001, 'bellman_discount': 0.99,
     'bellman_tau': 0.005, 'bellman_hidden_dim': 256,
     'her_reward_threshold': 0.05, 'step_penalty_reward': True,
     'log_probe_data': True,
     'wandb_group': 'RBC-DCC continual 10-task'},
    {'actor_mode': 'reset', 'critic_mode': 'rbc_decomposed', 'seed': 6,
     'dyn_aux_weight': 1.0, 'combine_mode': 'add',
     'goal_encoder_mode': 'shared', 'bellman_loss_weight': 1.0,
     'bellman_residual_l2_weight': 0.0001, 'bellman_discount': 0.99,
     'bellman_tau': 0.005, 'bellman_hidden_dim': 256,
     'her_reward_threshold': 0.05, 'step_penalty_reward': True,
     'log_probe_data': True,
     'wandb_group': 'RBC-DCC continual 10-task'},
    {'actor_mode': 'reset', 'critic_mode': 'rbc_decomposed', 'seed': 7,
     'dyn_aux_weight': 1.0, 'combine_mode': 'add',
     'goal_encoder_mode': 'shared', 'bellman_loss_weight': 1.0,
     'bellman_residual_l2_weight': 0.0001, 'bellman_discount': 0.99,
     'bellman_tau': 0.005, 'bellman_hidden_dim': 256,
     'her_reward_threshold': 0.05, 'step_penalty_reward': True,
     'log_probe_data': True,
     'wandb_group': 'RBC-DCC continual 10-task'},

    # ---- RBC-DCC: single-task task-5 probe (handle-press-side) ----------
    # {'actor_mode': 'reset', 'critic_mode': 'rbc_decomposed', 'seed': 5,
    #  'single_task': 'sawyer_handle_press_side',
    #  'dyn_aux_weight': 1.0, 'combine_mode': 'add',
    #  'goal_encoder_mode': 'shared', 'bellman_loss_weight': 1.0,
    #  'bellman_residual_l2_weight': 0.0001, 'bellman_discount': 0.99,
    #  'bellman_tau': 0.005, 'bellman_hidden_dim': 256,
    #  'her_reward_threshold': 0.05, 'step_penalty_reward': True,
    #  'log_probe_data': True,
    #  'wandb_group': 'RBC-DCC single-task k5'},
    # {'actor_mode': 'reset', 'critic_mode': 'rbc_decomposed', 'seed': 6,
    #  'single_task': 'sawyer_handle_press_side',
    #  'dyn_aux_weight': 1.0, 'combine_mode': 'add',
    #  'goal_encoder_mode': 'shared', 'bellman_loss_weight': 1.0,
    #  'bellman_residual_l2_weight': 0.0001, 'bellman_discount': 0.99,
    #  'bellman_tau': 0.005, 'bellman_hidden_dim': 256,
    #  'her_reward_threshold': 0.05, 'step_penalty_reward': True,
    #  'log_probe_data': True,
    #  'wandb_group': 'RBC-DCC single-task k5'},
    # {'actor_mode': 'reset', 'critic_mode': 'rbc_decomposed', 'seed': 7,
    #  'single_task': 'sawyer_handle_press_side',
    #  'dyn_aux_weight': 1.0, 'combine_mode': 'add',
    #  'goal_encoder_mode': 'shared', 'bellman_loss_weight': 1.0,
    #  'bellman_residual_l2_weight': 0.0001, 'bellman_discount': 0.99,
    #  'bellman_tau': 0.005, 'bellman_hidden_dim': 256,
    #  'her_reward_threshold': 0.05, 'step_penalty_reward': True,
    #  'log_probe_data': True,
    #  'wandb_group': 'RBC-DCC single-task k5'},

    # ---- RBC-DCC: single-task task-8 probe (window-close) ---------------
    # {'actor_mode': 'reset', 'critic_mode': 'rbc_decomposed', 'seed': 5,
    #  'single_task': 'sawyer_window_close',
    #  'dyn_aux_weight': 1.0, 'combine_mode': 'add',
    #  'goal_encoder_mode': 'shared', 'bellman_loss_weight': 1.0,
    #  'bellman_residual_l2_weight': 0.0001, 'bellman_discount': 0.99,
    #  'bellman_tau': 0.005, 'bellman_hidden_dim': 256,
    #  'her_reward_threshold': 0.05, 'step_penalty_reward': True,
    #  'log_probe_data': True,
    #  'wandb_group': 'RBC-DCC single-task k8'},
    # {'actor_mode': 'reset', 'critic_mode': 'rbc_decomposed', 'seed': 6,
    #  'single_task': 'sawyer_window_close',
    #  'dyn_aux_weight': 1.0, 'combine_mode': 'add',
    #  'goal_encoder_mode': 'shared', 'bellman_loss_weight': 1.0,
    #  'bellman_residual_l2_weight': 0.0001, 'bellman_discount': 0.99,
    #  'bellman_tau': 0.005, 'bellman_hidden_dim': 256,
    #  'her_reward_threshold': 0.05, 'step_penalty_reward': True,
    #  'log_probe_data': True,
    #  'wandb_group': 'RBC-DCC single-task k8'},
    # {'actor_mode': 'reset', 'critic_mode': 'rbc_decomposed', 'seed': 7,
    #  'single_task': 'sawyer_window_close',
    #  'dyn_aux_weight': 1.0, 'combine_mode': 'add',
    #  'goal_encoder_mode': 'shared', 'bellman_loss_weight': 1.0,
    #  'bellman_residual_l2_weight': 0.0001, 'bellman_discount': 0.99,
    #  'bellman_tau': 0.005, 'bellman_hidden_dim': 256,
    #  'her_reward_threshold': 0.05, 'step_penalty_reward': True,
    #  'log_probe_data': True,
    #  'wandb_group': 'RBC-DCC single-task k8'},
]


# =====================================================================
# Required keys
# =====================================================================

# Every emitted config must carry at least these three keys. Submit
# scripts assume their presence.
_REQUIRED_KEYS = ('actor_mode', 'critic_mode', 'seed')


# =====================================================================
# Build
# =====================================================================

def build_configs():
  """Return a list of dicts, one per experiment run.

  Order: Cartesian grid first, then explicit CELLS in their listed order.
  """
  configs: list = []

  # 1. Cartesian product (ACTOR_MODES × CRITIC_MODES × SEEDS).
  for actor_mode, critic_mode in itertools.product(ACTOR_MODES, CRITIC_MODES):
    for seed in SEEDS:
      cfg = {
          'actor_mode': actor_mode,
          'critic_mode': critic_mode,
          'seed': seed,
      }
      cfg.update(EXTRA_OVERRIDES)
      configs.append(cfg)

  # 2. Explicit CELLS — each must declare the required keys.
  for i, cell in enumerate(CELLS):
    missing = [k for k in _REQUIRED_KEYS if k not in cell]
    if missing:
      raise ValueError(
          f'CELLS[{i}] is missing required keys {missing}: {cell!r}')
    cfg = dict(EXTRA_OVERRIDES)  # base
    cfg.update(cell)              # cell overrides EXTRA_OVERRIDES
    configs.append(cfg)

  return configs


def _format_cell(c, num_cols=8):
  """Pretty-print one cell as a column-aligned table row."""
  cols = [
      f'{c.get("actor_mode", "-"):<10}',
      f'{c.get("critic_mode", "-"):<11}',
      f'{c.get("seed", "-"):>4}',
      f'{c.get("single_task", "-"):<26}',
      f'{c.get("dyn_aux_weight", "-")!s:>6}',
      f'{c.get("log_mixture_norm", "-")!s:>6}',
      f'{c.get("log_probe_data", "-")!s:>6}',
      f'{c.get("phi_task_width", "-")!s:>5}',
      f'{c.get("phi_task_depth", "-")!s:>5}',
  ]
  return '  '.join(cols[:num_cols])


def main():
  parser = argparse.ArgumentParser(
      description='Enumerate experiment configurations.')
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--setting', type=int,
                     help='Print the config for this index (0-based).')
  group.add_argument('--total', action='store_true',
                     help='Print total number of configs.')
  group.add_argument('--list', action='store_true',
                     help='Print all configs as a table.')
  args = parser.parse_args()

  configs = build_configs()

  if args.total:
    print(len(configs))
    return

  if args.list:
    print(f'Total: {len(configs)} configurations')
    print(f'  Cartesian: {len(ACTOR_MODES) * len(CRITIC_MODES) * len(SEEDS)} '
          f'(actor_modes={ACTOR_MODES}, critic_modes={CRITIC_MODES}, '
          f'seeds={SEEDS})')
    print(f'  Explicit cells: {len(CELLS)}')
    if EXTRA_OVERRIDES:
      print(f'  EXTRA_OVERRIDES applied to all: {EXTRA_OVERRIDES}')
    print('')
    header = '  '.join([
        f'{"idx":>4}', f'{"actor":<10}', f'{"critic":<11}', f'{"seed":>4}',
        f'{"single_task":<26}',
        f'{"dynw":>6}', f'{"mixN":>6}', f'{"prob":>6}',
        f'{"ptw":>5}', f'{"ptd":>5}',
    ])
    print(header)
    print('-' * len(header))
    for i, c in enumerate(configs):
      print(f'{i:>4}  {_format_cell(c)}')
    return

  # --setting N: emit KEY=VALUE pairs (shell-sourceable).
  idx = args.setting
  if idx < 0 or idx >= len(configs):
    print(f'ERROR: setting {idx} out of range [0, {len(configs) - 1}]',
          file=sys.stderr)
    sys.exit(1)

  cfg = configs[idx]
  for key, value in cfg.items():
    # Booleans: emit as 'true' / 'false' so the shell scripts can use
    # them in the existing `if [ "$VAR" = "true" ]; then ...` pattern.
    if isinstance(value, bool):
      value = 'true' if value else 'false'
    elif isinstance(value, str):
      value = shlex.quote(value)
    print(f'{key.upper()}={value}')


if __name__ == '__main__':
  main()
