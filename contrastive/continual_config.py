"""Configuration for continual goal-conditioned contrastive RL."""
import dataclasses
from typing import List, Tuple


# CKA-RL 10-task Meta-World sequence (env_name strings matching env_utils.load)
CONTINUAL_TASK_SEQUENCE: Tuple[str, ...] = (
    'sawyer_hammer',
    'sawyer_push_wall',
    'sawyer_faucet_close',
    'sawyer_push_back',
    'sawyer_stick_pull',
    'sawyer_handle_press_side',
    'sawyer_push',
    'sawyer_shelf_place',
    'sawyer_window_close',
    'sawyer_peg_unplug_side',
)

# 20-task sequence: two passes of the 10-task sequence.
CONTINUAL_TASK_SEQUENCE_20: Tuple[str, ...] = (
    CONTINUAL_TASK_SEQUENCE + CONTINUAL_TASK_SEQUENCE
)


@dataclasses.dataclass
class ContinualConfig:
  """Continual RL specific configuration (layered on top of ContrastiveConfig)."""

  # -- Task sequence ----------------------------------------------------------
  num_tasks: int = 10
  task_sequence: Tuple[str, ...] = CONTINUAL_TASK_SEQUENCE

  # -- Steps ------------------------------------------------------------------
  steps_per_task: int = 1_000_000       # env steps per task
  base_steps: int = 1_000_000           # env steps for the base (first) task

  # -- Knowledge pool ---------------------------------------------------------
  k_max: int = 5                        # max pool size before merging

  # -- Actor CKA parameters --------------------------------------------------
  beta_init_std: float = 0.01           # std for β_k initialisation (N(0, std))
  use_alpha_scale: bool = True          # learnable scalar multiplier on α
  adapt_full_policy: bool = True        # adapt all policy layers (not just head)
  adapt_heads_only: bool = True         # only adapt head layers (CKA-RL default)

  # -- Checkpointing ----------------------------------------------------------
  checkpoint_dir: str = 'logs/continual_goal_crl'

  # -- Evaluation -------------------------------------------------------------
  eval_every_steps: int = 50_000        # evaluate within each task every N steps
  eval_all_tasks_at_end: bool = True    # evaluate on all previous tasks at end

  # -- Misc -------------------------------------------------------------------
  clear_replay_per_task: bool = True    # clear replay buffer when switching task
  seed: int = 42
