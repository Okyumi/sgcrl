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
  steps_per_task: int = 8_000_000       # env steps per task
  base_steps: int = 8_000_000           # env steps for the base (first) task

  # -- Knowledge pool ---------------------------------------------------------
  k_max: int = 10                       # max pool size before merging

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

  # -- Decomposed critic (proposal 1) -----------------------------------------
  # When critic_mode='decomposed' the critic factors into a shared body
  # b_shared with two heads (contrastive h_phi + dynamics h_dyn) and a
  # task-specific encoder phi_task that is reset every task. The dynamics
  # auxiliary regresses the masked next state; mu controls its weight.
  # See docs/2026-05-08_plan_proposal1_dyn_aux.md for the full design.
  dyn_aux_weight: float = 1.0           # mu in the algorithm
  phi_task_width: int = 256             # smaller than the shared body
  phi_task_depth: int = 4               # one residual block (block_size=4)

  # -- Embedding combination & goal-encoder mode (DCC ablation handles) -----
  # `combine_mode='add'` is the default (z_sa = h_phi(b_shared) + phi_task);
  # `combine_mode='concat'` switches to z_sa = [h_phi(b_shared); phi_task],
  # automatically attaching a learnable Linear projection on top of psi(g)
  # so the contrastive score is taken in matching 2*repr_dim space.
  combine_mode: str = 'add'
  # `goal_encoder_mode='shared'` reuses a single psi across tasks (current
  # behaviour). `goal_encoder_mode='projected'` keeps the shared psi but
  # adds a Linear projection on top, regardless of combine_mode (handy
  # ablation for testing whether the projection itself is what helps).
  # The richer variants (task_specific / partial_shared / decomposed) are
  # exercised on the BuilderBench port; on sgcrl we keep the shared psi
  # path canonical and rely on the projection knob for ablations.
  goal_encoder_mode: str = 'shared'

  # -- RBC-DCC Bellman calibration ------------------------------------------
  bellman_loss_weight: float = 1.0
  bellman_residual_l2_weight: float = 1e-4
  bellman_discount: float = 0.99
  bellman_tau: float = 0.005
  bellman_hidden_dim: int = 256
  her_reward_threshold: float = 0.05
  step_penalty_reward: bool = True

  # -- CKA diagnostics --------------------------------------------------------
  # Pairwise cosine-similarity logging on the actor / critic knowledge pools.
  # Off by default so existing runs are bit-for-bit identical. Turn on for
  # CKA-failure diagnostic experiments (see docs/2026-05-08_plan_proposal1_dyn_aux.md
  # section 3.1 and section 9). Logged at every task boundary; cost is one
  # host-side matmul per pool, negligible relative to the 8M-step task.
  log_pool_cosine: bool = True

  # Per-step ratio || sum_j alpha_j v_j || / || v_k || logged inside the
  # CKA inner loop. See plan section 3.2: tests whether the actor / critic
  # update is dominated by v_k (small ratio) versus the mixture term
  # (large ratio). Off by default; one extra norm per inner step for each
  # active CKA path when on. See docs/2026-05-08_d5_mixture_norm.md.
  log_mixture_norm: bool = False

  # Per-task (obs, action) sample dumped at end of each task for the
  # linear-probe task-classifier diagnostic (plan section 3.4). Off by
  # default. When True, writes config.batch_size transitions per task to
  # `probe_data_task{k}_seed{s}.npz` next to the checkpoint. Consumed by
  # eval_linear_probe.py. See docs/2026-05-08_d6_linear_probe.md.
  log_probe_data: bool = False

  # -- Misc -------------------------------------------------------------------
  clear_replay_per_task: bool = True    # clear replay buffer when switching task
  seed: int = 42
