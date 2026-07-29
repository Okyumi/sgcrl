"""Dependency-light wiring checks for the RBC-DCC runner."""
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER = (REPO_ROOT / 'run_continual_contrastive.py').read_text()


def test_rbc_has_a_distinct_learner_route():
  assert "elif critic_mode == 'rbc_decomposed':" in RUNNER
  assert 'ContinualRBCDecomposedLearner(' in RUNNER
  assert 'ContinualDecomposedLearner(' in RUNNER


def test_rbc_reuses_the_canonical_sac_her_helper():
  assert 'sac_her.her_reward_and_discount(' in RUNNER
  assert "if critic_mode == 'rbc_decomposed':" in RUNNER


def test_rbc_checkpoint_paths_use_the_complete_fingerprint():
  assert 'rbc_checkpointing.config_fingerprint(rbc_config)' in RUNNER
  assert 'rbc_config=_rbc_identity_config()' not in RUNNER
  assert '_rbc_identity_config()' in RUNNER


def test_existing_decomposed_mode_remains_a_separate_branch():
  assert "if critic_mode == 'decomposed':\n    decomp_nets =" in RUNNER
  assert "if critic_mode == 'decomposed':\n    # Sibling learner" in RUNNER


def test_every_contrastive_launcher_forwards_rbc_and_dcc_architecture_flags():
  candidates = (
      'draft_3.sh', 'draft_4.sh', 'DRAFT.sh',
      'submit_continual_torch.sh')
  launchers = tuple(
      launcher for launcher in candidates
      if (REPO_ROOT / launcher).exists())
  assert {'draft_3.sh', 'draft_4.sh'} <= set(launchers)
  required = (
      '--combine_mode=',
      '--goal_encoder_mode=',
      '--bellman_loss_weight=',
      '--bellman_residual_l2_weight=',
      '--bellman_discount=',
      '--bellman_tau=',
      '--bellman_hidden_dim=',
      '--her_reward_threshold=',
  )
  for launcher in launchers:
    text = (REPO_ROOT / launcher).read_text()
    for flag in required:
      assert flag in text, (launcher, flag)
