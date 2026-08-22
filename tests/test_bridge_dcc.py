"""Dependency-light wiring checks for the Bridge-DCC pilots."""
from pathlib import Path

import experiment_configs


ROOT = Path(__file__).resolve().parents[1]


def test_bridge_cells_are_matched_task5_task8_ablations():
  cells = experiment_configs.build_configs()[18:36]
  assert len(cells) == 18
  assert {cell['seed'] for cell in cells} == {5, 6, 7}
  assert {cell['single_task'] for cell in cells} == {
      'sawyer_handle_press_side', 'sawyer_window_close'}
  assert {cell['critic_mode'] for cell in cells} == {
      'iwr_decomposed', 'advantage_decomposed', 'bridge_decomposed'}
  assert {cell['in_trajectory_negative_repeats'] for cell in cells} == {12}
  assert all(cell['action_landscape_interaction_aware_anchor']
             for cell in cells)
  assert {cell['action_landscape_rollout_horizon'] for cell in cells} == {100}


def test_mode_flags_match_the_three_ablation_families():
  cells = experiment_configs.build_configs()[18:36]
  for cell in cells:
    mode = cell['critic_mode']
    assert cell.get('interaction_weighted_relabeling', False) == (
        mode in {'iwr_decomposed', 'bridge_decomposed'})
    assert cell.get('action_effect_enabled', False) == (
        mode in {'advantage_decomposed', 'bridge_decomposed'})


def test_forward_action_effect_is_not_inverse_dynamics():
  learner = (ROOT / 'contrastive' /
             'continual_learning_decomposed.py').read_text()
  networks = (ROOT / 'contrastive' / 'decomposed_networks.py').read_text()
  assert 'action_effect_discount * continuation * psi_next - psi_state' in learner
  assert 'jax.lax.stop_gradient' in learner
  assert 'advantage_raw = jnp.sum(effect * goal_repr, axis=1)' in learner
  assert "name='u_task'" in networks


def test_iwr_changes_future_sampling_but_preserves_support():
  runner = (ROOT / 'run_continual_contrastive.py').read_text()
  assert 'def _interaction_candidate_weights(all_state):' in runner
  assert 'interaction_weight_floor + tf.exp' in runner
  assert 'probs *= _interaction_candidate_weights(all_state)[None, :]' in runner


def test_torch_wrapper_selects_all_eighteen_bridge_cells():
  wrapper = (ROOT / 'DRAFT_bridge_dcc.sh').read_text()
  assert '#SBATCH --array=0-5' in wrapper
  assert 'CONFIG_INDEX_OFFSET=18' in wrapper
  assert 'CONFIG_LIMIT=18' in wrapper
  assert 'ACTION_LANDSCAPE_SELF_TEST=true' in wrapper
  assert 'exec bash "$REPO_DIR/DRAFT.sh"' in wrapper


def test_restore_preflight_runs_after_torch_environment_activation():
  launcher = (ROOT / 'DRAFT.sh').read_text()
  preflight = 'python -m contrastive.action_ranking_diagnostics'
  assert 'ACTION_LANDSCAPE_SELF_TEST' in launcher
  assert '--self-test-env=sawyer_handle_press_side' in launcher
  assert '--self-test-env=sawyer_window_close' in launcher
  assert launcher.index(
      'conda activate contrastive_rl') < launcher.index(preflight)
  assert launcher.index(
      'source "${REPO_DIR}/set_up/torch_hpc_env.sh"') < launcher.index(preflight)


def test_canonical_launcher_forwards_bridge_and_anchor_flags():
  launcher = (ROOT / 'DRAFT.sh').read_text()
  for flag in (
      'interaction_weighted_relabeling',
      'interaction_threshold',
      'interaction_bandwidth',
      'interaction_weight_floor',
      'action_effect_enabled',
      'action_effect_loss_weight',
      'action_effect_discount',
      'action_effect_temperature',
      'action_effect_actor_weight',
      'action_effect_normalization_eps',
      'action_effect_q_scale_ema_decay',
      'action_effect_hidden_dim',
      'action_landscape_interaction_aware_anchor',
      'action_landscape_anchor_search_steps',
      'action_landscape_interaction_threshold',
  ):
    assert f'--{flag}' in launcher


def test_bridge_implementation_has_a_docs_note_and_repository_policy():
  note = ROOT / 'docs' / '2026-08-22_bridge_dcc_action_credit_implementation.md'
  assert note.is_file()
  assert 'DRAFT_bridge_dcc.sh' in note.read_text()
  policy = (ROOT / 'AGENTS.md').read_text()
  assert 'under `docs/`' in policy
  assert 'Do not push implementation-only commits' in policy
