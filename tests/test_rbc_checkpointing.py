"""Pure tests for RBC-DCC checkpoint identity."""
import pytest

from contrastive import rbc_checkpointing


def _config():
  return {
      'dyn_aux_weight': 1.0,
      'dyn_aux_after_task0': -1.0,
      'phi_task_width': 256,
      'phi_task_depth': 4,
      'combine_mode': 'add',
      'goal_encoder_mode': 'shared',
      'bellman_loss_weight': 1.0,
      'bellman_residual_l2_weight': 1e-4,
      'bellman_discount': 0.99,
      'bellman_tau': 0.005,
      'bellman_hidden_dim': 256,
      'her_reward_threshold': 0.05,
      'step_penalty_reward': True,
  }


@pytest.mark.parametrize('field', rbc_checkpointing.RBC_IDENTITY_FIELDS)
def test_every_rbc_setting_changes_the_checkpoint_fingerprint(field):
  baseline = _config()
  changed = dict(baseline)
  value = changed[field]
  if isinstance(value, bool):
    changed[field] = not value
  elif isinstance(value, str):
    changed[field] = value + '_other'
  else:
    changed[field] = value + 1
  assert (
      rbc_checkpointing.config_fingerprint(baseline)
      != rbc_checkpointing.config_fingerprint(changed))


def test_fingerprint_is_stable_under_mapping_order():
  config = _config()
  reversed_config = dict(reversed(list(config.items())))
  assert (
      rbc_checkpointing.config_fingerprint(config)
      == rbc_checkpointing.config_fingerprint(reversed_config))


def test_missing_identity_field_fails_loudly():
  config = _config()
  del config['her_reward_threshold']
  with pytest.raises(ValueError, match='her_reward_threshold'):
    rbc_checkpointing.config_fingerprint(config)
