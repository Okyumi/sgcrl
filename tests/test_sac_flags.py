"""Tests for flag defaults and the config builders in ``sac.flags``.

``sac.flags`` deliberately imports only absl plus the light ``sac`` modules, so
these run without reverb / TensorFlow / acme.  The builders all accept a
``flag_values`` argument; the helper below feeds them a stand-in built from the
*real* flag defaults, so no test mutates the global absl singleton and the
stand-in cannot drift from the definitions.
"""
import types

import pytest
from absl import flags as absl_flags

from sac import flags as sac_flags
from sac import tasks

FLAGS = sac_flags.FLAGS


def _flag_values(**overrides):
  """A FLAGS stand-in holding every registered default, plus ``overrides``."""
  unknown = [name for name in overrides if name not in FLAGS]
  assert not unknown, f'not real flags: {unknown}'
  values = {name: FLAGS[name].default for name in FLAGS}
  values.update(overrides)
  return types.SimpleNamespace(**values)


# ---- defaults ------------------------------------------------------------

@pytest.mark.parametrize('name,expected', [
    ('seed', 42),
    ('alg', 'sac_her'),
    ('num_tasks', 10),
    ('use_20_tasks', False),
    ('single_task', ''),
    ('task_sequence', ''),
    ('steps_per_task', 8_000_000),
    ('base_steps', 8_000_000),
    ('actor_mode', 'cka'),
    ('critic_mode', 'persistent'),
    ('k_max', 10),
    ('adapt_heads_only', True),
    ('use_task_id', False),
    ('her_reward_threshold', 0.05),
    ('step_penalty_reward', True),
    ('use_residual', True),
    ('network_width', 256),
    ('critic_depth', 4),
    ('actor_depth', 4),
    ('log_rl_metrics', True),
    ('actor_auto_reset', False),
    ('auto_resume', True),
    ('start_task', 0),
    ('use_wandb', True),
    ('wandb_project', 'continual_sac'),
    ('wandb_entity', ''),
    ('wandb_group', 'sac_baseline'),
    ('wandb_mode', 'online'),
])
def test_flag_default(name, expected):
  assert FLAGS[name].default == expected


def test_actor_and_critic_modes_are_restricted_to_the_known_modes():
  assert FLAGS['actor_mode'].parser.enum_values == list(tasks.ACTOR_MODES)
  assert FLAGS['critic_mode'].parser.enum_values == list(tasks.CRITIC_MODES)


def test_unknown_transfer_mode_is_rejected_at_parse_time():
  with pytest.raises(ValueError):
    FLAGS['actor_mode'].parser.parse('sideways')
  with pytest.raises(ValueError):
    FLAGS['critic_mode'].parser.parse('sideways')


def test_wandb_mode_offers_online_offline_disabled():
  assert FLAGS['wandb_mode'].parser.enum_values == [
      'online', 'offline', 'disabled']


def test_no_flag_default_looks_like_a_credential():
  """Guards the 'no API key in the repo' constraint."""
  for name in ('wandb_entity', 'wandb_project', 'wandb_group'):
    assert 'key' not in str(FLAGS[name].default).lower()
  assert not any('api_key' in name for name in FLAGS)


# ---- task sequence resolution -------------------------------------------

def test_resolve_tasks_defaults_to_the_10_task_sequence():
  sequence, num_tasks = sac_flags.resolve_tasks(_flag_values())
  assert num_tasks == 10
  assert len(sequence) == 10


def test_resolve_tasks_honours_an_explicit_sequence():
  sequence, num_tasks = sac_flags.resolve_tasks(
      _flag_values(task_sequence='sawyer_push,sawyer_hammer'))
  assert sequence == ('sawyer_push', 'sawyer_hammer')
  assert num_tasks == 2


def test_resolve_tasks_honours_single_task():
  sequence, num_tasks = sac_flags.resolve_tasks(
      _flag_values(single_task='sawyer_window_close', use_20_tasks=True))
  assert sequence == ('sawyer_window_close',)
  assert num_tasks == 1


def test_resolve_tasks_expands_use_20_tasks():
  sequence, num_tasks = sac_flags.resolve_tasks(
      _flag_values(use_20_tasks=True, num_tasks=20))
  assert num_tasks == 20
  assert sequence[10:] == sequence[:10]


def test_resolve_tasks_rejects_an_unknown_environment():
  with pytest.raises(ValueError, match='sawyer_teleport'):
    sac_flags.resolve_tasks(_flag_values(task_sequence='sawyer_teleport'))


def test_resolve_tasks_validates_only_the_prefix_it_will_run():
  """--num_tasks=2 over the 10-task default must not trip on later tasks."""
  sequence, num_tasks = sac_flags.resolve_tasks(_flag_values(num_tasks=2))
  assert num_tasks == 2
  assert len(sequence) == 10


# ---- ContinualConfig ----------------------------------------------------

def test_build_continual_config_copies_the_budget_and_pool_flags():
  cfg = sac_flags.build_continual_config(
      4, _flag_values(steps_per_task=123, base_steps=456, k_max=7, seed=9,
                      checkpoint_dir='/tmp/ck'))
  assert cfg.num_tasks == 4
  assert cfg.steps_per_task == 123
  assert cfg.base_steps == 456
  assert cfg.k_max == 7
  assert cfg.seed == 9
  assert cfg.checkpoint_dir == '/tmp/ck'


def test_build_continual_config_uses_the_resolved_num_tasks_not_the_flag():
  cfg = sac_flags.build_continual_config(2, _flag_values(num_tasks=10))
  assert cfg.num_tasks == 2


# ---- ContrastiveConfig kwargs -------------------------------------------

def test_sac_always_runs_adaptive_alpha_and_twin_q():
  params = sac_flags.build_contrastive_params(_flag_values())
  assert params['entropy_coefficient'] is None  # adaptive alpha
  assert params['twin_q'] is True
  # -0.5 * |A| for the 4-D Sawyer action space (brax / JaxGCRL convention).
  assert params['target_entropy'] == -2.0


def test_contrastive_params_keep_fixed_goals_and_the_random_actor_prefill():
  params = sac_flags.build_contrastive_params(_flag_values())
  assert params['fix_goals'] is True
  assert params['use_random_actor'] is True


def test_env_name_is_left_for_the_driver_to_fill_per_task():
  params = sac_flags.build_contrastive_params(_flag_values())
  assert params['env_name'] == ''


def test_contrastive_params_track_the_architecture_flags():
  params = sac_flags.build_contrastive_params(
      _flag_values(use_residual=False, network_width=512, critic_depth=8,
                   actor_depth=12))
  assert params['use_residual'] is False
  assert params['network_width'] == 512
  assert params['critic_depth'] == 8
  assert params['actor_depth'] == 12


def test_alg_tag_flows_into_the_config_so_logs_never_collide_with_crl():
  params = sac_flags.build_contrastive_params(_flag_values())
  assert params['alg_name'] == 'sac_her'


# ---- W&B run config -----------------------------------------------------

def _wandb_config(**overrides):
  f = _flag_values(**overrides)
  params = sac_flags.build_contrastive_params(f)
  return sac_flags.wandb_run_config(params, task_id=3, env_name='sawyer_push',
                                    num_tasks=10, k_max=10, flag_values=f)


def test_wandb_config_records_the_task_identity():
  config = _wandb_config()
  assert config['task_id'] == 3
  assert config['env_name'] == 'sawyer_push'
  assert config['num_tasks'] == 10
  assert config['k_max'] == 10


@pytest.mark.parametrize('key', [
    'actor_mode', 'critic_mode', 'use_task_id', 'adapt_heads_only',
    'encoder_from_base', 'use_20_tasks', 'eval_episodes',
    'intra_eval_previous_tasks', 'k_sample_k', 'actor_auto_reset',
    'her_reward_threshold', 'step_penalty_reward', 'task_sequence',
])
def test_wandb_config_records_every_ablation_axis(key):
  assert key in _wandb_config()


def test_wandb_config_reflects_overrides():
  config = _wandb_config(actor_mode='reset', critic_mode='reset',
                         step_penalty_reward=False, her_reward_threshold=0.12)
  assert config['actor_mode'] == 'reset'
  assert config['critic_mode'] == 'reset'
  assert config['step_penalty_reward'] is False
  assert config['her_reward_threshold'] == 0.12


def test_wandb_config_carries_no_account_details():
  """Entity / project / group belong on wandb.init, not in the run config."""
  config = _wandb_config(wandb_entity='some-team')
  assert 'wandb_entity' not in config
  assert 'some-team' not in str(config)


# ---- every flag the launcher passes actually exists ---------------------

@pytest.mark.parametrize('name', [
    'seed', 'alg', 'num_tasks', 'steps_per_task', 'base_steps', 'k_max',
    'start_task', 'eval_every', 'eval_episodes', 'k_sample_k', 'log_dir',
    'checkpoint_dir', 'actor_mode', 'critic_mode', 'her_reward_threshold',
    'network_width', 'critic_depth', 'actor_depth',
    'actor_reset_dormant_threshold', 'actor_reset_warmup', 'actor_reset_max',
    'wandb_project', 'wandb_group', 'wandb_mode', 'wandb_entity',
    'single_task', 'task_sequence', 'use_wandb', 'add_uid', 'auto_resume',
    'use_task_id', 'adapt_heads_only', 'encoder_from_base', 'use_20_tasks',
    'intra_eval_previous_tasks', 'log_rl_metrics', 'use_residual',
    'step_penalty_reward', 'actor_auto_reset',
])
def test_flag_used_by_draft_sac_sh_is_defined(name):
  assert name in FLAGS, f'draft_sac.sh passes --{name} but it is not defined'


def test_flags_module_does_not_need_reverb_or_tensorflow():
  """Import hygiene: --help must work in a bare environment."""
  import sys
  assert 'reverb' not in sys.modules
  assert 'tensorflow' not in sys.modules
  assert isinstance(FLAGS, absl_flags.FlagValues)
