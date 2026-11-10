"""Paper seeds and hyperparameters, using the genuine vanilla CRL learner."""
from experiment_configs_full_critic_gate_5seed import _shared, SEEDS

WANDB_GROUP = 'PAPER-VANILLA-CRL-GATE-BC-10SEED'


def config_for_seed(seed, smoke=False):
  if seed not in SEEDS:
    raise ValueError('Paper seeds are 5 through 14 inclusive.')
  config = _shared(seed)
  # These flags configure branches which do not exist in the vanilla network.
  for name in ('phi_task_width', 'phi_task_depth', 'combine_mode',
               'goal_encoder_mode', 'variant'):
    del config[name]
  config['intra_eval_previous_tasks'] = config.pop('intra_eval_previous')
  config.update(critic_mode='reset', actor_mode='reset',
                alg='contrastive_cpc', use_wandb=True,
                wandb_group=WANDB_GROUP,
                max_tasks_per_invocation=1, add_uid=True)
  if smoke:
    config.update(num_tasks=2, steps_per_task=20_000, base_steps=20_000,
                  eval_every=10_000, eval_episodes=2,
                  max_tasks_per_invocation=0,
                  wandb_group=WANDB_GROUP + '-SMOKE')
  return config


def runner_arguments(seed, root, smoke=False):
  config = config_for_seed(seed, smoke)
  config.update(checkpoint_dir=str(root / 'checkpoints'), log_dir=str(root / 'metrics'))
  return [f'--{key}={str(value).lower() if isinstance(value, bool) else value}'
          for key, value in config.items()]
