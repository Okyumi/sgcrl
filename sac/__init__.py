"""Continual goal-conditioned SAC + HER baseline.

The SAC counterpart of the ``contrastive`` package: same environments, HER
data pipeline, CKA knowledge pool, checkpointing and evaluation, with the
InfoNCE critic replaced by a twin scalar Q trained on a sparse HER reward.
Entry point: ``run_continual_sac.py``.

Module map, lightest first:

============================ ==============================================
``sac.her``                  sparse reward + terminal-discount rule
``sac.tasks``                task sequence, fixed goals, transfer rules
``sac.checkpointing``        cross-task checkpoint paths and I/O
``sac.flags``                absl flags and config builders (absl only)
``sac.metrics``              representation metrics for the SAC critic
``sac.networks``            twin-Q + Gaussian-tanh actor (jax/haiku)
``sac.learning``             ``ContinualSACLearner`` (jax/optax/acme)
``sac.training``             training loop (also needs reverb + tensorflow)
============================ ==============================================

Exports resolve lazily (PEP 562) so importing e.g. :mod:`sac.her` in a test
does not pull in ``reverb``/``tensorflow``.
"""
import importlib

# Public name -> module that defines it.
_LAZY_EXPORTS = {
    'ContinualSACLearner': 'sac.learning',
    'ContinualSACTrainingState': 'sac.learning',
    'SACNetworks': 'sac.networks',
    'make_sac_networks': 'sac.networks',
    'apply_policy_and_sample': 'sac.networks',
    'apply_policy_k_sample_argmax': 'sac.networks',
    'her_reward_and_discount': 'sac.her',
    'compute_sac_metrics': 'sac.metrics',
    'FIXED_GOALS': 'sac.tasks',
    'resolve_task_sequence': 'sac.tasks',
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
  """Import the defining module on first attribute access (PEP 562)."""
  module_name = _LAZY_EXPORTS.get(name)
  if module_name is None:
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
  value = getattr(importlib.import_module(module_name), name)
  globals()[name] = value  # cache so later lookups skip __getattr__
  return value


def __dir__():
  return sorted(list(globals()) + __all__)
