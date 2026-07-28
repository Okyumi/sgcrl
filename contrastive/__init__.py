"""Contrastive RL agent.

Exports resolve lazily (PEP 562) rather than eagerly.  Python runs a package's
``__init__`` before any of its submodules, so the previous eager import list
meant that ``from contrastive.knowledge_pool import KnowledgePool`` pulled in
the entire CRL agent stack (``agents`` -> ``builder`` -> ``distributed_layout``
-> ``launchpad``, plus ``learning`` and ``continual_learning``).  The SAC
baseline reuses only ``knowledge_pool`` / ``networks`` / ``rl_metrics`` /
``continual_config``, and its tests run without launchpad installed.

Attribute access is unchanged for callers: ``contrastive.ContrastiveConfig``,
``from contrastive import KnowledgePool``, ``contrastive.make_networks`` etc.
all still work — the defining module is simply imported on first use.
"""
import importlib

# Public name -> module that defines it.
_LAZY_EXPORTS = {
    'DistributedContrastive': 'contrastive.agents',
    'ContrastiveBuilder': 'contrastive.builder',
    'ContrastiveConfig': 'contrastive.config',
    'target_entropy_from_env_spec': 'contrastive.config',
    'ContrastiveLearner': 'contrastive.learning',
    'apply_policy_and_sample': 'contrastive.networks',
    'ContrastiveNetworks': 'contrastive.networks',
    'make_networks': 'contrastive.networks',
    # Continual RL extensions
    'ContinualConfig': 'contrastive.continual_config',
    'CONTINUAL_TASK_SEQUENCE': 'contrastive.continual_config',
    'ContinualContrastiveLearner': 'contrastive.continual_learning',
    'ContinualContrastiveBuilder': 'contrastive.continual_builder',
    'KnowledgePool': 'contrastive.knowledge_pool',
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
