"""Static checks on ``sac.training`` and the entry point.

``sac/training.py`` imports reverb + TensorFlow + acme, so it cannot be
imported in a bare environment.  These tests read it with :mod:`ast` instead
and assert the wiring that is easy to get silently wrong:

* the W&B step axes must name a key inside their own metric family, otherwise
  ``define_metric(step_metric=...)`` silently points at a series that is never
  logged and the family falls back to the global step;
* every declared family must be a label actually passed to
  ``make_default_logger`` or a prefix the driver builds itself;
* ``wandb_auto_step=True`` is what makes the whole scheme necessary, so it must
  stay switched on here (and stay off for the contrastive driver);
* ``run_continual_sac.py`` must not import ``sac.training`` at module level, or
  ``--help`` would need reverb installed.
"""
import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAINING = REPO_ROOT / 'sac' / 'training.py'
ENTRYPOINT = REPO_ROOT / 'run_continual_sac.py'

# acme's EnvironmentLoop and learner both increment a counter keyed 'steps',
# which WandbLogger namespaces with the logger label.
_ACME_LABELS = frozenset({'learner', 'actor', 'evaluator'})


def _tree(path):
  return ast.parse(path.read_text())


def _step_metrics():
  """The ``_WANDB_STEP_METRICS`` literal as a list of (family, axis)."""
  for node in ast.walk(_tree(TRAINING)):
    if (isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == '_WANDB_STEP_METRICS'
                for t in node.targets)):
      return [tuple(ast.literal_eval(e)) for e in node.value.elts]
  raise AssertionError('No _WANDB_STEP_METRICS assignment in sac/training.py')


def _logger_labels():
  """First positional arg of every ``make_default_logger(...)`` call."""
  labels = set()
  for node in ast.walk(_tree(TRAINING)):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == 'make_default_logger' and node.args):
      labels.add(ast.literal_eval(node.args[0]))
  return labels


def _payload_prefixes():
  """Namespaces the driver writes directly, e.g. ``'rl_metrics/env_steps'``."""
  prefixes = set()
  for node in ast.walk(_tree(TRAINING)):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
      head, _, rest = node.value.partition('/')
      if rest and head.isidentifier():
        prefixes.add(head)
  return prefixes


# ---- W&B step metrics ----------------------------------------------------

def test_step_metric_families_are_unique():
  families = [family for family, _ in _step_metrics()]
  assert len(families) == len(set(families))


@pytest.mark.parametrize('family,axis', _step_metrics())
def test_step_axis_lives_inside_its_own_family(family, axis):
  """``define_metric('<family>/*', step_metric=axis)`` needs axis in-family."""
  assert axis.startswith(f'{family}/'), (family, axis)


@pytest.mark.parametrize('family,axis', _step_metrics())
def test_acme_families_are_plotted_against_the_counter_key(family, axis):
  """acme's counters emit 'steps'; anything else is a typo."""
  if family in _ACME_LABELS:
    assert axis == f'{family}/steps', (family, axis)


@pytest.mark.parametrize('family,axis', _step_metrics())
def test_every_family_is_either_a_logger_label_or_a_driver_payload(family, axis):
  assert family in _logger_labels() | _payload_prefixes(), family


@pytest.mark.parametrize('family,axis', _step_metrics())
def test_driver_built_axes_are_actually_logged(family, axis):
  """A non-acme axis must appear verbatim as a payload key."""
  if family not in _ACME_LABELS:
    assert f"'{axis}'" in TRAINING.read_text(), axis


def test_every_acme_logger_label_has_a_step_axis():
  families = {family for family, _ in _step_metrics()}
  assert _logger_labels() <= families, _logger_labels() - families


# ---- auto-step opt-in ----------------------------------------------------

def test_the_sac_driver_opts_into_wandb_auto_step():
  """Without it, wandb drops the family whose explicit step went backwards."""
  assert 'wandb_auto_step=True' in TRAINING.read_text()


def test_the_contrastive_driver_keeps_the_explicit_step():
  """Preserves bit-identical W&B behaviour for existing CRL runs."""
  crl = (REPO_ROOT / 'run_continual_contrastive.py').read_text()
  assert 'wandb_auto_step' not in crl


# ---- entry point import hygiene -----------------------------------------

def test_entrypoint_defers_the_heavy_import_into_main():
  tree = _tree(ENTRYPOINT)
  top_level = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)}
  assert 'sac.training' not in top_level
  assert not any(alias.name == 'sac.training'
                 for n in tree.body if isinstance(n, ast.Import)
                 for alias in n.names)
  # ... but it is imported somewhere (inside main).
  assert any(isinstance(n, ast.ImportFrom) and n.module == 'sac'
             and any(a.name == 'training' for a in n.names)
             for n in ast.walk(tree))


def test_entrypoint_adopts_the_flag_module_so_help_lists_every_flag():
  assert 'adopt_module_key_flags' in ENTRYPOINT.read_text()
