"""Default logger."""

import logging
from typing import Any, Callable, Mapping, Optional

from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal


class WandbLogger(base.Logger):
  """Logger that sends metrics to Weights & Biases. Use when wandb.init() has already been called."""

  def __init__(self, steps_key: str = 'learner_steps', prefix: Optional[str] = None,
               auto_step: bool = False):
    self._steps_key = steps_key
    self._prefix = prefix  # e.g. 'learner', 'actor', 'evaluator' for namespaced keys
    # When several WandbLogger instances share one wandb run (actor / learner /
    # evaluator) their counters advance at very different rates, so the
    # explicit `step=` values below are non-monotonic across labels and wandb
    # drops the entries with the smaller step ("user provided step ... is less
    # than current step ... Dropping entry").  With auto_step=True the explicit
    # step is omitted: wandb auto-increments the global step and the driver is
    # expected to declare `wandb.define_metric('<label>/*', step_metric=...)`
    # so each family is plotted against its own logical x-axis.  Defaults to
    # False so existing contrastive runs are unchanged.
    self._auto_step = auto_step

  def _to_float(self, v: Any) -> Optional[float]:
    """Convert numeric values (int, float, numpy/jax scalars) to float for wandb."""
    if isinstance(v, (int, float)):
      return float(v)
    try:
      return float(v)
    except (TypeError, ValueError):
      return None

  def write(self, data: Mapping[str, Any]) -> None:
    try:
      import wandb
      step = data.get(self._steps_key) or data.get('steps')
      metrics = {}
      for k, v in data.items():
        fv = self._to_float(v)
        if fv is not None:
          key = f'{self._prefix}/{k}' if self._prefix else k
          metrics[key] = fv
      if metrics:
        if self._auto_step:
          wandb.log(metrics)
        else:
          wandb.log(metrics, step=step)
    except Exception:  # pylint: disable=broad-except
      pass  # Don't break training if wandb is unavailable or misconfigured

  def close(self) -> None:
    pass  # Required by acme.utils.loggers.base.Logger; wandb run is finished by the main process


def make_default_logger(
    label: str,
    save_data: bool = True,
    save_dir: str = 'logs',
    add_uid: bool = True,
    use_wandb: bool = False,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
    steps_key: str = 'steps',
    wandb_auto_step: bool = False,
) -> base.Logger:
  """Makes a default Acme logger.

  Args:
    label: Name to give to the logger.
    save_data: Whether to persist data.
    use_wandb: Whether to also log metrics to Weights & Biases (wandb.init must be called elsewhere).
    time_delta: Time (in seconds) between logging events.
    asynchronous: Whether the write function should block or not.
    print_fn: How to print to terminal (defaults to print).
    serialize_fn: An optional function to apply to the write inputs before
      passing them to the various loggers.
    steps_key: Key used for step count (e.g. 'learner_steps'); used by WandbLogger when use_wandb=True.
    wandb_auto_step: Let wandb auto-increment the global step instead of passing
      an explicit one (see WandbLogger). Requires the caller to declare
      wandb.define_metric(...) step metrics. Default False keeps the explicit
      step used by the contrastive driver.

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  if not print_fn:
    print_fn = logging.info
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

  loggers = [terminal_logger]

  if save_data:
    loggers.append(csv.CSVLogger(label=label, directory_or_file = save_dir, add_uid = add_uid))

  if use_wandb:
    loggers.append(WandbLogger(steps_key=steps_key, prefix=label,
                               auto_step=wandb_auto_step))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)
  logger = filters.TimeFilter(logger, time_delta)

  return logger
