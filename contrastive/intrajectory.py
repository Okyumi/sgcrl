"""Pure helpers for CRTR/StableCRL in-trajectory negative sampling."""
from __future__ import annotations


def validate_repetition_factor(
    repeats: int,
    *,
    batch_size: int,
    episode_transitions: int,
) -> None:
  """Validate an in-trajectory repetition factor."""
  if repeats < 1:
    raise ValueError(
        'in_trajectory_negative_repeats must be at least 1; '
        f'got {repeats}.')
  if batch_size < 1:
    raise ValueError(f'batch_size must be positive; got {batch_size}.')
  if episode_transitions < 1:
    raise ValueError(
        'episode_transitions must be positive; '
        f'got {episode_transitions}.')
  if repeats > episode_transitions:
    raise ValueError(
        'in_trajectory_negative_repeats cannot exceed the number of '
        f'transitions per episode ({episode_transitions}); got {repeats}.')


def trajectories_per_batch(batch_size: int, repeats: int) -> int:
  """Return the episodes needed to fill one fixed-size critic batch."""
  validate_repetition_factor(
      repeats, batch_size=batch_size, episode_transitions=max(repeats, 1))
  return (batch_size + repeats - 1) // repeats


def in_batch_repetition_counts(batch_size: int, repeats: int) -> tuple[int, ...]:
  """Return the number of rows contributed by each sampled episode."""
  groups = trajectories_per_batch(batch_size, repeats)
  full_groups, remainder = divmod(batch_size, repeats)
  counts = [repeats] * full_groups
  if remainder:
    counts.append(remainder)
  assert len(counts) == groups
  return tuple(counts)
