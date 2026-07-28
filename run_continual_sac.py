r"""Sequential continual goal-conditioned SAC+HER driver.

The supported entry point for the SAC baseline. Mirrors
``run_continual_contrastive.py`` -- same Sawyer environments, HER data
pipeline, CKA actor decomposition, knowledge pool, checkpointing, cross-task
evaluation and W&B conventions -- but swaps the InfoNCE contrastive critic for
a standard goal-conditioned SAC critic, so a paired run isolates the effect of
the critic.

Differences from the CRL driver:

  * Networks come from ``sac.networks.make_sac_networks`` (twin scalar Q;
    the actor architecture is identical to CRL).
  * The learner is ``sac.learning.ContinualSACLearner`` (SAC TD loss against
    HER-relabeled rewards, SAC actor loss against min-Q, adaptive alpha).
  * The reverb pipeline recomputes reward and discount from the sampled
    relabeled goal (``sac.her``), because SAC needs a TD signal consistent
    with that goal while InfoNCE never reads the reward.
  * Contrastive-only machinery is absent: negative bank, ``energy_fn``,
    ``logsumexp_penalty``, and the dual-encoder representation metrics
    (replaced by ``sac.metrics``).
  * ``--alg`` defaults to ``sac_her`` so logs and checkpoints never collide
    with CRL runs.

Smoke test (2 tasks, 10k steps each, no W&B):

  python run_continual_sac.py \
      --seed=42 --num_tasks=2 --steps_per_task=10000 --base_steps=10000 \
      --k_max=2 --nouse_wandb

Full runs (R/R and P/P over the 10-task sequence):

  python run_continual_sac.py --seed=1 --actor_mode=reset --critic_mode=reset
  python run_continual_sac.py --seed=1 --actor_mode=persistent \
      --critic_mode=persistent

``--help`` lists every flag with its default. All flags are defined in
``sac/flags.py``; the training loop lives in ``sac/training.py`` and is
imported inside ``main`` so ``--help`` works without reverb / TensorFlow
installed.
"""
from absl import app
from absl import flags

from sac import flags as sac_flags  # Defines every flag; keep this import.

FLAGS = flags.FLAGS

# The flags live in sac/flags.py, so absl would not consider them "key flags"
# of this module and --help would list nothing. Adopting them makes --help the
# full flag reference (--helpfull additionally shows absl's own flags).
flags.adopt_module_key_flags(sac_flags)


def main(_):
  # Imported here (not at module level) so that --help, and the flag/config
  # unit tests, do not require reverb + tensorflow + acme.
  from sac import training  # pylint: disable=import-outside-toplevel
  training.run(FLAGS)


if __name__ == '__main__':
  app.run(main)
