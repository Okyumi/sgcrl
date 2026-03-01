import argparse
import os
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from acme import specs
from acme import core
from acme.tf import savers as tf_savers

import contrastive.config as contrastive_config
import contrastive.learning as contrastive_learning
import contrastive.networks as contrastive_networks
import contrastive.utils as contrastive_utils
import lp_contrastive


def build_config(env_name: str, alg_name: str, seed: int) -> contrastive_config.ContrastiveConfig:
    """Reconstruct a ContrastiveConfig consistent with training."""
    params = {
        "seed": seed,
        "env_name": env_name,
        "alg_name": alg_name,
        "max_number_of_steps": 8_000_000,
    }

    if alg_name == "contrastive_cpc":
        params["use_cpc"] = True
    elif alg_name == "c_learning":
        params["use_td"] = True
        params["twin_q"] = True
    elif alg_name == "nce+c_learning":
        params["use_td"] = True
        params["twin_q"] = True
        params["add_mc_to_td"] = True

    return contrastive_config.ContrastiveConfig(**params)


def build_env_and_networks(
    env_name: str,
    alg_name: str,
    seed: int,
    fix_goals: bool,
) -> Tuple[object, contrastive_config.ContrastiveConfig, contrastive_networks.ContrastiveNetworks]:
    """Create dm_env environment, config, and networks matching training."""
    config = build_config(env_name, alg_name, seed)

    # Mirrors lp_contrastive.get_program / get_env.
    env, obs_dim = lp_contrastive.get_env(
        env_name,
        config.start_index,
        config.end_index,
        seed,
        fix_goals=fix_goals,
    )
    config.obs_dim = obs_dim
    config.max_episode_steps = getattr(env, "_step_limit") + 1

    env_spec = specs.make_environment_spec(env)
    networks = contrastive_networks.make_networks(
        env_spec,
        obs_dim=obs_dim,
        repr_dim=config.repr_dim,
        repr_norm=config.repr_norm,
        twin_q=config.twin_q,
        use_image_obs=config.use_image_obs,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )
    return env, config, networks


def build_learner_and_checkpointer(
    run_dir: str,
    env_name: str,
    alg_name: str,
    seed: int,
    fix_goals: bool,
):
    """Rebuild a ContrastiveLearner and a TF Checkpointer (no restore yet)."""
    import optax

    env, config, networks = build_env_and_networks(env_name, alg_name, seed, fix_goals)

    policy_optimizer = optax.adam(learning_rate=config.actor_learning_rate, eps=1e-7)
    q_optimizer = optax.adam(learning_rate=config.learning_rate, eps=1e-7)

    dummy_iterator = iter(())

    learner = contrastive_learning.ContrastiveLearner(
        networks=networks,
        rng=jax.random.PRNGKey(seed),
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        iterator=dummy_iterator,
        counter=None,
        logger=None,
        obs_to_goal=jax.jit(
            lambda obs: contrastive_utils.obs_to_goal_2d(
                obs, start_index=config.start_index, end_index=config.end_index
            )
        ),
        config=config,
    )

    class LearnerSaveable(core.Saveable):
        """Adapter so tf_savers.Checkpointer can checkpoint the JAX learner."""

        def __init__(self, wrapped_learner):
            self._wrapped_learner = wrapped_learner

        def save(self):
            return self._wrapped_learner.save()

        def restore(self, state):
            self._wrapped_learner.restore(state)

    saveable = LearnerSaveable(learner)

    checkpointer = tf_savers.Checkpointer(
        objects_to_save={"learner": saveable},
        directory=run_dir,
        subdirectory="learner",
        time_delta_minutes=10_000.0,
        add_uid=False,
        max_to_keep=10,
    )

    return env, config, networks, learner, checkpointer


def _get_base_mujoco_env(env) -> object:
    """Unwrap dm_env/gym wrappers until we get an env with .sim (mujoco_py MjSim)."""
    e = env
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if hasattr(e, "sim"):
            return e
        e = getattr(e, "_environment", None) or getattr(e, "env", None) or getattr(e, "environment", None)
    return None


def _render_rgb_array(dm_env_env) -> np.ndarray:
    """Best-effort helper to get an RGB frame from a dm_env-wrapped gym env.

    For Meta-World/mujoco_py envs, uses offscreen rendering (MjRenderContextOffscreen)
    so no display/GLFW is required. Falls back to env.render() for other envs.
    """
    # 1) Headless MuJoCo path: unwrap to base env with .sim and use offscreen context.
    base = _get_base_mujoco_env(dm_env_env)
    if base is not None and hasattr(base, "sim"):
        try:
            import mujoco_py
            sim = base.sim
            # Reuse a single offscreen context per env to avoid creating one per frame.
            if not hasattr(base, "_sgcrl_offscreen_ctx"):
                base._sgcrl_offscreen_ctx = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
            ctx = base._sgcrl_offscreen_ctx
            w, h = 640, 480
            ctx.render(w, h)
            frame = ctx.read_pixels(w, h, depth=False)
            # read_pixels often returns (H,W,3) with rows in reverse order.
            if frame is not None and len(frame.shape) == 3:
                return np.asarray(frame[::-1])
        except Exception:
            pass

    # 2) Generic path: try env.render(mode="rgb_array") or env.render().
    candidates = [
        getattr(dm_env_env, "environment", None),
        getattr(dm_env_env, "_environment", None),
        getattr(dm_env_env, "gym_env", None),
        dm_env_env,
    ]
    for env in candidates:
        if env is None:
            continue
        render = getattr(env, "render", None)
        if callable(render):
            try:
                frame = render(mode="rgb_array")
            except TypeError:
                frame = render()
            if frame is not None:
                return np.asarray(frame)

    raise RuntimeError(
        "Could not obtain RGB frame from environment; "
        "check that the env supports render(mode='rgb_array') or use headless MuJoCo."
    )


def _record_single_video(
    env,
    networks: contrastive_networks.ContrastiveNetworks,
    learner: contrastive_learning.ContrastiveLearner,
    env_name: str,
    alg_name: str,
    seed: int,
    num_episodes: int,
    output_dir: str,
    suffix: str,
) -> str:
    """Record num_episodes from the current learner parameters into one MP4."""
    import imageio

    os.makedirs(output_dir, exist_ok=True)
    video_filename = f"{env_name}_{alg_name}_seed{seed}{suffix}.mp4"
    video_path = os.path.join(output_dir, video_filename)

    policy_apply = contrastive_networks.apply_policy_and_sample(
        networks, eval_mode=True
    )
    policy_params = learner.get_variables(["policy"])[0]
    rng = jax.random.PRNGKey(seed + 12345)

    writer = imageio.get_writer(video_path, fps=30)
    try:
        for _ in range(num_episodes):
            timestep = env.reset()
            obs = timestep.observation

            writer.append_data(_render_rgb_array(env))

            while not timestep.last():
                rng, subkey = jax.random.split(rng)
                obs_batch = jnp.asarray(obs)[None, ...]
                action = policy_apply(policy_params, subkey, obs_batch)
                action_np = np.asarray(action)[0]
                timestep = env.step(action_np)
                obs = timestep.observation
                writer.append_data(_render_rgb_array(env))
    finally:
        writer.close()

    return video_path


def record_videos_from_checkpoint(
    run_dir: str,
    env_name: str,
    alg_name: str,
    seed: int,
    num_episodes: int,
    output_dir: str,
    fix_goals: bool,
    all_checkpoints: bool,
) -> List[str]:
    """Record videos from either the latest or all checkpoints for a run.

    Returns a list of paths to the generated MP4 files.
    """
    env, _, networks, learner, checkpointer = build_learner_and_checkpointer(
        run_dir=run_dir,
        env_name=env_name,
        alg_name=alg_name,
        seed=seed,
        fix_goals=fix_goals,
    )

    video_paths: List[str] = []

    if all_checkpoints:
        # Enumerate checkpoints directly from the TF checkpoint state file,
        # without relying on any private attributes of Acme's Checkpointer.
        import tensorflow as tf  # Imported lazily to avoid unnecessary deps.

        ckpt_dir = os.path.join(run_dir, "checkpoints", "learner")
        ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
        if not ckpt_state or not ckpt_state.all_model_checkpoint_paths:
            raise RuntimeError(
                f"No checkpoints found under {ckpt_dir}; "
                "did training finish and save any checkpoints?"
            )

        ckpt_paths = list(ckpt_state.all_model_checkpoint_paths)
        # Iterate oldest -> newest.
        for ckpt_path in ckpt_paths:
            # Normalize to an absolute path if needed.
            if not os.path.isabs(ckpt_path):
                ckpt_path_full = os.path.join(ckpt_dir, os.path.basename(ckpt_path))
            else:
                ckpt_path_full = ckpt_path

            # Restore this specific checkpoint into the learner.
            status = checkpointer._checkpoint.restore(ckpt_path_full)  # type: ignore[attr-defined]
            # Allow missing/non-critical fields without raising.
            try:
                status.expect_partial()
            except Exception:
                pass

            ckpt_name = os.path.basename(ckpt_path_full)  # e.g. 'ckpt-57'
            suffix = f"_{ckpt_name}"
            video_paths.append(
                _record_single_video(
                    env=env,
                    networks=networks,
                    learner=learner,
                    env_name=env_name,
                    alg_name=alg_name,
                    seed=seed,
                    num_episodes=num_episodes,
                    output_dir=output_dir,
                    suffix=suffix,
                )
            )
    else:
        # Just restore and record from the latest checkpoint.
        checkpointer.restore()
        video_paths.append(
            _record_single_video(
                env=env,
                networks=networks,
                learner=learner,
                env_name=env_name,
                alg_name=alg_name,
                seed=seed,
                num_episodes=num_episodes,
                output_dir=output_dir,
                suffix="",
            )
        )

    return video_paths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load contrastive RL checkpoints and record rollouts to MP4. "
            "Can either use only the latest checkpoint or loop over all saved "
            "checkpoints for a run."
        )
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help=(
            "Path to a single run directory, e.g. "
            "logs/contrastive_cpc_sawyer_push_2/<uuid>"
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment name used during training, e.g. 'sawyer_push'.",
    )
    parser.add_argument(
        "--alg",
        type=str,
        required=True,
        help="Algorithm name used during training, e.g. 'contrastive_cpc'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Training seed for this run.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to record per checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_videos",
        help="Directory where MP4 files will be written.",
    )
    parser.add_argument(
        "--no_fix_goals",
        action="store_true",
        help="If set, do not use fixed goals when recreating the environment.",
    )
    parser.add_argument(
        "--all_checkpoints",
        action="store_true",
        help=(
            "If set, loop over all saved checkpoints for this run and write "
            "one video per checkpoint."
        ),
    )

    args = parser.parse_args()

    video_paths = record_videos_from_checkpoint(
        run_dir=args.run_dir,
        env_name=args.env,
        alg_name=args.alg,
        seed=args.seed,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        fix_goals=not args.no_fix_goals,
        all_checkpoints=args.all_checkpoints,
    )
    for path in video_paths:
        print(f"Wrote video to {path}")


if __name__ == "__main__":
    main()

