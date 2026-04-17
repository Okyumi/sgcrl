### Recording training videos from checkpoints

This document explains how to load a trained contrastive RL checkpoint and export rollouts as MP4 videos.

- **Script location**: `rollout_training_videos.py`
- **Output directory**: `training_videos/` (created automatically if missing)

#### 1. Prerequisites

- Checkpoints for a finished run, e.g.:
  - `logs/contrastive_cpc_sawyer_push_2/<UUID>/checkpoints/learner/…`
- Python dependencies (in the same environment you used to train):
  - `jax`, `acme`, `haiku`, etc. (already required for training)
  - `imageio` with ffmpeg support:

```bash
pip install "imageio[ffmpeg]"
```

#### 2. Basic usage

From the repo root:

```bash
python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_push_2/<UUID> \
  --env sawyer_push \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 5 \
  --output_dir training_videos
```

- **`--run_dir`**: per-run directory under `logs/` (the one containing `checkpoints/` and `logs/`).
- **`--env`**: environment name used during training (e.g. `sawyer_bin`, `sawyer_box`, `sawyer_push`, …).
- **`--alg`**: algorithm name (`contrastive_cpc`, `c_learning`, or `nce+c_learning`).
- **`--seed`**: training seed for this run (matches `--seed` in `lp_contrastive.py`).
- **`--num_episodes`**: number of episodes to concatenate into the MP4.
- **`--output_dir`**: where to write the MP4 (default: `training_videos`).

The script will create a file named like:

```text
training_videos/<env>_<alg>_seed<seed>.mp4
```

#### 3. Fixed vs. sampled goals

By default, the script recreates the environment with **fixed goals** (matching runs trained without `--sample_goals`).

- If you trained with uniform goal sampling (`--sample_goals` in `lp_contrastive.py`), disable fixed goals:

```bash
python rollout_training_videos.py ... --no_fix_goals
```

