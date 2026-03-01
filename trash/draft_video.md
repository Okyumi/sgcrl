### Commands to record training videos from all checkpoints

Each command below:
- Loads the specified run (alg = `contrastive_cpc`, seed = 2).
- Loops over **all saved checkpoints** for that run.
- Records `--num_episodes 3` episodes per checkpoint.
- Writes videos into `training_videos/` with filenames ending in the checkpoint name (e.g. `..._ckpt-57.mp4`).


export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONNOUSERSITE=1
module load conda-gcc/11.2.0
module load ffmpeg/4.2.2
module load mesa/20.2.1

```bash
python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_bin_2/46e790d4-0b28-11f1-a002-d4f5efb9a54c \
  --env sawyer_bin \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 3 \
  --output_dir training_videos \
  --all_checkpoints

python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_box_2/0358500e-13f4-11f1-aee0-b07b25d4b700 \
  --env sawyer_box \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 3 \
  --output_dir training_videos \
  --all_checkpoints

python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_hammer_2/a3b08c32-1323-11f1-9eb1-b07b25d4a036 \
  --env sawyer_hammer \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 3 \
  --output_dir training_videos \
  --all_checkpoints

python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_push_wall_2/b59a8f1a-1323-11f1-8bd1-ec2a72463c44 \
  --env sawyer_push_wall \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 3 \
  --output_dir training_videos \
  --all_checkpoints

python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_faucet_close_2/6cc2fe24-134d-11f1-bee6-b07b25d4aaa0 \
  --env sawyer_faucet_close \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 3 \
  --output_dir training_videos \
  --all_checkpoints

python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_push_back_2/54818ffa-1317-11f1-998f-1423f2e9bd40 \
  --env sawyer_push_back \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 1 \
  --output_dir training_videos \
  --all_checkpoints


python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_push_back_1/4c2830fc-1565-11f1-8349-b07b25d4aac2 \
  --env sawyer_push_back \
  --alg contrastive_cpc \
  --seed 1 \
  --num_episodes 1 \
  --output_dir training_videos \
  --all_checkpoints

python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_stick_pull_2/7ea2f9d2-134d-11f1-8cb1-b07b25d4a2f8 \
  --env sawyer_stick_pull \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 3 \
  --output_dir training_videos \
  --all_checkpoints


python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_handle_press_side_2/7eaa92aa-134d-11f1-8d47-b07b25d4a2f8 \
  --env sawyer_handle_press_side \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 3 \
  --output_dir training_videos \
  --all_checkpoints

python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_push_2/908ee48a-134d-11f1-a62c-ec2a7232f864 \
  --env sawyer_push \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 3 \
  --output_dir training_videos \
  --all_checkpoints

python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_shelf_place_2/908f8a2a-134d-11f1-b780-b07b25d4aac2 \
  --env sawyer_shelf_place \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 3 \
  --output_dir training_videos \
  --all_checkpoints

python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_window_close_2/ea0eec3e-134e-11f1-a111-b07b25d4aac2 \
  --env sawyer_window_close \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 3 \
  --output_dir training_videos \
  --all_checkpoints

python rollout_training_videos.py \
  --run_dir logs/contrastive_cpc_sawyer_peg_unplug_side_2/ea0eec8e-134e-11f1-be3b-b07b25d4aac2 \
  --env sawyer_peg_unplug_side \
  --alg contrastive_cpc \
  --seed 2 \
  --num_episodes 3 \
  --output_dir training_videos \
  --all_checkpoints
```

