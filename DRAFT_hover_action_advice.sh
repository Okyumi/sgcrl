#!/bin/bash
#SBATCH --job-name=hover_action
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/task58_success_propagation/hover_action_%j.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/task58_success_propagation/hover_action_%j.err

# Offline same-state action-advice probe (no training). Short job meant to
# jump ahead of long paper arrays once a GPU frees.

set -euo pipefail
REPO_DIR=/scratch/yd2247/sgcrl
cd "$REPO_DIR"
source /scratch/yd2247/miniconda3/etc/profile.d/conda.sh
conda activate contrastive_rl
source set_up/torch_hpc_env.sh
export MUJOCO_GL=egl

OUT=/scratch/yd2247/sgcrl/logs/task58_success_propagation
HANDLE_PEAK=$OUT/checkpoints/actor_reset_critic_decomposed_tid_False_heads_False_success_corrected_dyn1.000_pt256x4_env_sawyer_handle_press_side/seed_6/task_0_step_100200.pkl
HANDLE_LATE=$OUT/checkpoints/actor_reset_critic_decomposed_tid_False_heads_False_success_corrected_dyn1.000_pt256x4_env_sawyer_handle_press_side/seed_6/task_0_step_951900.pkl
PUSH_PEAK=$OUT/checkpoints/actor_reset_critic_decomposed_tid_False_heads_False_success_corrected_dyn1.000_pt256x4_env_sawyer_push/seed_6/task_0_step_250500.pkl

python scripts/measure_hover_action_advice.py \
  --checkpoint "$HANDLE_PEAK" \
  --env-name sawyer_handle_press_side --episodes 50 --seed 6 \
  --output "$OUT/action_advice_handle_s6_100k.json"

python scripts/measure_hover_action_advice.py \
  --checkpoint "$HANDLE_LATE" \
  --env-name sawyer_handle_press_side --episodes 40 --seed 6 \
  --output "$OUT/action_advice_handle_s6_late.json"

python scripts/measure_hover_action_advice.py \
  --checkpoint "$PUSH_PEAK" \
  --env-name sawyer_push --episodes 40 --seed 6 \
  --output "$OUT/action_advice_push_s6_250k.json"

echo "done"
