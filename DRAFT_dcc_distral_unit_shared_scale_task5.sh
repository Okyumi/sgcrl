#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=dcc_distral_unit_t5
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/dcc_distral_unit_shared_scale/task5_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/dcc_distral_unit_shared_scale/task5_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-5

# Primary Distral-faithful Task-5 test. The shared, task, and goal embeddings
# are unit-normalized, but the score remains alpha*f_shared + f_task without
# division by alpha+1. Each array task runs one alpha and seeds 5/6/7.
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
CONFIG_SCRIPT="experiment_configs_dcc_distral_unit_shared_scale_task5.py"
cd "$REPO_DIR"

export MUJOCO_GL=egl
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONNOUSERSITE=1
export TF_CPP_MIN_LOG_LEVEL=2
export XDG_CACHE_HOME=/scratch/yd2247/.cache
export PIP_CACHE_DIR=/scratch/yd2247/.cache/pip
export TMPDIR=/scratch/yd2247/tmp
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

export MKL_INTERFACE_LAYER=LP64,GNU
module purge 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
export PATH="${CONDA_PREFIX}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
source "$REPO_DIR/set_up/torch_hpc_env.sh"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

mkdir -p /scratch/yd2247/sgcrl/logs/dcc_distral_unit_shared_scale
python tests/test_dcc_distral_unit_shared_scale.py

run_one() {
  local setting="$1"
  eval "$(python "$CONFIG_SCRIPT" --setting "$setting")"
  if [ ! -d "$RESUME_CHECKPOINT_DIR" ]; then
    echo "Missing Tasks-0-to-4 prefix checkpoints: $RESUME_CHECKPOINT_DIR" >&2
    return 1
  fi
  local scale_tag
  scale_tag=$(printf '%g' "$SHARED_REPR_SCALE" | tr '.' 'p')
  local run_root="/scratch/yd2247/sgcrl/logs/dcc_distral_unit_shared_scale/task5/a${scale_tag}/seed${SEED}"
  mkdir -p "$run_root" "$run_root/checkpoints"
  python run_continual_contrastive.py \
    --seed="$SEED" \
    --alg=contrastive_cpc \
    --num_tasks="$NUM_TASKS" \
    --steps_per_task="$STEPS_PER_TASK" \
    --base_steps="$BASE_STEPS" \
    --start_task="$START_TASK" \
    --resume_checkpoint_dir="$RESUME_CHECKPOINT_DIR" \
    --eval_every="$EVAL_EVERY" \
    --eval_episodes="$EVAL_EPISODES" \
    --log_dir="$run_root/runs" \
    --checkpoint_dir="$run_root/checkpoints" \
    --use_wandb \
    --wandb_project=continual-contrastive-rl \
    --wandb_group="$WANDB_GROUP" \
    --critic_mode="$CRITIC_MODE" \
    --actor_mode="$ACTOR_MODE" \
    --nouse_task_id \
    --network_width="$NETWORK_WIDTH" \
    --critic_depth="$CRITIC_DEPTH" \
    --actor_depth="$ACTOR_DEPTH" \
    --energy_fn="$ENERGY_FN" \
    --dyn_aux_weight="$DYN_AUX_WEIGHT" \
    --shared_repr_scale="$SHARED_REPR_SCALE" \
    --shared_repr_normalization="$SHARED_REPR_NORMALIZATION" \
    --phi_task_width="$PHI_TASK_WIDTH" \
    --phi_task_depth="$PHI_TASK_DEPTH" \
    --combine_mode="$COMBINE_MODE" \
    --sawyer_success_mode="$SAWYER_SUCCESS_MODE" \
    --goal_conditioning_mode="$GOAL_CONDITIONING_MODE" \
    --noactor_auto_reset \
    --in_trajectory_negative_repeats="$IN_TRAJECTORY_NEGATIVE_REPEATS" \
    --nointeraction_weighted_relabeling \
    --noaction_effect_enabled \
    --success_bc_weight="$SUCCESS_BC_WEIGHT" \
    --counterfactual_rank_interval_steps="$COUNTERFACTUAL_RANK_INTERVAL_STEPS" \
    --counterfactual_oracle_interval_steps="$COUNTERFACTUAL_ORACLE_INTERVAL_STEPS" \
    --action_landscape_diagnostic_interval_steps="$ACTION_LANDSCAPE_DIAGNOSTIC_INTERVAL_STEPS" \
    --shortcut_diagnostic_interval="$SHORTCUT_DIAGNOSTIC_INTERVAL" \
    --nouse_20_tasks \
    --use_residual \
    --profile_runtime \
    --nointra_eval_previous_tasks \
    --nolog_rl_metrics \
    --post_task_eval_scope=current
}

base_setting=$((SLURM_ARRAY_TASK_ID * 3))
run_one "$base_setting" &
pid0=$!
run_one "$((base_setting + 1))" &
pid1=$!
run_one "$((base_setting + 2))" &
pid2=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
exit "$status"
