#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=dcc_scale_t5_prefix
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/dcc_shared_scale/prefix_%A.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/dcc_shared_scale/prefix_%A.err
#SBATCH --mail-user=yd2247@nyu.edu

# Train Tasks 0-4 once for each seed. The alpha sweep reuses these exact
# checkpoints and trains only Task 5.
set -euo pipefail

REPO_DIR="/scratch/yd2247/sgcrl"
CONFIG_SCRIPT="experiment_configs_dcc_shared_scale_task5.py"
PREFIX_ROOT="/scratch/yd2247/sgcrl/logs/dcc_shared_scale/task5_prefix5"
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

mkdir -p /scratch/yd2247/sgcrl/logs/dcc_shared_scale
python tests/test_dcc_shared_scale.py
mkdir -p "$PREFIX_ROOT/runs" "$PREFIX_ROOT/checkpoints"

run_one() {
  local setting="$1"
  eval "$(python "$CONFIG_SCRIPT" --phase prefix --setting "$setting")"
  python run_continual_contrastive.py \
    --seed="$SEED" \
    --alg=contrastive_cpc \
    --num_tasks="$NUM_TASKS" \
    --steps_per_task="$STEPS_PER_TASK" \
    --base_steps="$BASE_STEPS" \
    --eval_every="$EVAL_EVERY" \
    --eval_episodes="$EVAL_EPISODES" \
    --log_dir="$PREFIX_ROOT/runs/seed${SEED}" \
    --checkpoint_dir="$PREFIX_ROOT/checkpoints" \
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
    --phi_task_width="$PHI_TASK_WIDTH" \
    --phi_task_depth="$PHI_TASK_DEPTH" \
    --combine_mode="$COMBINE_MODE" \
    --sawyer_success_mode="$SAWYER_SUCCESS_MODE" \
    --goal_conditioning_mode="$GOAL_CONDITIONING_MODE" \
    --noactor_auto_reset \
    --nouse_20_tasks \
    --use_residual \
    --profile_runtime \
    --nointra_eval_previous_tasks \
    --nolog_rl_metrics \
    --post_task_eval_scope=current
}

run_one 0 &
pid0=$!
run_one 1 &
pid1=$!
run_one 2 &
pid2=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
exit "$status"
