#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=dcc_distral_c7
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/dcc_distral_unit_continual/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/dcc_distral_unit_continual/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=0-5

# From-scratch seven-task unit_distral alpha pilot. One array element is one
# alpha. That element runs the selected seeds in parallel on one L40S.
# Default seeds are 5 and 6. Create the log directory before submitting:
#   mkdir -p /scratch/yd2247/sgcrl/logs/dcc_distral_unit_continual
#   sbatch DRAFT_dcc_distral_unit_shared_scale_continual.sh
set -euo pipefail

CHAIN_INDEX="${DISTRAL_CHAIN_INDEX:-0}"
MAX_CHAIN="${DISTRAL_MAX_CHAIN:-1}"
REPO_DIR="/scratch/yd2247/sgcrl"
CONFIG_SCRIPT="experiment_configs_dcc_distral_unit_shared_scale_continual.py"
SLURM_LOG_DIR="/scratch/yd2247/sgcrl/logs/dcc_distral_unit_continual"
cd "$REPO_DIR"

export MUJOCO_GL=egl
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONNOUSERSITE=1
export TF_CPP_MIN_LOG_LEVEL=2
export XDG_CACHE_HOME=/scratch/yd2247/.cache
export PIP_CACHE_DIR=/scratch/yd2247/.cache/pip
export TMPDIR=/scratch/yd2247/tmp
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$SLURM_LOG_DIR"

export MKL_INTERFACE_LAYER=LP64,GNU
export SCRATCH="${SCRATCH:-/scratch/$(whoami)}"
MINICONDA_ROOT="${MINICONDA_ROOT:-$SCRATCH/miniconda3}"
module purge 2>/dev/null || true
# shellcheck source=/dev/null
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
export PATH="${CONDA_PREFIX}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# shellcheck source=/dev/null
source "$REPO_DIR/set_up/torch_hpc_env.sh"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.wandb_api_key" ]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_api_key")"
  export WANDB_API_KEY
fi

if [ "$CHAIN_INDEX" -eq 0 ]; then
  python tests/test_dcc_distral_unit_shared_scale.py
  python tests/test_dcc_distral_unit_shared_scale_continual.py
fi

n_alphas="$(python "$CONFIG_SCRIPT" --num-alphas)"
n_seeds="$(python "$CONFIG_SCRIPT" --seeds-per-alpha)"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$ARRAY_TASK_ID" -lt 0 ] || [ "$ARRAY_TASK_ID" -ge "$n_alphas" ]; then
  echo "Array id ${ARRAY_TASK_ID} is outside 0-$((n_alphas - 1))." >&2
  exit 1
fi
base_setting=$((ARRAY_TASK_ID * n_seeds))

array_complete() {
  local offset
  for ((offset=0; offset<n_seeds; offset++)); do
    if ! python "$CONFIG_SCRIPT" --complete --setting "$((base_setting + offset))"; then
      return 1
    fi
  done
}

if array_complete; then
  echo "Array task ${ARRAY_TASK_ID} already finished every seed."
  exit 0
fi

NEXT_JOB=""
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$CHAIN_INDEX" -lt "$MAX_CHAIN" ]; then
  NEXT_CHAIN=$((CHAIN_INDEX + 1))
  NEXT_JOB="$(sbatch --parsable \
      --dependency="afterany:${SLURM_JOB_ID}" \
      --array="${ARRAY_TASK_ID}" \
      --export=ALL,DISTRAL_CHAIN_INDEX="${NEXT_CHAIN}",DISTRAL_MAX_CHAIN="${MAX_CHAIN}" \
      "$REPO_DIR/DRAFT_dcc_distral_unit_shared_scale_continual.sh" || true)"
  if [ -n "$NEXT_JOB" ]; then
    echo "Scheduled continuation ${NEXT_JOB} (chain ${NEXT_CHAIN}/${MAX_CHAIN})."
  else
    echo "WARNING: failed to schedule a 48h continuation job." >&2
  fi
fi

run_one() {
  local setting="$1"
  if python "$CONFIG_SCRIPT" --complete --setting "$setting"; then
    echo "Setting ${setting} already has its final checkpoint."
    return 0
  fi
  eval "$(python "$CONFIG_SCRIPT" --setting "$setting")"
  mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
  echo "Starting ${RUN_NAME} (alpha=${SHARED_REPR_SCALE}, seed=${SEED})."
  python run_continual_contrastive.py \
    --seed="$SEED" \
    --alg=contrastive_cpc \
    --num_tasks="$NUM_TASKS" \
    --steps_per_task="$STEPS_PER_TASK" \
    --base_steps="$BASE_STEPS" \
    --eval_every="$EVAL_EVERY" \
    --eval_episodes="$EVAL_EPISODES" \
    --log_dir="$LOG_DIR" \
    --checkpoint_dir="$CHECKPOINT_DIR" \
    --use_wandb \
    --wandb_project="$WANDB_PROJECT" \
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
    --post_task_eval_scope="$POST_TASK_EVAL_SCOPE"
}

echo "Distral continual chain index: ${CHAIN_INDEX}/${MAX_CHAIN}"
pids=()
for ((offset=0; offset<n_seeds; offset++)); do
  run_one "$((base_setting + offset))" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

if [ -n "$NEXT_JOB" ] && array_complete; then
  echo "Array task ${ARRAY_TASK_ID} finished the curriculum; cancelling ${NEXT_JOB}."
  scancel "$NEXT_JOB" || true
fi
exit "$status"
