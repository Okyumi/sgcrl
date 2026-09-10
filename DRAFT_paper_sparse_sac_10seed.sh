#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=paper_sac
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --partition=l40s_public
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/paper_sparse_sac/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/paper_sparse_sac/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-4

# 20 matched runs: sparse SAC R/R and P/P on seeds 5..14, native_info
# wrapper, step-penalty reward r in {-1, 0}, tau=0.05. Four learners share
# each L40S. Each array task chains an afterany continuation before
# training; run_continual_sac.py resumes from the latest task_*.pkl.
#
#   sbatch DRAFT_paper_sparse_sac_10seed.sh
#   ARRAY=$(python scripts/paper_sparse_sac_status.py --incomplete-array-ids)
#   sbatch --array="$ARRAY" DRAFT_paper_sparse_sac_10seed.sh
set -euo pipefail

CHAIN_INDEX="${PAPER_SPARSE_SAC_CHAIN_INDEX:-0}"
MAX_CHAIN="${PAPER_SPARSE_SAC_MAX_CHAIN:-8}"
REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"

CONFIG_SCRIPT="experiment_configs_paper_sparse_sac_10seed.py"
CONFIG_INDEX_OFFSET=0
CONFIG_LIMIT=20
TASKS_PER_GPU="${TASKS_PER_GPU:-4}"
LOG_DIR="/scratch/yd2247/sgcrl/logs/paper_sparse_sac/10seed/runs"
CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/paper_sparse_sac_checkpoints/10seed"
export CHECKPOINT_DIR
export LOG_DIR
export WANDB_PROJECT="${WANDB_PROJECT:-continual_gcrl_paper}"
export WANDB_ENTITY="${WANDB_ENTITY:-nyuad_mmvc}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.22}"
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

export SCRATCH="${SCRATCH:-/scratch/$(whoami)}"
MINICONDA_ROOT="${MINICONDA_ROOT:-$SCRATCH/miniconda3}"
module purge 2>/dev/null || true
# shellcheck source=/dev/null
source "${MINICONDA_ROOT}/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
# shellcheck source=/dev/null
source "${REPO_DIR}/set_up/torch_hpc_env.sh"

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" \
    /scratch/yd2247/sgcrl/logs/paper_sparse_sac

python tests/test_paper_sparse_sac_10seed.py

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if python scripts/paper_sparse_sac_status.py \
    --array-task-complete \
    --array-task-id="$ARRAY_TASK_ID" \
    --tasks-per-gpu="$TASKS_PER_GPU" \
    --offset="$CONFIG_INDEX_OFFSET" \
    --limit="$CONFIG_LIMIT"; then
  echo "Array task ${ARRAY_TASK_ID} already complete. Skipping."
  exit 0
fi

NEXT_JOB=""
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$CHAIN_INDEX" -lt "$MAX_CHAIN" ]; then
  NEXT_CHAIN=$((CHAIN_INDEX + 1))
  NEXT_JOB="$(sbatch --parsable \
      --dependency="afterany:${SLURM_JOB_ID}" \
      --array="${ARRAY_TASK_ID}" \
      --export=ALL,PAPER_SPARSE_SAC_CHAIN_INDEX="${NEXT_CHAIN}",PAPER_SPARSE_SAC_MAX_CHAIN="${MAX_CHAIN}" \
      "$REPO_DIR/DRAFT_paper_sparse_sac_10seed.sh" || true)"
  if [ -n "$NEXT_JOB" ]; then
    echo "Scheduled continuation ${NEXT_JOB} (chain ${NEXT_CHAIN}/${MAX_CHAIN})."
  else
    echo "WARNING: failed to schedule a 48h continuation job." >&2
  fi
fi

bool_flag() {
  local name="$1"
  local value="$2"
  if [ "$value" = "true" ]; then
    echo "--${name}"
  else
    echo "--no${name}"
  fi
}

run_one() {
  local config_idx="$1"
  local slot="$2"
  eval "$(python "$CONFIG_SCRIPT" --setting "$config_idx")"
  local exp_log="${LOG_DIR}/${SLURM_ARRAY_JOB_ID:-local}_${ARRAY_TASK_ID}_${config_idx}"
  local flags=""
  flags="$flags --seed=$SEED --alg=sac_her"
  flags="$flags --num_tasks=$NUM_TASKS --steps_per_task=$STEPS_PER_TASK"
  flags="$flags --base_steps=$BASE_STEPS --eval_every=$EVAL_EVERY"
  flags="$flags --eval_episodes=$EVAL_EPISODES"
  flags="$flags --log_dir=$LOG_DIR --checkpoint_dir=$CHECKPOINT_DIR"
  flags="$flags --actor_mode=$ACTOR_MODE --critic_mode=$CRITIC_MODE"
  flags="$flags --her_reward_threshold=$HER_REWARD_THRESHOLD"
  flags="$flags --network_width=$NETWORK_WIDTH"
  flags="$flags --critic_depth=$CRITIC_DEPTH --actor_depth=$ACTOR_DEPTH"
  flags="$flags --sawyer_success_mode=$SAWYER_SUCCESS_MODE"
  flags="$flags --goal_conditioning_mode=$GOAL_CONDITIONING_MODE"
  flags="$flags --rl_metrics_occasional_multiplier=$RL_METRICS_OCCASIONAL_MULTIPLIER"
  flags="$flags --post_task_eval_scope=$POST_TASK_EVAL_SCOPE"
  flags="$flags --wandb_project=$WANDB_PROJECT --wandb_group=$WANDB_GROUP"
  flags="$flags --wandb_entity=$WANDB_ENTITY --wandb_mode=online"
  flags="$flags $(bool_flag use_wandb "$USE_WANDB")"
  flags="$flags $(bool_flag auto_resume "$AUTO_RESUME")"
  flags="$flags $(bool_flag use_task_id "$USE_TASK_ID")"
  flags="$flags $(bool_flag step_penalty_reward "$STEP_PENALTY_REWARD")"
  flags="$flags $(bool_flag log_rl_metrics "$LOG_RL_METRICS")"
  flags="$flags $(bool_flag use_residual "$USE_RESIDUAL")"
  flags="$flags $(bool_flag actor_auto_reset "$ACTOR_AUTO_RESET")"
  flags="$flags $(bool_flag intra_eval_previous_tasks "$INTRA_EVAL_PREVIOUS")"
  flags="$flags --nouse_20_tasks --noencoder_from_base --add_uid"

  echo "[slot $slot] config $config_idx actor=$ACTOR_MODE critic=$CRITIC_MODE seed=$SEED"
  python -u run_continual_sac.py $flags \
      > "${exp_log}.out" 2> "${exp_log}.err"
}

echo "Paper sparse SAC chain index: ${CHAIN_INDEX}/${MAX_CHAIN}"
PIDS=()
for ((i = 0; i < TASKS_PER_GPU; i++)); do
  CONFIG_IDX=$(( CONFIG_INDEX_OFFSET + TASKS_PER_GPU * ARRAY_TASK_ID + i ))
  if [ "$CONFIG_IDX" -ge $(( CONFIG_INDEX_OFFSET + CONFIG_LIMIT )) ]; then
    continue
  fi
  run_one "$CONFIG_IDX" "$i" &
  PIDS+=($!)
done

if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "No configs for array task ${ARRAY_TASK_ID}."
  exit 0
fi
wait "${PIDS[@]}"

if [ -n "$NEXT_JOB" ] && python scripts/paper_sparse_sac_status.py \
    --array-task-complete \
    --array-task-id="$ARRAY_TASK_ID" \
    --tasks-per-gpu="$TASKS_PER_GPU" \
    --offset="$CONFIG_INDEX_OFFSET" \
    --limit="$CONFIG_LIMIT"; then
  echo "Array task ${ARRAY_TASK_ID} finished the curriculum; cancelling ${NEXT_JOB}."
  scancel "$NEXT_JOB" || true
fi
