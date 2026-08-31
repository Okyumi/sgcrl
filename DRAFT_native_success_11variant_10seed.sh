#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=native_11var_10seed
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/native_success/11var_%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/native_success/11var_%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-36

# 110 matched runs: nine actor/critic reset-persistent-CKA combinations plus
# DCC dynamics-on and dynamics-off, all on seeds 5..14. Three runs share each
# GPU; the final array element contains the remaining two runs.
set -euo pipefail

if [ "${NATIVE_SUCCESS_11VARIANT_PROMOTED:-false}" != "true" ]; then
  echo "Task-5/8 DCC comparison and ten-task wrapper smoke must pass first." >&2
  exit 2
fi

REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"
export NATIVE_SUCCESS_11VARIANT_PROMOTED
export CONFIG_SCRIPT="experiment_configs_native_success_11variant_10seed.py"
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=110
export TASKS_PER_GPU=3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export LOG_DIR="/scratch/yd2247/sgcrl/logs/native_success/11variant_10seed/runs"
export CHECKPOINT_DIR="/scratch/yd2247/sgcrl/logs/native_success_checkpoints/11variant_10seed"

python tests/test_native_success_11variant_configs.py
python tests/test_sawyer_native_success_wrapper.py
exec bash "$REPO_DIR/DRAFT.sh"
