#!/bin/bash
###############################################################################
# run_experiment.slurm - SLURM job script for running SGCRL experiments on GPU
#
# Usage:
#   sbatch run_experiment.slurm
#
# Or override defaults via environment variables:
#   sbatch --export=ENV=sawyer_box,ALG=c_learning,SEED=0 run_experiment.slurm
#
# Or pass as SLURM --export:
#   sbatch --export=ALL,ENV=sawyer_peg,ALG=contrastive_cpc run_experiment.slurm
###############################################################################

#SBATCH --job-name=sgcrl
#SBATCH --output=joblog/%x_%j_%A.out
#SBATCH --error=joblog/%x_%j_%A.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# ======================== CONFIGURABLE PARAMETERS ===========================
# These can be overridden via sbatch --export or environment variables

# Environment: sawyer_bin, sawyer_box, sawyer_peg, point_Spiral11x11
ENV="${ENV:-sawyer_bin}"

# Algorithm: contrastive_nce, contrastive_cpc, c_learning, nce+c_learning
ALG="${ALG:-contrastive_cpc}"

# Random seed
SEED="${SEED:-42}"

# Number of training steps
NUM_STEPS="${NUM_STEPS:-8000000}"

# Whether to sample goals uniformly (set to "true" for original Contrastive RL behavior)
SAMPLE_GOALS="${SAMPLE_GOALS:-false}"

# Log directory
LOG_DIR="${LOG_DIR:-logs/}"

# Whether to add unique id to log dir
ADD_UID="${ADD_UID:-true}"

# Path to MuJoCo
MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"

# ======================== MODULE LOADING ====================================
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================="
echo ""
echo "Configuration:"
echo "  Environment: $ENV"
echo "  Algorithm:   $ALG"
echo "  Seed:        $SEED"
echo "  Num steps:   $NUM_STEPS"
echo "  Sample goals: $SAMPLE_GOALS"
echo "  Log dir:     $LOG_DIR"
echo "============================================="

# Auto-detect the correct anaconda module name
ANACONDA_MODULE=""
for mod in anaconda3 anaconda anaconda/2023.07 anaconda/3; do
    if module avail "$mod" 2>&1 | grep -q "$mod"; then
        ANACONDA_MODULE="$mod"
        break
    fi
done
if [ -z "$ANACONDA_MODULE" ]; then
    echo "ERROR: Could not find an anaconda module."
    module avail 2>&1 | grep -i conda || true
    exit 1
fi
module load "$ANACONDA_MODULE"
module load cudatoolkit/11.3 cudnn/cuda-11.x/8.2.0

# ======================== ACTIVATE ENVIRONMENT ==============================
eval "$(conda shell.bash hook)"
conda activate contrastive_rl

echo "Python: $(python --version)"
echo "Conda env: $CONDA_PREFIX"

# ======================== SET LIBRARY PATHS =================================
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"

# Add CUDA library paths
if [ -d "/usr/local/cuda-11.3/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-11.3/lib64"
elif [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
fi

if [ -d "/usr/lib/nvidia" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/nvidia"
fi

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# ======================== VERIFY GPU ========================================
echo ""
echo "Checking GPU availability..."
python -c "
import jax
devices = jax.devices()
gpu_devices = [d for d in devices if d.platform == 'gpu']
print(f'JAX version: {jax.__version__}')
print(f'Devices: {devices}')
if not gpu_devices:
    print('ERROR: No GPU found!')
    exit(1)
print(f'Using {len(gpu_devices)} GPU(s)')
"

# ======================== CREATE LOG/JOB DIRECTORIES ========================
mkdir -p "${LOG_DIR}"
mkdir -p joblog

# ======================== BUILD COMMAND =====================================
CMD="python lp_contrastive.py"
CMD="${CMD} --env=${ENV}"
CMD="${CMD} --alg=${ALG}"
CMD="${CMD} --seed=${SEED}"
CMD="${CMD} --num_steps=${NUM_STEPS}"
CMD="${CMD} --log_dir_path=${LOG_DIR}"

if [ "${SAMPLE_GOALS}" = "true" ]; then
    CMD="${CMD} --sample_goals"
fi

if [ "${ADD_UID}" = "true" ]; then
    CMD="${CMD} --add_uid"
fi

echo ""
echo "Running command:"
echo "  ${CMD}"
echo ""

# ======================== RUN EXPERIMENT ====================================
cd /workspace
${CMD}

echo ""
echo "============================================="
echo "Job finished at: $(date)"
echo "============================================="
