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

# NYUAD HPC uses miniconda
ANACONDA_MODULE="${ANACONDA_MODULE:-miniconda/3-4.11.0}"
module load "$ANACONDA_MODULE"

# Try to load CUDA/cuDNN modules (names vary across HPC systems)
for cuda_mod in "cudatoolkit/11.3" "cuda/11.3" "cuda/11.3.1" "cuda11.3/toolkit" "cuda/11"; do
    if module avail "$cuda_mod" 2>&1 | grep -q "$cuda_mod"; then
        module load "$cuda_mod"; break
    fi
done
for cudnn_mod in "cudnn/cuda-11.x/8.2.0" "cudnn/8.2" "cudnn/8.2.0" "cudnn" "cudnn/cuda11"; do
    if module avail "$cudnn_mod" 2>&1 | grep -q "$cudnn_mod"; then
        module load "$cudnn_mod"; break
    fi
done

# ======================== ACTIVATE ENVIRONMENT ==============================
eval "$(conda shell.bash hook)"
conda activate contrastive_rl

echo "Python: $(python --version)"
echo "Conda env: $CONDA_PREFIX"

# ======================== SET LIBRARY PATHS =================================
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"

# Add CUDA library paths
for cuda_lib_dir in /usr/local/cuda-11.3/lib64 /usr/local/cuda-11/lib64 /usr/local/cuda/lib64; do
    if [ -d "$cuda_lib_dir" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${cuda_lib_dir}"; break
    fi
done
for nvidia_dir in /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu/nvidia; do
    if [ -d "$nvidia_dir" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${nvidia_dir}"; break
    fi
done
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
elif [ -n "$CUDA_PATH" ] && [ -d "$CUDA_PATH/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_PATH}/lib64"
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
cd /scratch/$(whoami)/sgcrl
${CMD}

echo ""
echo "============================================="
echo "Job finished at: $(date)"
echo "============================================="
