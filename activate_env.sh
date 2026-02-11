#!/bin/bash
###############################################################################
# activate_env.sh - Source this to activate the environment for interactive use
#
# Usage:
#   source activate_env.sh
#
# This loads the required modules, activates conda, and sets library paths.
# Use this when running experiments interactively or debugging.
###############################################################################

MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"
ENV_NAME="contrastive_rl"

# NYUAD HPC uses miniconda
ANACONDA_MODULE="${ANACONDA_MODULE:-miniconda/3-4.11.0}"
module load "$ANACONDA_MODULE"
module load cudatoolkit/11.3 cudnn/cuda-11.x/8.2.0

# Activate conda
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Set library paths
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"

# CUDA paths
if [ -d "/usr/local/cuda-11.3/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-11.3/lib64"
elif [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
fi

if [ -d "/usr/lib/nvidia" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/nvidia"
fi

echo "Environment '${ENV_NAME}' activated."
echo "Python: $(python --version)"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

# Verify JAX GPU
python -c "
import jax
devices = jax.devices()
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f'JAX {jax.__version__} with {len(gpu_devices)} GPU(s): {gpu_devices}')
else:
    print(f'JAX {jax.__version__} (CPU only - no GPU detected)')
" 2>/dev/null || echo "JAX not yet installed or import error."
