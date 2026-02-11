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

# Activate conda
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Set library paths
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"

# CUDA paths
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
