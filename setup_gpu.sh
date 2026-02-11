#!/bin/bash
###############################################################################
# setup_gpu.sh - GPU-specific setup for SGCRL on NYUAD HPC
#
# IMPORTANT: This script must be run from a GPU node (e.g., inside an
# interactive GPU session or a SLURM GPU job).
#
# Usage:
#   # First, get a GPU interactive session:
#   srun --gres=gpu:1 --pty bash
#   # Then run:
#   bash setup_gpu.sh
###############################################################################

set -e

ENV_NAME="contrastive_rl"
MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"

echo "============================================="
echo "GPU Setup for SGCRL"
echo "============================================="

# ======================== STEP 1: Load Modules ==============================
echo ""
echo "Step 1: Loading modules (anaconda, CUDA, cuDNN)..."

# NYUAD HPC uses miniconda
ANACONDA_MODULE="${ANACONDA_MODULE:-miniconda/3-4.11.0}"
echo "Loading: $ANACONDA_MODULE"
module load "$ANACONDA_MODULE"

# Try to load CUDA/cuDNN modules - names vary across HPC systems
CUDA_LOADED=false
for cuda_mod in "cudatoolkit/11.3" "cuda/11.3" "cuda/11.3.1" "cuda11.3/toolkit" "cuda/11"; do
    if module avail "$cuda_mod" 2>&1 | grep -q "$cuda_mod"; then
        echo "Loading CUDA module: $cuda_mod"
        module load "$cuda_mod"
        CUDA_LOADED=true
        break
    fi
done

if [ "$CUDA_LOADED" = false ]; then
    echo "No CUDA module found via 'module load'. Checking if CUDA is already available..."
    if command -v nvcc &>/dev/null; then
        echo "nvcc found: $(nvcc --version | grep release)"
    elif [ -d "/usr/local/cuda" ]; then
        echo "Found CUDA at /usr/local/cuda"
    else
        echo ""
        echo "WARNING: No CUDA module or installation detected."
        echo "Available CUDA-related modules on this system:"
        module avail 2>&1 | grep -iE "cuda|cudnn|gpu|nvidia" || echo "  (none found)"
        echo ""
        echo "If you know the correct module name, re-run with:"
        echo "  CUDA_MODULE=<name> CUDNN_MODULE=<name> bash setup_gpu.sh"
        echo ""
        echo "Continuing anyway -- JAX install may still work if CUDA libs are in the default path."
    fi
fi

# Try cuDNN module
for cudnn_mod in "cudnn/cuda-11.x/8.2.0" "cudnn/8.2" "cudnn/8.2.0" "cudnn" "cudnn/cuda11"; do
    if module avail "$cudnn_mod" 2>&1 | grep -q "$cudnn_mod"; then
        echo "Loading cuDNN module: $cudnn_mod"
        module load "$cudnn_mod"
        break
    fi
done

echo "Module loading done."

# ======================== STEP 2: Activate Conda Env ========================
echo ""
echo "Step 2: Activating conda environment..."

eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "Active environment: $CONDA_PREFIX"

# ======================== STEP 3: Set Library Paths =========================
echo ""
echo "Step 3: Setting library paths..."

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"

# Add CUDA library paths - check common locations
for cuda_lib_dir in /usr/local/cuda-11.3/lib64 /usr/local/cuda-11/lib64 /usr/local/cuda/lib64; do
    if [ -d "$cuda_lib_dir" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${cuda_lib_dir}"
        echo "Added CUDA lib: $cuda_lib_dir"
        break
    fi
done

for nvidia_dir in /usr/lib/nvidia /usr/lib64/nvidia /usr/lib/x86_64-linux-gnu/nvidia; do
    if [ -d "$nvidia_dir" ]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${nvidia_dir}"
        echo "Added NVIDIA lib: $nvidia_dir"
        break
    fi
done

# Also check if CUDA_HOME or CUDA_PATH is set by the module
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
    echo "Added CUDA_HOME lib: $CUDA_HOME/lib64"
elif [ -n "$CUDA_PATH" ] && [ -d "$CUDA_PATH/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_PATH}/lib64"
    echo "Added CUDA_PATH lib: $CUDA_PATH/lib64"
fi

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# ======================== STEP 4: Install GPU JAX ===========================
echo ""
echo "Step 4: Installing GPU-compatible JAX and optax..."

# Install optax version compatible with this JAX
pip install optax==0.1.7

# Install JAX with CUDA 11 + cuDNN 8.2 support
pip install --upgrade "jax==0.4.7" "jaxlib==0.4.7+cuda11.cudnn82" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ""
echo "============================================="
echo "Step 5: Verifying GPU access from JAX..."
echo "============================================="

python -c "
import jax
print('JAX version:', jax.__version__)
devices = jax.devices()
print('Available devices:', devices)
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print('GPU devices found:', len(gpu_devices))
    for d in gpu_devices:
        print('  -', d)
    print('GPU setup successful!')
else:
    print('WARNING: No GPU devices found. Check CUDA/cuDNN installation.')
"

echo ""
echo "============================================================"
echo "  GPU SETUP COMPLETE"
echo "============================================================"
echo ""
echo "You can now run experiments with GPU acceleration."
echo "Use the SLURM scripts to submit jobs:"
echo "  sbatch run_experiment.slurm"
echo "  bash run_all_experiments.sh"
echo ""
