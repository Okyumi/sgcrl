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

# Redirect caches to /scratch to avoid home quota issues
export PIP_CACHE_DIR="/scratch/$(whoami)/.cache/pip"
export CONDA_PKGS_DIRS="/scratch/$(whoami)/.cache/conda/pkgs"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" 2>/dev/null || true

echo "============================================="
echo "GPU Setup for SGCRL"
echo "============================================="

# ======================== STEP 1: Load Modules ==============================
echo ""
echo "Step 1: Loading modules..."

ANACONDA_MODULE="${ANACONDA_MODULE:-miniconda/3-4.11.0}"
echo "Loading: $ANACONDA_MODULE"
module load "$ANACONDA_MODULE"

echo "Loading: cuda/11.8.0"
module load cuda/11.8.0

echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc: $(nvcc --version 2>/dev/null | grep release || echo 'not found')"

# Verify GPU is accessible (nvidia-smi needs CUDA module loaded first on some HPC systems)
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &>/dev/null; then
    echo "GPU(s) detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found even after loading CUDA module."
    echo "Continuing anyway -- GPU may still be usable by JAX."
fi

# ======================== STEP 2: Activate Conda Env ========================
echo ""
echo "Step 2: Activating conda environment..."

eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "Active environment: $CONDA_PREFIX"

# ======================== STEP 3: Install cuDNN via pip =====================
echo ""
echo "Step 3: Installing cuDNN via pip (no cuDNN HPC module available)..."

# nvidia-cudnn-cu11 provides cuDNN 8.x libraries for CUDA 11
pip install nvidia-cudnn-cu11==8.6.0.163

# ======================== STEP 4: Set Library Paths =========================
echo ""
echo "Step 4: Setting library paths..."

# Find where pip installed the cuDNN libraries
CUDNN_LIB=$(python -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null || echo "")

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"

# Add cuDNN lib path
if [ -n "$CUDNN_LIB" ] && [ -d "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH}"
    echo "Added cuDNN lib: $CUDNN_LIB"
else
    echo "WARNING: Could not locate nvidia.cudnn pip package lib dir"
fi

# Add CUDA lib from the loaded module
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
    echo "Added CUDA_HOME lib: $CUDA_HOME/lib64"
fi

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Verify cuDNN is findable
echo ""
echo "Checking cuDNN library..."
python -c "
import ctypes
try:
    cudnn = ctypes.cdll.LoadLibrary('libcudnn.so.8')
    print('libcudnn.so.8 loaded successfully')
except OSError as e:
    print(f'WARNING: Cannot load libcudnn.so.8: {e}')
"

# ======================== STEP 5: Install GPU JAX ===========================
echo ""
echo "Step 5: Installing GPU-compatible JAX, optax, and pinning dependencies..."

# IMPORTANT: Install jax+jaxlib FIRST with --no-deps to prevent pip from
# pulling in jax 0.4.30 via transitive dependencies of optax/chex.
pip install --no-deps "jax==0.4.7" "jaxlib==0.4.7+cuda11.cudnn86" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Pin ALL jax-ecosystem packages to versions compatible with jax 0.4.7
pip install --no-deps chex==0.1.7
pip install --no-deps optax==0.1.7
pip install --no-deps dm-haiku==0.0.9
pip install --no-deps distrax==0.1.3
pip install --no-deps flax==0.6.11
pip install --no-deps orbax-checkpoint==0.2.3
pip install ml_dtypes==0.2.0

echo ""
echo "============================================="
echo "Step 6: Verifying GPU access from JAX..."
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
    print('ERROR: No GPU devices found!')
    print('')
    print('Debug info:')
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(result.stdout[:500] if result.stdout else 'nvidia-smi failed')
    import os
    print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', 'not set'))
    print('')
    print('Try running with more verbose output:')
    print('  TF_CPP_MIN_LOG_LEVEL=0 python -c \"import jax; print(jax.devices())\"')
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
