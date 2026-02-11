#!/bin/bash
###############################################################################
# setup_env.sh - Complete environment setup for SGCRL on NYUAD HPC
#
# This script automates the conda environment creation and package installation
# as described in the README. Run this from the repository root directory.
#
# Prerequisites:
#   - MuJoCo 210 already downloaded and placed in ~/.mujoco/mujoco210/
#   - Access to NYUAD HPC with module system
#
# Usage:
#   chmod +x setup_env.sh
#   bash setup_env.sh
###############################################################################

set -e  # Exit on any error

# ======================== CONFIGURABLE PATHS ================================
# Adjust these if your paths differ from the defaults

# Path to your conda installation (auto-detected after module load)
CONDA_BASE="${CONDA_BASE:-}"

# Path to MuJoCo installation
MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"

# Environment name
ENV_NAME="contrastive_rl"

# ======================== REDIRECT CACHES TO SCRATCH ========================
# Avoid filling home directory quota with large pip/conda caches
export PIP_CACHE_DIR="/scratch/$(whoami)/.cache/pip"
export CONDA_PKGS_DIRS="/scratch/$(whoami)/.cache/conda/pkgs"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" 2>/dev/null || true
echo "Pip cache: $PIP_CACHE_DIR"
echo "Conda cache: $CONDA_PKGS_DIRS"

# ======================== STEP 1: Load Modules ==============================
echo "============================================="
echo "Step 1: Loading required modules..."
echo "============================================="

# NYUAD HPC uses miniconda (override with ANACONDA_MODULE env var if needed)
ANACONDA_MODULE="${ANACONDA_MODULE:-miniconda/3-4.11.0}"
echo "Loading module: $ANACONDA_MODULE"
module load "$ANACONDA_MODULE"

# Capture conda base path after loading
if [ -z "$CONDA_BASE" ]; then
    CONDA_BASE="$(conda info --base)"
fi
echo "Conda base: $CONDA_BASE"

# ======================== STEP 1b: Clean corrupted conda cache ==============
echo ""
echo "Cleaning corrupted conda package cache..."
conda clean --packages --tarballs -y 2>/dev/null || true
rm -rf "$HOME/.conda/pkgs/wheel-0.45.1-py39h06a4308_0" 2>/dev/null || true

# ======================== STEP 2: Create Conda Environment ==================
echo ""
echo "============================================="
echo "Step 2: Creating conda environment '${ENV_NAME}' with Python 3.9..."
echo "============================================="

# Check if environment already exists (check both conda list and directory)
if conda env list | grep -q "${ENV_NAME}" || [ -d "$HOME/.conda/envs/${ENV_NAME}" ]; then
    echo "Environment '${ENV_NAME}' already exists."
    echo "Removing and recreating it..."
    conda deactivate 2>/dev/null || true
    conda env remove -n ${ENV_NAME} -y 2>/dev/null || true
    # Also remove the directory if conda env remove didn't clean it
    rm -rf "$HOME/.conda/envs/${ENV_NAME}" 2>/dev/null || true
    conda create -n ${ENV_NAME} python=3.9 -y
else
    conda create -n ${ENV_NAME} python=3.9 -y
fi

# ======================== STEP 3: Activate Environment ======================
echo ""
echo "============================================="
echo "Step 3: Activating environment..."
echo "============================================="

# Initialize conda for this shell if needed
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "Active environment: $CONDA_PREFIX"
echo "Python: $(python --version)"

# ======================== STEP 4: Set Library Paths =========================
echo ""
echo "============================================="
echo "Step 4: Setting LD_LIBRARY_PATH..."
echo "============================================="

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/"
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

# Add MuJoCo to library path and set MUJOCO_PATH for pip builds
if [ -d "${MUJOCO_DIR}/bin" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"
    export MUJOCO_PATH="${MUJOCO_DIR}"
    echo "Added MuJoCo bin to LD_LIBRARY_PATH: ${MUJOCO_DIR}/bin"
    echo "MUJOCO_PATH set to: ${MUJOCO_DIR}"
else
    echo "ERROR: MuJoCo directory not found at ${MUJOCO_DIR}"
    echo "Please set MUJOCO_DIR to your MuJoCo installation path, e.g.:"
    echo "  MUJOCO_DIR=\$HOME/.mujoco/mujoco210 bash setup_env.sh"
    exit 1
fi

# ======================== STEP 5: Pre-install Cython fix ====================
echo ""
echo "============================================="
echo "Step 5: Pre-installing Cython < 3 (prevents build errors)..."
echo "============================================="

pip install "cython<3"

# ======================== STEP 6: Install GL dependencies ===================
echo ""
echo "============================================="
echo "Step 6: Installing GL/rendering dependencies via conda..."
echo "============================================="

conda install -c conda-forge glew mesalib -y
conda install -c menpo glfw3 -y
pip install patchelf

# ======================== STEP 7: Install requirements ======================
echo ""
echo "============================================="
echo "Step 7: Installing pip requirements (--no-deps)..."
echo "============================================="

# Filter out 'mujoco' (the newer DeepMind binding) from requirements.
# The codebase uses mujoco-py (old binding) via metaworld, not the mujoco package.
# mujoco==2.2.0 requires MuJoCo 2.2.0 binaries; we only have MuJoCo 210 for mujoco-py.
grep -v '^mujoco ' requirements.txt | grep -v '^mujoco==' > /tmp/requirements_filtered.txt
pip install -r /tmp/requirements_filtered.txt --no-deps
rm -f /tmp/requirements_filtered.txt

# ======================== STEP 8: Install strict version overrides ==========
echo ""
echo "============================================="
echo "Step 8: Installing strict package version overrides..."
echo "============================================="

# dm-acme with jax and tf extras (but constrain transitive deps)
pip install "dm-acme[jax,tf]"

# Pin jax-ecosystem packages to mutually compatible versions.
# setup_gpu.sh will later replace jax/jaxlib with GPU builds,
# but we need valid versions now to avoid broken transitive deps.
pip install --no-deps chex==0.1.7
pip install --no-deps optax==0.1.7
pip install --no-deps dm-haiku==0.0.9
pip install --no-deps distrax==0.1.3
pip install --no-deps flax==0.6.11
pip install --no-deps orbax-checkpoint==0.2.3
pip install ml_dtypes==0.2.0

# gymnasium-robotics (for additional env support)
pip install gymnasium-robotics

# scipy - uninstall first then install specific version
pip uninstall scipy -y
pip install scipy==1.12

# torch CPU-only (only used for tensorboard SummaryWriter, no GPU needed)
# This avoids downloading ~2GB of bundled nvidia-cu12 libraries
pip install torch==2.1.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-learn pandas

echo ""
echo "============================================="
echo "Step 8 complete. CPU-only setup is done."
echo "============================================="

echo ""
echo "============================================================"
echo "  SETUP COMPLETE (CPU)"
echo "============================================================"
echo ""
echo "To set up GPU support, run:"
echo "  bash setup_gpu.sh"
echo ""
echo "To activate this environment in future sessions:"
echo "  module load $ANACONDA_MODULE"
echo "  conda activate ${ENV_NAME}"
echo "  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib/:${MUJOCO_DIR}/bin"
echo ""
