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

# ======================== STEP 1: Load Modules ==============================
echo "============================================="
echo "Step 1: Loading required modules..."
echo "============================================="

# Auto-detect the correct anaconda module name on this HPC
ANACONDA_MODULE=""
for mod in anaconda3 anaconda anaconda/2023.07 anaconda/3 miniconda/3-4.11.0 miniconda-nobashrc/3-4.11.0; do
    if module avail "$mod" 2>&1 | grep -q "$mod"; then
        ANACONDA_MODULE="$mod"
        break
    fi
done

if [ -z "$ANACONDA_MODULE" ]; then
    echo "ERROR: Could not find an anaconda module. Available modules:"
    module avail 2>&1 | grep -i conda || true
    echo "Please set ANACONDA_MODULE and re-run, e.g.:"
    echo "  ANACONDA_MODULE=anaconda/2023.07 bash setup_env.sh"
    exit 1
fi

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

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        conda deactivate 2>/dev/null || true
        conda env remove -n ${ENV_NAME} -y
        conda create -n ${ENV_NAME} python=3.9 -y
    else
        echo "Keeping existing environment."
    fi
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

# Add MuJoCo to library path
if [ -d "${MUJOCO_DIR}/bin" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"
    echo "Added MuJoCo bin to LD_LIBRARY_PATH: ${MUJOCO_DIR}/bin"
else
    echo "WARNING: MuJoCo directory not found at ${MUJOCO_DIR}/bin"
    echo "Please set MUJOCO_DIR to your MuJoCo installation path."
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

pip install -r requirements.txt --no-deps

# ======================== STEP 8: Install strict version overrides ==========
echo ""
echo "============================================="
echo "Step 8: Installing strict package version overrides..."
echo "============================================="

# dm-acme with jax and tf extras
pip install "dm-acme[jax,tf]"

# JAX and JAXlib (CPU versions first, GPU versions installed in GPU setup step)
pip install jax==0.4.10 jaxlib==0.4.10

# ml_dtypes compatible with jax 0.4.10
pip install ml_dtypes==0.2.0

# dm-haiku
pip install dm-haiku==0.0.9

# gymnasium-robotics (for additional env support)
pip install gymnasium-robotics

# scipy - uninstall first then install specific version
pip uninstall scipy -y
pip install scipy==1.12

# torch, scikit-learn, pandas
pip install torch==2.1.2 scikit-learn pandas

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
