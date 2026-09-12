#!/usr/bin/env bash
# NYUAD Jubail HPC: runtime env after `conda activate contrastive_rl`.
# CUDA comes from `module load cuda/11.8.0`; conda-gcc provides conda.
#
# Usage (interactive or inside batch):
#   module load cuda/11.8.0
#   module load conda-gcc/11.2.0
#   eval "$(conda shell.bash hook)"
#   conda activate contrastive_rl
#   source /scratch/yd2247/sgcrl/set_up/jubail_hpc_env.sh

: "${CONDA_PREFIX:?Run: conda activate contrastive_rl first}"

export SCRATCH="${SCRATCH:-/scratch/$(whoami)}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_MIN_VLOG_LEVEL="${TF_CPP_MIN_VLOG_LEVEL:-3}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64,GNU}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/.cache/pip}"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

export PATH="${CONDA_PREFIX}/bin:$PATH"

export MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
if [ -d "${MUJOCO_DIR}/bin" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"
fi
if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/lib64" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
fi

# mujoco_py string-checks that /usr/lib/nvidia appears in LD_LIBRARY_PATH.
for _d in /usr/lib/nvidia /usr/lib64/nvidia; do
  if [ -d "$_d" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${_d}"
    break
  fi
done
for _nd in /usr/lib/nvidia /usr/lib64/nvidia; do
  if [[ ":${LD_LIBRARY_PATH}:" != *":${_nd}:"* ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${_nd}"
  fi
done

CUDNN_LIB="$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null)" || CUDNN_LIB=""
if [ -n "${CUDNN_LIB}" ] && [ -d "${CUDNN_LIB}" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDNN_LIB}"
fi
