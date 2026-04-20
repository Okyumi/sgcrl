#!/usr/bin/env bash
# NYU Torch HPC: runtime env after `conda activate contrastive_rl`.
# CUDA user libs: JAX CUDA12 pip stack + conda; no `module load cuda`.
#
# Usage (interactive or inside batch):
#   source "$SCRATCH/miniconda3/etc/profile.d/conda.sh"
#   conda activate contrastive_rl
#   source /scratch/yd2247/sgcrl/set_up/torch_hpc_env.sh

: "${CONDA_PREFIX:?Run: conda activate contrastive_rl (or your env name) first}"

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

# Host NVIDIA driver libs MUST come before $CONDA_PREFIX/lib. Otherwise a conda
# cuda-toolkit "libcuda.so" stub can win and JAX/XLA fails with:
#   cudaGetErrorString symbol not found / stream is uninitialized
_DRIVER_LP=""
for _d in /usr/lib64/nvidia /usr/lib/nvidia /usr/lib64 /usr/lib/x86_64-linux-gnu; do
  if [ -f "${_d}/libcuda.so.1" ] || [ -f "${_d}/libcuda.so" ]; then
    _DRIVER_LP="${_DRIVER_LP:+${_DRIVER_LP}:}${_d}"
  fi
done

CUDNN_LIB="$(python -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null)" || CUDNN_LIB=""

export LD_LIBRARY_PATH="${_DRIVER_LP:+${_DRIVER_LP}:}"
if [ -n "${CUDNN_LIB}" ] && [ -d "${CUDNN_LIB}" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}${CUDNN_LIB}:"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}${CONDA_PREFIX}/lib"
if [ -d "${MUJOCO_DIR}/bin" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_DIR}/bin"
fi

# mujoco_py string-checks that /usr/lib/nvidia appears in LD_LIBRARY_PATH (see
# mujoco_py.builder). Append even if the dir is missing on RHEL — the check is
# what unblocks import; GL libs may still resolve via /usr/lib64/nvidia.
for _nd in /usr/lib/nvidia /usr/lib64/nvidia; do
  if [[ ":${LD_LIBRARY_PATH}:" != *":${_nd}:"* ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${_nd}"
  fi
done

# Preserve any extra paths the job already set (e.g. modules), after our stack
if [ -n "${LD_LIBRARY_PATH_EXTRA:-}" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${LD_LIBRARY_PATH_EXTRA}"
fi

# Optional: if old CUDA11 jaxlib ever needs it on a node, force host libcuda:
#   export LD_PRELOAD=/usr/lib64/libcuda.so.1
