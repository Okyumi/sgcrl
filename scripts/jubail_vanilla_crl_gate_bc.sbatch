#!/usr/bin/env bash
#SBATCH --job-name=crl_gate_bc
#SBATCH --partition=nvidia
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
set -euo pipefail
cd "${VANILLA_REPO_DIR:?Submit using scripts/submit_vanilla_crl_gate_bc.sh}"
root="${VANILLA_RUN_ROOT:?}"
[[ ! -e "$root/STOP" ]] || exit 0
[[ "$(git rev-parse HEAD)" == "${VANILLA_EXPECTED_COMMIT:?}" ]] || {
  echo 'Repository commit changed since submission; refusing to mix versions.' >&2; exit 1;
}
git diff --quiet HEAD -- || { echo 'Tracked code has changed since submission.' >&2; exit 1; }
module purge
module load cuda/11.8.0 conda-gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate contrastive_rl
source set_up/jubail_hpc_env.sh
export MUJOCO_GL=egl WANDB_MODE=online WANDB_BASE_URL=https://api.wandb.ai
export WANDB_ENTITY="${WANDB_ENTITY:-nyuad_mmvc}"
export WANDB_START_METHOD=thread
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 XLA_PYTHON_CLIENT_PREALLOCATE=true
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.wandb_api_key" ]]; then
  export WANDB_API_KEY="$(tr -d '[:space:]' < "$HOME/.wandb_api_key")"
fi
python scripts/vanilla_crl_gate_bc.py --mode=preflight --root="$root"
if [[ "${VANILLA_MODE:-run}" == smoke ]]; then
  # Numerical tests need only JAX and NumPy already in the RL environment.
  JAX_PLATFORM_NAME=cpu python tests/test_vanilla_success_bc.py
  JAX_PLATFORM_NAME=cpu python tests/test_vanilla_learner_integration.py
  python scripts/vanilla_crl_gate_bc.py --mode=smoke --seed=5 \
    --root="$root/smoke_${SLURM_JOB_ID}"
  exit $?
fi
index="${SLURM_ARRAY_TASK_ID:?}"
seed=$((index + 5))
mkdir -p "$root"
# Submission retries cannot train the same seed concurrently.
exec 9>"$root/seed_${seed}.lock"
flock -n 9 || { echo "Seed $seed already running."; exit 0; }
if python scripts/vanilla_crl_gate_bc.py --mode=complete --root="$root" --seed="$seed"; then
  exit 0
fi
chain="${VANILLA_CHAIN_INDEX:-0}"
max_jobs="${VANILLA_MAX_JOBS:-20}"
next_job=""
if (( chain + 1 < max_jobs )); then
  # Queue BEFORE training, so Slurm TIMEOUT/node failure still releases the next job.
  next_job=$(sbatch --parsable \
    --dependency="afterany:${SLURM_ARRAY_JOB_ID}_${index}" --array="$index" \
    --output="$root/slurm/%A_%a.out" --error="$root/slurm/%A_%a.err" \
    --export=ALL,VANILLA_CHAIN_INDEX="$((chain + 1))" \
    "$VANILLA_REPO_DIR/scripts/jubail_vanilla_crl_gate_bc.sbatch")
  next_job=${next_job%%;*}
  printf 'Seed %s: queued continuation %s (job %s/%s)\n' "$seed" "$next_job" "$((chain + 2))" "$max_jobs"
else
  echo "Last allocation in chain ($max_jobs); resubmit if incomplete."
fi
# On scheduler termination leave the successor queued. Ordinary Python errors
# cancel it, avoiding repeated expensive launches of broken code.
trap 'exit 143' TERM INT
set +e
python scripts/vanilla_crl_gate_bc.py --mode=run --seed="$seed" --root="$root"
rc=$?
set -e
if (( rc != 0 )); then
  [[ -z "$next_job" ]] || scancel "$next_job"
  echo "Training failed (exit $rc); continuation cancelled. Inspect this job's log." >&2
  exit "$rc"
fi
if python scripts/vanilla_crl_gate_bc.py --mode=complete --root="$root" --seed="$seed"; then
  [[ -z "$next_job" ]] || scancel "$next_job"
fi
