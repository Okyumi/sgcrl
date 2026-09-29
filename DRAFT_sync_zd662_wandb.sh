#!/bin/bash
#SBATCH --job-name=zd662_wandb_sync
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --partition=cs
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/zd662_wandb_sync_slurm_%j.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/zd662_wandb_sync_slurm_%j.err

# CPU job: finish uploading collaborator offline W&B runs into
# nyuad_mmvc/zd662_sparse_sac_her. Skips IDs already in the project.
set -euo pipefail
REPO_DIR="/scratch/yd2247/sgcrl"
cd "$REPO_DIR"
mkdir -p "$REPO_DIR/logs" "$REPO_DIR/results/data/raw/zd662_continual_sac"

export PATH="/scratch/yd2247/miniconda3/envs/contrastive_rl/bin:$PATH"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.wandb_api_key" ]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${HOME}/.wandb_api_key")"
  export WANDB_API_KEY
fi
export PYTHONUNBUFFERED=1

python -u scripts/sync_zd662_offline_wandb.py --workers 8
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} finished at $(date -Is)"
