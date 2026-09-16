#!/bin/bash
#SBATCH --job-name=paper_disp
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --partition=cs
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/paper_dispatcher/%j.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/paper_dispatcher/%j.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --mail-type=END,FAIL

# CPU watcher: after paper_fs is running, submit leftover 10-seed GPU
# jobs into free qos-gpu48 slots (max 16 pending+running).
set -euo pipefail
REPO_DIR="/scratch/yd2247/sgcrl"
mkdir -p /scratch/yd2247/sgcrl/logs/paper_dispatcher
cd "$REPO_DIR"
python scripts/paper_cap_dispatcher.py --loop --sleep 120
