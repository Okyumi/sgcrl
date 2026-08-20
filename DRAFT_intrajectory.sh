#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=dcc_itn
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --output=/scratch/yd2247/sgcrl/logs/continual/%A_%a.out
#SBATCH --error=/scratch/yd2247/sgcrl/logs/continual/%A_%a.err
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-1

# Torch-HPC-only wrapper for the six DCC in-trajectory-negative experiments.
# DRAFT.sh launches three configs per GPU; this maps the two array jobs to:
#   array 0 -> configs 6, 7, 8   (Task 5; seeds 5, 6, 7)
#   array 1 -> configs 9, 10, 11 (Task 8; seeds 5, 6, 7)
# The canonical environment and flag construction remain in DRAFT.sh.

set -euo pipefail

export CONFIG_INDEX_OFFSET=6
export CONFIG_LIMIT=6

exec bash /scratch/yd2247/sgcrl/DRAFT.sh
