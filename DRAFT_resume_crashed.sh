#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=dcc_resume
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

# Torch-HPC wrapper for the six unresolved continual runs audited on
# 2026-08-20. DRAFT.sh launches three configurations per GPU:
#   array 0 -> C2 DCC seeds 100/101, AC-DCC seed 5
#   array 1 -> AC-DCC seeds 6/7, DCC-SAC seed 7
#
# START_TASK remains 0 intentionally: run_continual_contrastive.py scans the
# exact configuration/seed checkpoint directory and resumes after the latest
# completed task. A crash in the middle of a task restarts that task.

set -euo pipefail

export CONFIG_SCRIPT=experiment_configs_resume_crashed.py
export CONFIG_INDEX_OFFSET=0
export CONFIG_LIMIT=6
export START_TASK=0

exec bash /scratch/yd2247/sgcrl/DRAFT.sh
