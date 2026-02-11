#!/bin/bash
###############################################################################
# run_all_experiments.sh - Submit all experiment combinations to SLURM
#
# This script submits SLURM jobs for every combination of:
#   - Environments: sawyer_bin, sawyer_box, sawyer_peg, point_Spiral11x11
#   - Algorithms:   contrastive_nce, contrastive_cpc, c_learning, nce+c_learning
#   - Seeds:        configurable (default: 3 seeds per combination)
#
# Usage:
#   bash run_all_experiments.sh              # Submit all experiments
#   bash run_all_experiments.sh --dry-run    # Print commands without submitting
###############################################################################

set -e

# ======================== CONFIGURABLE PARAMETERS ===========================

# Environments to run
ENVS=("sawyer_bin" "sawyer_box" "sawyer_peg" "point_Spiral11x11")

# Algorithms to run
ALGS=("contrastive_nce" "contrastive_cpc" "c_learning" "nce+c_learning")

# Seeds for each experiment (multiple seeds for statistical significance)
SEEDS=(0 1 2)

# Number of training steps
NUM_STEPS=8000000

# Log directory base
LOG_DIR="logs/"

# Whether to sample goals (false = fixed goals, true = original CRL behavior)
SAMPLE_GOALS="false"

# Whether to add UID to log directories
ADD_UID="true"

# SLURM settings
PARTITION="nvidia"
TIME="48:00:00"
MEM="32G"
CPUS=8
GPUS=1

# ======================== PARSE ARGUMENTS ===================================

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No jobs will be submitted ==="
    echo ""
fi

# ======================== CREATE DIRECTORIES ================================

mkdir -p joblog
mkdir -p "${LOG_DIR}"

# ======================== SUBMIT JOBS =======================================

TOTAL=0
SUBMITTED=0

echo "============================================="
echo "Submitting SGCRL Experiments"
echo "============================================="
echo ""
echo "Environments: ${ENVS[*]}"
echo "Algorithms:   ${ALGS[*]}"
echo "Seeds:        ${SEEDS[*]}"
echo "Num steps:    ${NUM_STEPS}"
echo "Sample goals: ${SAMPLE_GOALS}"
echo ""
echo "Total experiments: $(( ${#ENVS[@]} * ${#ALGS[@]} * ${#SEEDS[@]} ))"
echo "============================================="
echo ""

for ENV in "${ENVS[@]}"; do
    for ALG in "${ALGS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            TOTAL=$((TOTAL + 1))
            JOB_NAME="sgcrl_${ENV}_${ALG}_s${SEED}"

            # Truncate job name if too long for SLURM (max ~128 chars)
            if [ ${#JOB_NAME} -gt 64 ]; then
                JOB_NAME="${JOB_NAME:0:64}"
            fi

            SBATCH_CMD="sbatch \
                --job-name=${JOB_NAME} \
                --partition=${PARTITION} \
                --gres=gpu:${GPUS} \
                --cpus-per-task=${CPUS} \
                --mem=${MEM} \
                --time=${TIME} \
                --output=joblog/${JOB_NAME}_%j.out \
                --error=joblog/${JOB_NAME}_%j.err \
                --export=ALL,ENV=${ENV},ALG=${ALG},SEED=${SEED},NUM_STEPS=${NUM_STEPS},SAMPLE_GOALS=${SAMPLE_GOALS},LOG_DIR=${LOG_DIR},ADD_UID=${ADD_UID} \
                run_experiment.slurm"

            if [ "$DRY_RUN" = true ]; then
                echo "[DRY RUN] ${JOB_NAME}"
                echo "  env=${ENV} alg=${ALG} seed=${SEED}"
                echo "  cmd: ${SBATCH_CMD}"
                echo ""
            else
                echo "Submitting: ${JOB_NAME} (env=${ENV}, alg=${ALG}, seed=${SEED})"
                eval ${SBATCH_CMD}
                SUBMITTED=$((SUBMITTED + 1))

                # Small delay between submissions to avoid overwhelming the scheduler
                sleep 0.5
            fi
        done
    done
done

echo "============================================="
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN complete. ${TOTAL} jobs would be submitted."
else
    echo "Submitted ${SUBMITTED}/${TOTAL} jobs."
    echo ""
    echo "Monitor jobs with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Cancel all jobs with:"
    echo "  scancel -u \$USER"
    echo ""
    echo "View job output with:"
    echo "  tail -f joblog/sgcrl_*_<JOB_ID>.out"
fi
echo "============================================="
