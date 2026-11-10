#!/usr/bin/env bash
# Submit a two-task GPU smoke test, then all ten paper seeds after it passes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export VANILLA_REPO_DIR="$PWD"
export VANILLA_RUN_ROOT="${VANILLA_RUN_ROOT:-$PWD/logs/vanilla_crl_gate_bc}"
export VANILLA_EXPECTED_COMMIT="$(git rev-parse HEAD)"
if ! git diff --quiet HEAD --; then
  echo 'Commit or stash tracked code changes before submitting.' >&2
  exit 1
fi
if [[ -e "$VANILLA_RUN_ROOT/STOP" ]]; then
  echo "Remove $VANILLA_RUN_ROOT/STOP only when ready to resume." >&2
  exit 1
fi
mkdir -p "$VANILLA_RUN_ROOT/slurm"
job_script="$PWD/scripts/jubail_vanilla_crl_gate_bc.sbatch"
smoke=$(sbatch --parsable --time=02:00:00 \
  --output="$VANILLA_RUN_ROOT/slurm/smoke_%j.out" \
  --error="$VANILLA_RUN_ROOT/slurm/smoke_%j.err" \
  --export=ALL,VANILLA_MODE=smoke,VANILLA_CHAIN_INDEX=0 "$job_script")
smoke=${smoke%%;*}
if ! production=$(sbatch --parsable --array=0-9 --dependency="afterok:$smoke" \
  --kill-on-invalid-dep=yes \
  --output="$VANILLA_RUN_ROOT/slurm/%A_%a.out" \
  --error="$VANILLA_RUN_ROOT/slurm/%A_%a.err" \
  --export=ALL,VANILLA_MODE=run,VANILLA_CHAIN_INDEX=0 "$job_script"); then
  scancel "$smoke"
  exit 1
fi
printf 'Smoke job: %s\nProduction array: %s\nLogs: %s/slurm\n' "$smoke" "$production" "$VANILLA_RUN_ROOT"
