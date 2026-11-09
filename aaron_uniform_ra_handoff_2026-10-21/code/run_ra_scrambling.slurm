#!/bin/bash
#SBATCH --job-name=rascr
#SBATCH --partition=cpucluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/Users/922559219/flux_h0h1_20260830/IceCube_Population_SBI/logs/rascr_%j.out
#SBATCH --error=/Users/922559219/flux_h0h1_20260830/IceCube_Population_SBI/logs/rascr_%j.err
#SBATCH --chdir=/Users/922559219/flux_h0h1_20260830/IceCube_Population_SBI

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: sbatch $0 GLOBAL_SIM_ID MODE METHOD"
    echo "MODE must be: combined, astro-only, or astro-plus-fixed-atmo"
    echo "METHOD must be: permute-within-period or uniform-full-ra"
    exit 1
fi

GLOBAL_SIM_ID="$1"
MODE="$2"
METHOD="$3"

case "${MODE}" in
    combined)
        MODE_DIR="combined"
        ;;
    astro-only)
        MODE_DIR="astro_only"
        ;;
    astro-plus-fixed-atmo)
        MODE_DIR="astro_plus_fixed_atmo"
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        exit 1
        ;;
esac

case "${METHOD}" in
    permute-within-period)
        METHOD_DIR=""
        ;;
    uniform-full-ra)
        METHOD_DIR="uniform_full_ra"
        ;;
    *)
        echo "Unknown method: ${METHOD}"
        exit 1
        ;;
esac

DEV_ROOT="/Users/922559219/flux_scrambling_dev"
ANALYSIS_ROOT="${DEV_ROOT}/analysis"

REPLAY_DIR="${ANALYSIS_ROOT}/replay/replay_${GLOBAL_SIM_ID}"

if [ -n "${METHOD_DIR}" ]; then
    SCRAMBLE_DIR="${ANALYSIS_ROOT}/scrambles/${MODE_DIR}/${METHOD_DIR}/ra_scrambles_${GLOBAL_SIM_ID}"
else
    SCRAMBLE_DIR="${ANALYSIS_ROOT}/scrambles/${MODE_DIR}/ra_scrambles_${GLOBAL_SIM_ID}"
fi

echo "========================================"
echo "Starting RA scrambling"
echo "global_sim_id: ${GLOBAL_SIM_ID}"
echo "mode:          ${MODE}"
echo "method:        ${METHOD}"
echo "replay_dir:    ${REPLAY_DIR}"
echo "output_dir:    ${SCRAMBLE_DIR}"
echo "Host:          $(hostname)"
echo "Date:          $(date)"
echo "PWD:           $(pwd)"
echo "========================================"

source /Users/922559219/miniconda3/etc/profile.d/conda.sh
conda activate icecube_clean_clone

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=src

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/run_ra_scrambling.py \
  --global-sim-id "${GLOBAL_SIM_ID}" \
  --mode "${MODE}" \
  --method "${METHOD}" \
  --replay-dir "${REPLAY_DIR}" \
  --out-dir "${SCRAMBLE_DIR}" \
  --n-scrambles 1000 \
  --seed 20261003