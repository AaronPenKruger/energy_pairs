#!/bin/bash
#SBATCH --job-name=paircount
#SBATCH --partition=cpucluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/paircount_%j.out
#SBATCH --error=logs/paircount_%j.err

# Usage (from ~/energy_pairs):  mkdir -p logs && sbatch code/run_pair_counts.slurm
set -euo pipefail

source /Users/917910739/miniconda/etc/profile.d/conda.sh
conda activate icecube_clean
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

DATA=aaron_pair_counter_data_2026-10-24
echo "host $(hostname)  date $(date)"

python code/test_pair_counter.py

for SIM in 26392 14265; do
    python code/run_pair_counts.py \
        --null-dir "${DATA}/uniform_${SIM}" \
        --out "results/counts_${SIM}.npz"
done
