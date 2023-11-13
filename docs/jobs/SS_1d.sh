#!/bin/bash
#SBATCH --job-name=1d_SS
#SBATCH --array=302-401
#SBATCH --time=00:45:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5000

cd ../../scripts/data_generation
module load python
source activate main

K_value=$(awk -F',' -v i=${SLURM_ARRAY_TASK_ID} 'NR==i{print $2}' ../../docs/parameter_csv_files/SS_1d_params.csv)
m_value=$(awk -F',' -v i=${SLURM_ARRAY_TASK_ID} 'NR==i{print $3}' ../../docs/parameter_csv_files/SS_1d_params.csv)

./run_eigenanalysis.sh -d SteppingStones_1d -k $K_value -m $m_value -o ../../data/SS_1d/