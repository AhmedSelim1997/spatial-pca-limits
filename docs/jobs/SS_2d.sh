#!/bin/bash
#SBATCH --job-name=2d_SS
#SBATCH --output=../job_outputs/SS_2d/SS_2d_%a.out
#SBATCH --error=../job_outputs/SS_2d/SS_2d_%a.err
#SBATCH --array=98-301
#SBATCH --time=05:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000

cd ../../scripts/data_generation
module load python
source activate main
echo job started

K_value=$(awk -F',' -v i=${SLURM_ARRAY_TASK_ID} 'NR==i{print $2}' ../../docs/parameter_csv_files/SS_2d_params.csv)
m_value=$(awk -F',' -v i=${SLURM_ARRAY_TASK_ID} 'NR==i{print $3}' ../../docs/parameter_csv_files/SS_2d_params.csv)
echo params loaded

./run_eigenanalysis.sh -d SteppingStones_2d -k $K_value -m $m_value -o ../../data/SS_2d/ >> ../../docs/job_outputs/SS_2d/SS_2d_${SLURM_ARRAY_TASK_ID}.out