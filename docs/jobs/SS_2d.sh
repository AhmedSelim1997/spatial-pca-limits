#!/bin/bash
#SBATCH --job-name=2d_SS
#SBATCH --output=../job_outputs/2d_SS/2d_SS_%a.out
#SBATCH --error=../job_outputs/2d_SS/2d_SS_%a.err
#SBATCH --array=2-300
#SBATCH --time=02:30:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=6000

cd ../../scripts/data_generation
module load python
source activate main
echo job started

K_value=$(awk -F',' -v i=${SLURM_ARRAY_TASK_ID} 'NR==i{print $2}' ../../docs/parameter_csv_files/SS_2d_params.csv)
m_value=$(awk -F',' -v i=${SLURM_ARRAY_TASK_ID} 'NR==i{print $3}' ../../docs/parameter_csv_files/SS_2d_params.csv)
echo params loaded

./run_eigenanalysis.sh -d SteppingStones_2d -k $K_value -m $m_value -o ../../data/2d_SS/ >> ../../docs/job_outputs/2d_SS/2d_SS_${SLURM_ARRAY_TASK_ID}.out