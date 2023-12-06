#!/bin/bash
#SBATCH --job-name=pop_split
#SBATCH --output=../job_outputs/pop_split/pop_split_%a.out
#SBATCH --error=../job_outputs/pop_split/pop_split_%a.err
#SBATCH --array=2-3
#SBATCH --time=02:00:00
#SBATCH --partition=jnovembre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5000

cd ../../scripts/data_generation
module load python
source activate main
pwd

Tau_value=$(awk -F',' -v i=${SLURM_ARRAY_TASK_ID} 'NR==i{print $2}' ../../docs/parameter_csv_files/pop_split_params.csv)
./run_eigenanalysis.sh -d split -t $Tau_value -o ../../data/pop_split/ >> ../../docs/job_outputs/pop_split/pop_split_${SLURM_ARRAY_TASK_ID}.out

