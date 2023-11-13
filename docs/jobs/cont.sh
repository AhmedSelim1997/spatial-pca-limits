#!/bin/bash
#SBATCH --job-name=cont
#SBATCH --output=../job_outputs/cont_more/cont%a.out
#SBATCH --error=../job_outputs/cont_more/cont%a.err
#SBATCH --array=2-3
#SBATCH --time=02:30:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5000

cd ../../scripts/data_generation
module load python
source activate main
echo job started
pwd

N_value=$(awk -F',' -v i=${SLURM_ARRAY_TASK_ID} 'NR==i{print $2}' ../../docs/parameter_csv_files/cont_params_more.csv)
m_value=$(awk -F',' -v i=${SLURM_ARRAY_TASK_ID} 'NR==i{print $3}' ../../docs/parameter_csv_files/cont_params_more.csv)

time ./run_eigenanalysis.sh -d cont -N $N_value -k 3 -m $m_value -o ../../data/cont/ >> ../../docs/job_outputs/cont/cont${SLURM_ARRAY_TASK_ID}.out

echo job ended