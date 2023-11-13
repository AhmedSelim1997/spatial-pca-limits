#!/bin/bash
#SBATCH --job-name=biased_sampling
#SBATCH --output=../job_outputs/biased_sampling_%a.out
#SBATCH --error=../job_outputs/biased_sampling_%a.err
#SBATCH --array=5,7,9,11
#SBATCH --time=16:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5000

cd ../../scripts/data_generation
module load python
source activate main

python generate_baised_samples.py --K ${SLURM_ARRAY_TASK_ID} >> ../../docs/job_outputs/biased_sampling/K_${SLURM_ARRAY_TASK_ID}.out
