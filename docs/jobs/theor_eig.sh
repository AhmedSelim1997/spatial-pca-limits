#!/bin/bash
#SBATCH --job-name=1d_SS_branch_length
#SBATCH --time=20:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5000

module load python
source activate main

python ../../scripts/branch_length.py