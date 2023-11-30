#!/bin/bash
#SBATCH --job-name=1d_SS
#SBATCH --time=01:30:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000

cd ../../scripts/data_generation
module load python
source activate main


./run_eigenanalysis.sh -d SteppingStones_1d -k 11 -m 0.0001 -o ../../data/SS_1d/