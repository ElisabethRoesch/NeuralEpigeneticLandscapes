#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=eroesch@student.unimelb.edu.au
#SBATCH --mail-type=ALL
# Note: SLURM defaults to running jobs in the directory

# Load required modules
module load julia/1.5.1-linux-x86_64

# Launch julia code
julia test_diffeqflux_neuralsde_example.jl
