#!/bin/bash
#SBATCH --partition gpgpu
#SBATCH --account=punim0638
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=eroesch@student.unimelb.edu.au
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBACTH --qos=gpgpuresplat
# Note: SLURM defaults to running jobs in the directory

# Load required modules
module load julia/1.5.1-linux-x86_64

# Launch julia code
julia gpu_server_test.jl
