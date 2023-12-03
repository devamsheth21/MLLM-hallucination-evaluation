#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 8            # number of cores
#SBATCH --mem 64        # amount of memory
#SBATCH -G a100:1      # GPU
#SBATCH -t 1-00:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest
# Using python, so source activate an appropriate environment
source activate FullPipeline1

python /scratch/ajoshi72/script-QnA_Blip_9000_10000.py