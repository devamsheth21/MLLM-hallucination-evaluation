#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 10            # number of cores
#SBATCH --mem=64G       # amount of memory
#SBATCH -G a100:1       # GPU
#SBATCH -t 0-12:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest
# Using python, so source activate an appropriate environment
source activate pipeline1

python /scratch/dmsheth1/nlp/MLLM-hallucination-evaluation/MLLM-OpenFlamingo/flamingo_script-final.py $1 $2