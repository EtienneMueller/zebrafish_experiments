#!/bin/bash
#SBATCH --account="punim2142"
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=200G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="process_stitched"
#SBATCH --output /data/projects/punim2142/zebrafish_experiments/slurm_out/psv-%j.out
#SBATCH --error /data/projects/punim2142/zebrafish_experiments/slurm_out/psv-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etienne.mueller@unimelb.edu.au

python3 /data/projects/punim2142/zebrafish_experiments/scratch/process_stitched_vol.py
