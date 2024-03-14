#!/bin/bash
#SBATCH --account="punim2142"
#SBATCH --ntasks=1
##SBATCH -p bigmem
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=256G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="reformat_dataset"
#SBATCH --output /data/projects/punim2142/zebrafish_experiments/data/slurm_output/rf-%j.out
#SBATCH --error /data/projects/punim2142/zebrafish_experiments/data/slurm_output/rf-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etienne.mueller@unimelb.edu.au

python3 -u /data/projects/punim2142/zebrafish_experiments/scratch/reformat_dataset.py
