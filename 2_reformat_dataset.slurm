#!/bin/bash
#SBATCH --account="punim2142"
#SBATCH --ntasks=1
##SBATCH -p bigmem
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=128G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="reformat_dataset"
#SBATCH --output /data/projects/punim2142/zebrafish_experiments/slurm_out/rf-%j.out
#SBATCH --error /data/projects/punim2142/zebrafish_experiments/slurm_out/rf-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etienne.mueller@unimelb.edu.au

python3 -u /data/projects/punim2142/zebrafish_experiments/scratch/reformat_dataset.py
