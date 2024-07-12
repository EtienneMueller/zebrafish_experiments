#!/bin/bash
#SBATCH --account="punim2142"
#SBATCH --ntasks=1
#SBATCH --time=0-01:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="post_processing"
#SBATCH --output /data/projects/punim2142/zebrafish_experiments/slurm_out/post02-%j.out
#SBATCH --error /data/projects/punim2142/zebrafish_experiments/slurm_out/post02-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etienne.mueller@unimelb.edu.au

python scripts/post_processing/02_extract_fragments_blockwise.py