#!/bin/bash
#SBATCH --account="punim2142"
#SBATCH --ntasks=1
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="d_submit"
#SBATCH --output /data/projects/punim2142/zebrafish_experiments/slurm_out/debug-%j.out
#SBATCH --error /data/projects/punim2142/zebrafish_experiments/slurm_out/debug-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etienne.mueller@unimelb.edu.au

python scripts/submit.py predict -w 1 -p /data/projects/punim2142/zebrafish_experiments/configs/zebrafish/predictions/2024-04-22.yaml
