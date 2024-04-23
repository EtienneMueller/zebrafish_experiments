#!/bin/bash
#SBATCH --account="punim2142"
#SBATCH --ntasks=1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-cpu=256G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="submit"
#SBATCH --output /data/projects/punim2142/zebrafish_experiments/slurm_out/s-%j.out
#SBATCH --error /data/projects/punim2142/zebrafish_experiments/slurm_out/s-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etienne.mueller@unimelb.edu.au

python scripts/submit.py predict -p /data/projects/punim2142/zebrafish_experiments/configs/zebrafish/predictions/2024-04-22.yaml
