#!/bin/bash
#SBATCH --account="punim2142"
#SBATCH --ntasks=1
##SBATCH -p gpu-a100
##SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=256G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="dacapo"
#SBATCH --output /data/projects/punim2142/zebrafish_experiments/slurm_out/dc-%j.out
#SBATCH --error /data/projects/punim2142/zebrafish_experiments/slurm_out/dc-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etienne.mueller@unimelb.edu.au

dacapo train -r scratch_3d_lsdaffs_zebrafish_cells_upsample-unet_default_v6__0