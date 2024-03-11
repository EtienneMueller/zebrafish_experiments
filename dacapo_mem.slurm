#!/bin/bash
#SBATCH --account="punim2142"
#SBATCH --ntasks=1
#SBATCH -p bigmem
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=1024G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="dacapo"
#SBATCH --output /data/projects/punim2142/zebrafish_experiments/data/dc-%j.out
#SBATCH --error /data/projects/punim2142/zebrafish_experiments/data/dc-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etienne.mueller@unimelb.edu.au

dacapo train -r scratch_3d_lsdaffs_zebrafish_cells_upsample-unet_default_v4__0