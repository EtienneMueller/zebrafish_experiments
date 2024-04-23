#!/bin/bash
#SBATCH --account="punim2142"
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="config_dacapo"
#SBATCH --output /data/projects/punim2142/zebrafish_experiments/slurm_out/c1-%j.out
#SBATCH --error /data/projects/punim2142/zebrafish_experiments/slurm_out/c1-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etienne.mueller@unimelb.edu.au

python configs/scripts/datasplits/datasets.py update --force
python configs/scripts/tasks/tasks.py update
python configs/scripts/trainers/trainers.py update --force
python configs/scripts/architectures/architectures.py update
python configs/scripts/runs/runs.py update -n zebrafish_cells -d zebrafish_cells -a upsample-unet -ta 3d_lsdaffs -tr default_v4 -s "" -r 2 -i 80000 -v 5000 --force
