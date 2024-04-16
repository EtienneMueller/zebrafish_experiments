#!/bin/bash
#SBATCH --account="punim2142"
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="config_dacapo"
#SBATCH --output /data/projects/punim2142/zebrafish_experiments/data/slurm_output/p-%j.out
#SBATCH --error /data/projects/punim2142/zebrafish_experiments/data/slurm_output/p-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etienne.mueller@unimelb.edu.au

python scripts/predict_daisy.py predict -n "2024-04-07" -c "70000" -oc data/predictions/zebrafish.n5 -od data/predictions/2024-04-07/70000 -ic /data/projects/punim2142/zebrafish_experiments/data/s16_bottom_110nm_rec_.n5 -id raw --local -w 1