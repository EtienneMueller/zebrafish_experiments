#!/bin/bash
#SBATCH --account="punim2142"
#SBATCH --ntasks=1
##SBATCH -p bigmem
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=1
#SBATCH --job-name="reformat_dataset"
#SBATCH --output /data/projects/punim2142/zebrafish_experiments/data/rf-%j.out
#SBATCH --error /data/projects/punim2142/zebrafish_experiments/data/rf-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=etienne.mueller@unimelb.edu.au

python3 -u /data/projects/punim2142/zebrafish_experiments/scripts/predict_daisy.py predict -n "2023-05-09.yaml" -c "60000" -oc "out_container" -od "out_dataset" -ic "data/zebrafish_out.n5" -id "volumes/8-1/cells" --local
