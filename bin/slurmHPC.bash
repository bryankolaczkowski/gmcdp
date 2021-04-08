#!/bin/bash

# job IO
#SBATCH  --job-name=cgan
#SBATCH  --output=cgan.out
#SBATCH  --error=cgan.err

# cpu resources
#SBATCH  --nodes=1
#SBATCH  --tasks=1
#SBATCH  --cpus-per-task=8
#SBATCH  --mem-per-cpu=2GB

# gpu resources and queue
#SBATCH  --partition=gpu
#SBATCH  --gres=gpu:1
#SBATCH  --account=microbiome-ai
#SBATCH  --qos=microbiome-ai

# time resources
#SBATCH  --time=96:00:00

module load tensorflow

echo "date $(date)"
echo "host $(hostname -s)"
echo "dir  $(pwd)"

echo "starting."
tbeg=$(date +%s)
## begin analysis commands ##
./cgan -f ../examples/sim2/data.csv
## done  analysis commands ##
tend=$(date +%s)

rtime=$((tend - tbeg))
echo "time ${rtime}"
echo "finished."
