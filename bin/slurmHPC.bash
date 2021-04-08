#!/bin/bash

# job IO
#SBATCH  --job-name=cgan
#SBATCH  --output=cgan.out
#SBATCH  --error=cgan.err

# cpu resources
#SBATCH  --nodes=1
#SBATCH  --ntasks-per-node=8
#SBATCH  --mem-per-cpu=500MB

# gpu resources and queue
#SBATCH  --gres=gpu:1
#SBATCH  --account=microbiome-ai
#SBATCH  --qos=microbiome-ai

# time resources
#SBATCH  --time=96:00:00

module load tensorflow

echo "date $(date)"
echo "host $(hostname -s)"
echo "dir  $(pwd)"

tbeg=$(date +%s)
## begin analysis commands ##

## done  analysis commands ##
tend=$(date +%s)

rtime=$((tend - tbeg))
echo "time ${rtime}"
echo "finished."
