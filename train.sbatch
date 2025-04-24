#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=befo-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=2


source /u/wchen11/anaconda3/bin/activate
conda activate zero

# Run training script
cd /u/wchen11/verl
bash ./train.sh 
