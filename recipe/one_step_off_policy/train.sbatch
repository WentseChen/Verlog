#!/bin/bash
#SBATCH --job-name=async
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=bezc-delta-gpu
#SBATCH --time=47:59:59
#SBATCH --gpus-per-node=4

nvidia-smi

source /u/wchen11/anaconda3/bin/activate 
conda activate verl
cd /u/wchen11/verl/recipe/one_step_off_policy

./grpo_0.6b_gsm8k_fsdp2_2_6.sh 