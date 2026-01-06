#!/bin/bash
#SBATCH --job-name=gsm8k-4gpu
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfoz-delta-gpu
#SBATCH --time=47:59:59
#SBATCH --gpus-per-node=4

source /u/aseo/anaconda3/bin/activate
conda activate verlog
cd /u/aseo/Verlog

ulimit -n 65535

bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_4xgpu.sh