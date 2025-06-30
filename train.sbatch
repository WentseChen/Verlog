#!/bin/bash
#SBATCH --job-name=test              # Name of the job
#SBATCH --output=logs/my_job-%j.out  # Standard output and error log
#SBATCH --error=logs/my_job-%j.err
#SBATCH --partition=preempt          # Partition to use
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --gres=gpu:8                 # Number of GPUs (max 8 per node)
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=16           # Adjust based on your needs
#SBATCH --mem=0                      # Use all available memory
#SBATCH --time=24:00:00               # Max time (48 hours)

conda activate verlog
cd verlog

VERLOG=$HOME/verlog

remote=gs://cmu-gpucloud-wentsec
local=/dev/shm

PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$VERLOG/examples/data/placeholder.parquet \
 data.val_files=$VERLOG/examples/data/placeholder.parquet \
 data.train_batch_size=512 \
 data.max_prompt_length=3072 \
 data.max_response_length=512 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=256 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.actor.entropy_coeff=0.001 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.temperature=1.0 \
 actor_rollout_ref.actor.ppo_epochs=1 \
 actor_rollout_ref.actor.clip_ratio_high=0.2 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-7B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=8 \
 critic.ppo_epochs=1 \
 critic.highlight_first=True \
 critic.highlight_ratio=3.0 \
 algorithm.step_gamma=0.99 \
 algorithm.step_lam=0.95 \
 algorithm.use_kl_in_reward=True \
 algorithm.kl_ctrl.kl_coef=0.003 \
 envs.n_rollouts=64 \
 envs.env_name=crafter \
 envs.task=default \
 envs.format_penalty=0.1 \
 envs.binary_reward=False \
 envs.captioner.type=cot \
 envs.captioner.max_text_history=4 \
 envs.captioner.max_cot_history=4\
 trainer.logger=['console','wandb'] \
 trainer.project_name=zero \
 trainer.experiment_name=test \
 trainer.default_hdfs_dir=null \
 trainer.val_before_train=False \
 trainer.critic_warmup=30 \
 trainer.critic_warmup_step=10 \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=99 \
 trainer.max_actor_ckpt_to_keep=1 \
 trainer.max_critic_ckpt_to_keep=1 \
 trainer.test_freq=100000000 \
 trainer.total_epochs=20000 2>&1 | tee verl_demo.log


