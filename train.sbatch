#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=befo-delta-gpu
#SBATCH --time=47:59:59
#SBATCH --gpus-per-node=4

source /u/wchen11/anaconda3/bin/activate
conda activate zero

# Run training script
cd /u/wchen11/verl

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/test.parquet \
 data.val_files=$HOME/verl/examples/data/placeholder.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=1024 \
 data.max_response_length=384 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.actor.entropy_coeff=0.001 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.temperature=1.0 \
 actor_rollout_ref.actor.ppo_epochs=1 \
 actor_rollout_ref.actor.clip_ratio_high=0.2 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-3B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=8 \
 critic.ppo_epochs=1 \
 critic.highlight_first=True \
 critic.highlight_ratio=3.0 \
 algorithm.step_gamma=0.99 \
 algorithm.step_lam=0.95 \
 algorithm.use_kl_in_reward=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 envs.n_rollouts=32 \
 envs.env_name=crafter \
 envs.task=default \
 envs.format_penalty=0.1 \
 envs.binary_reward=True \
 envs.captioner.type=cot \
 envs.captioner.max_text_history=1 \
 envs.captioner.max_cot_history=1 \
 trainer.logger=['console','wandb'] \
 trainer.project_name=zero \
 trainer.experiment_name=test \
 trainer.default_hdfs_dir=null \
 trainer.val_before_train=False \
 trainer.critic_warmup=60 \
 trainer.critic_warmup_step=10 \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=100000000 \
 trainer.test_freq=15 \
 trainer.total_epochs=400 2>&1 | tee verl_demo.log
