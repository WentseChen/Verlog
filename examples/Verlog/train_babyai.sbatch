#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mem=200GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4
#SBATCH --account=YOUR_ACCOUNT_NAME
#SBATCH --time=47:59:59
#SBATCH --gpus-per-node=4

cd Verlog
conda activate verlog

model=Qwen/Qwen2.5-3B-Instruct
micro_batch_size=8

PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 python3 -m verl.trainer.main_ppo \
 data.max_prompt_length=1024 \
 data.max_response_length=512 \
 data.train_batch_size=256 \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
 actor_rollout_ref.model.path=$model \
 critic.model.path=$model \
 critic.ppo_micro_batch_size_per_gpu=$micro_batch_size \
 critic.highlight_first=True \
 critic.highlight_ratio=3.0 \
 algorithm.step_gamma=0.99 \
 algorithm.step_lam=0.95 \
 algorithm.use_kl_in_reward=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 envs.n_rollouts=32 \
 envs.env_name=babyai \
 envs.task=BabyAI-MixedTrainLocal-v0/goto \
 envs.format_penalty=0.1 \
 envs.binary_reward=True \
 envs.captioner.type=cot \
 envs.captioner.max_text_history=1 \
 envs.captioner.max_cot_history=1 \
 trainer.project_name=zero \
 trainer.experiment_name=test \
 trainer.val_before_train=False \
 trainer.critic_warmup=40 \
 trainer.critic_warmup_step=5 \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=1000000 \
 trainer.test_freq=30 \
 trainer.render=False \
 trainer.total_epochs=300 2>&1 | tee verl_demo.log
