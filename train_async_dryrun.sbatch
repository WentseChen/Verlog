#!/bin/bash
#SBATCH --job-name=loop
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

NUM_GPUS_PER_NODE=4
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS_PER_NODE-1)))

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
LOG_PATH="${VERL_AGENT_IO_LOG_PATH:-$PROJECT_DIR/logs/agent_model_io5.log}"
mkdir -p "$(dirname "$LOG_PATH")"
: > "$LOG_PATH"
export VERL_AGENT_IO_LOG_PATH="$LOG_PATH"

# Dry run parameters - keep these conservative to avoid OOM on a single 44GB GPU.
# Async agent loop chunks prompts across workers, so keep NUM_ENVS <= BATCH_SIZE.
# NUM_ENVS=1
# BATCH_SIZE=2
# MINI_BATCH_SIZE=2
# MICRO_BATCH_SIZE=8
# FORWARD_BATCH_SIZE=2
# OFFLOAD=true
# PPO_EPOCHS=1
NUM_ENVS=32
BATCH_SIZE=256
MINI_BATCH_SIZE=$((BATCH_SIZE))
MICRO_BATCH_SIZE=8
FORWARD_BATCH_SIZE=$((4 * MICRO_BATCH_SIZE))
OFFLOAD=false
PPO_EPOCHS=3

export VLLM_USE_V1=1
export WANDB_API_KEY=5cd6b6564c0f79ebf65ce694e5e3b54b725ff225
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=gae \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=4096 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.prompt_length=4096 \
    actor_rollout_ref.rollout.agent.default_agent_loop=multi_tool_agent_loop \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    trainer.balance_batch=False \
    trainer.critic_warmup=0 \
    trainer.critic_warmup_batch_repeat_times=4 \
    trainer.critic_warmup_batch_divide_ratio=1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='async-ticker-dryrun' \
    trainer.experiment_name='test-run' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=3 \
    trainer.val_before_train=False \
    envs.num_envs=${NUM_ENVS} \
    envs.env_name=async_ticker_admissions \
    +envs.env_config.professor_ids='["prof_1","prof_2","prof_3"]' \
    +envs.env_config.students_per_batch=2 \
    +envs.env_config.token_budget=1000 \
    +envs.env_config.feature_dim=5 \
    +envs.env_config.vote_threshold=0.5 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=256 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=256 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=256 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_epochs=${PPO_EPOCHS} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    critic.model.fsdp_config.param_offload=${OFFLOAD} \
    critic.model.fsdp_config.optimizer_offload=${OFFLOAD} \
    critic.ppo_max_token_len_per_gpu=256 \
    critic.forward_max_token_len_per_gpu=256 \
    critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    data.train_files=$HOME/data/gsm8k/test.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_files=$HOME/data/gsm8k/test.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    $@

# python3 -m verl.trainer.main_ppo \
#     --config-path="$CONFIG_PATH" \
#     --config-name='gsm8k_multiturn_grpo' \
#     algorithm.adv_estimator=gae \
#     data.train_batch_size=${BATCH_SIZE} \
#     data.max_prompt_length=512 \
#     data.max_response_length=256 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     data.return_raw_chat=True \
#     actor_rollout_ref.rollout.mode=async \
#     actor_rollout_ref.rollout.agent_loop_type=multi_tool_agent_loop \
#     actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
#     actor_rollout_ref.actor.entropy_coeff=0.001 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=False \
#     actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
#     actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
#     actor_rollout_ref.rollout.n=1 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=False \
#     trainer.balance_batch=False \
#     trainer.critic_warmup=2 \
#     trainer.critic_warmup_batch_repeat_times=4 \
#     trainer.critic_warmup_batch_divide_ratio=2 \
#     trainer.logger='["console"]' \
#     trainer.project_name='async-ticker-dryrun' \
#     trainer.experiment_name='test-run' \
#     trainer.n_gpus_per_node=1 \
#     trainer.nnodes=1 \
#     trainer.save_freq=-1 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=10 \
#     trainer.val_before_train=False \
#     envs.num_envs=${NUM_ENVS} \
#     envs.env_name=async_ticker_admissions \
#     +envs.env_config.professor_ids='["prof_1","prof_2","prof_3"]' \
#     +envs.env_config.students_per_batch=5 \
#     +envs.env_config.token_budget=500 \
#     +envs.env_config.feature_dim=5 \
#     +envs.env_config.vote_threshold=0.5 \
#     actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
#     actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
#     actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
#     critic.optim.lr=1e-5 \
#     critic.model.use_remove_padding=True \
#     critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
#     critic.model.enable_gradient_checkpointing=False \
#     critic.ppo_epochs=${PPO_EPOCHS} \
#     critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
#     critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
#     critic.model.fsdp_config.param_offload=${OFFLOAD} \
#     critic.model.fsdp_config.optimizer_offload=${OFFLOAD} \
#     critic.ppo_max_token_len_per_gpu=4096 \
#     critic.forward_max_token_len_per_gpu=4096 \
#     critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
#     data.train_files=$HOME/data/gsm8k/test.parquet \
#     data.val_files=$HOME/data/gsm8k/test.parquet \
#     data.train_files=$HOME/data/gsm8k/test.parquet \
#     data.val_files=$HOME/data/gsm8k/test.parquet \
#     $@
