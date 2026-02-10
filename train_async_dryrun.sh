#!/bin/bash
# Dry run script for AsyncTickerAdmissionsEnv local testing
# This can be run locally without SLURM

set -e

# Setup environment
ulimit -n 65535

NUM_GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# Dry run parameters - very small for quick testing
NUM_ENVS=2
BATCH_SIZE=4
MINI_BATCH_SIZE=4
MICRO_BATCH_SIZE=2
FORWARD_BATCH_SIZE=4
OFFLOAD=false
PPO_EPOCHS=1

export VLLM_USE_V1=1

echo "========================================="
echo "AsyncTickerAdmissionsEnv Dry Run"
echo "========================================="
echo "NUM_ENVS: $NUM_ENVS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MODEL: Qwen/Qwen3-0.6B"
echo "TOTAL_EPOCHS: 10"
echo "========================================="
echo ""

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=gae \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent_loop_type=multi_tool_agent_loop \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.balance_batch=False \
    trainer.critic_warmup=2 \
    trainer.critic_warmup_batch_repeat_times=4 \
    trainer.critic_warmup_batch_divide_ratio=2 \
    trainer.logger='["console"]' \
    trainer.project_name='async-ticker-dryrun' \
    trainer.experiment_name='test-run' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.val_before_train=False \
    envs.num_envs=${NUM_ENVS} \
    envs.env_name=async_ticker_admissions \
    envs.env_config.professor_ids='["prof_1","prof_2","prof_3"]' \
    envs.env_config.students_per_batch=5 \
    envs.env_config.token_budget=500 \
    envs.env_config.feature_dim=5 \
    envs.env_config.vote_threshold=0.5 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen3-0.6B \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_epochs=${PPO_EPOCHS} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    critic.model.fsdp_config.param_offload=${OFFLOAD} \
    critic.model.fsdp_config.optimizer_offload=${OFFLOAD} \
    critic.ppo_max_token_len_per_gpu=4096 \
    critic.forward_max_token_len_per_gpu=4096 \
    critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    "$@"
