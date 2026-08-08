#!/bin/bash
#SBATCH --job-name=verlog_sp_smoke
#SBATCH --output=/zfsauton/scratch/wentsec/Verlog_sp/logs/sp_smoke_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/Verlog_sp/logs/sp_smoke_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a6000:4

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate /zfsauton/scratch/wentsec/envs/verlog2
source /zfsauton/scratch/wentsec/.env_roll
cd /zfsauton/scratch/wentsec/Verlog_sp

ulimit -n 65535

export PYTHONPATH=/zfsauton/scratch/wentsec/Verlog_sp:$PYTHONPATH
export TRITON_CACHE_DIR=/zfsauton/scratch/wentsec/triton_cache
export RAY_TMPDIR=/zfsauton/scratch/wentsec/ray_tmp
export TMPDIR=/zfsauton/scratch/wentsec/tmp_ray_$$
mkdir -p $TMPDIR $TRITON_CACHE_DIR $RAY_TMPDIR

ray stop --force 2>/dev/null || true
sleep 2

NUM_GPUS_PER_NODE=4
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS_PER_NODE-1)))

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

NUM_ENVS=4
BATCH_SIZE=64
MINI_BATCH_SIZE=$((BATCH_SIZE))
MICRO_BATCH_SIZE=4
FORWARD_BATCH_SIZE=$((4 * MICRO_BATCH_SIZE))
OFFLOAD=True
PPO_EPOCHS=2

export VLLM_USE_V1=1

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=gae \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.1 \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.critic_warmup_batch_repeat_times=1 \
    trainer.critic_warmup_batch_divide_ratio=1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='zero' \
    trainer.experiment_name='debug_sp' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=30 \
    trainer.total_epochs=60 \
    trainer.val_before_train=False \
    envs.num_envs=${NUM_ENVS} \
    envs.env_name=babyai \
    envs.task=BabyAI-MixedTrainLocal-v0/pick_up_seq_go_to \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_epochs=${PPO_EPOCHS} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    critic.model.fsdp_config.param_offload=${OFFLOAD} \
    critic.model.fsdp_config.optimizer_offload=${OFFLOAD} \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.forward_max_token_len_per_gpu=8192 \
    critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    data.train_files=$HOME/data/mmlu/test.parquet \
    data.val_files=$HOME/data/mmlu/test.parquet \
    data.val_batch_size=${BATCH_SIZE} \
    $@


