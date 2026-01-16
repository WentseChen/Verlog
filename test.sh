
source /u/wchen11/anaconda3/bin/activate 
conda activate verlog
cd /u/wchen11/Verlog

NUM_GPUS_PER_NODE=4
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS_PER_NODE-1)))

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

NUM_ENVS=32
BATCH_SIZE=256
MINI_BATCH_SIZE=$((BATCH_SIZE / 2))
MICRO_BATCH_SIZE=8 
FORWARD_BATCH_SIZE=$((4 * MICRO_BATCH_SIZE))
OFFLOAD=false
PPO_EPOCHS=1

export HF_HOME=$HOME/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1

MODEL_NAME="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

export VLLM_USE_V1=1

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=gae \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=1536 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.model.path=${MODEL_NAME} \
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
    actor_rollout_ref.rollout.skip_loop_rate=0.5 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_ENVS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    trainer.balance_batch=False \
    trainer.critic_warmup=2 \
    trainer.critic_warmup_batch_repeat_times=2 \
    trainer.critic_warmup_batch_divide_ratio=2 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='zero' \
    trainer.experiment_name='debug' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=30 \
    trainer.total_epochs=60 \
    trainer.val_before_train=False \
    envs.num_envs=${NUM_ENVS} \
    envs.env_name=babaisai \
    envs.task=env/two_room-maybe_break_stop-goto_win \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_NAME} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_epochs=${PPO_EPOCHS} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    critic.model.fsdp_config.param_offload=${OFFLOAD} \
    critic.model.fsdp_config.optimizer_offload=${OFFLOAD} \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.forward_max_token_len_per_gpu=8192 \
    critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    data.train_files=$HOME/data/gsm8k/test.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    $@


