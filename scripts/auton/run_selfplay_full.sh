#!/bin/bash
#SBATCH --job-name=verlog_sp_full
#SBATCH --output=/zfsauton/scratch/wentsec/Verlog_sp/logs/sp_full_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/Verlog_sp/logs/sp_full_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
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

# Rollout concurrency. Each generate_sequences call is sliced to num_envs prompts
# (agent_loop.py), so BATCH_SIZE/NUM_ENVS = number of sequential rollout waves:
# 32 envs meant 8 waves with only ~8 sequences decoding per GPU, deep in the
# memory-bandwidth-bound regime. 256 = one wave. Envs are now hosted several per
# worker, so raising this does NOT raise the process/CPU count.
NUM_ENVS=256
NUM_WORKERS=32
BATCH_SIZE=256
MINI_BATCH_SIZE=$((BATCH_SIZE))
MICRO_BATCH_SIZE=4
FORWARD_BATCH_SIZE=$((4 * MICRO_BATCH_SIZE))
# One optimizer step per collected batch (ppo_mini_batch_size == train_batch_size,
# so 1 mini-batch per epoch). Benchmarked 2026-08-07: 158.9 s/step -> 126.9 s/step
# (-20%); the actor/critic updates each fell ~39%, not 50%, because the per-update
# FSDP all-gather + optimizer cost does not scale with epochs.
# TRADEOFF: half the gradient steps per sample. Wall-clock per step is not
# time-to-quality -- validate against the exploitability curve before trusting it.
PPO_EPOCHS=1
# critic_warmup: 40 -> 20. The value head converges by ~step 19 in both the
# historical run and a fresh one (vf_explained_var -16.3 -> ~0.01, vf_loss
# 0.30 -> 0.02, then flat), so the second 20 steps were pure cost (~79 s each,
# the actor is frozen during warmup).

# ---- throughput settings (benchmarked 2026-08-07: 583 s/step -> 262 s/step) ----
# Only the OPTIMIZER state is kept off-GPU-offload; offloading it ran Adam on the
# CPU and left the GPUs at 0% util (that config also OOM'd the 240G cgroup at
# step 41 and hung). param_offload stays True: turning it off measured as a wash
# (289 s vs 281 s) and only costs GPU memory.
PARAM_OFFLOAD=True
OPT_OFFLOAD=False
# Dynamic batching packs equal-token micro-batches (Karmarkar-Karp) instead of a
# fixed 4 sequences whose token counts varied 3-4x. Verified not to disturb the
# z_content ref-logprob splice: actor/reward_kl_penalty stayed inside the
# offline band 1.4259 +/- 0.0223 across all runs.
MAX_TOKEN_LEN=16384        # fwd+bwd budget (actor + critic)
FWD_TOKEN_LEN=32768        # forward-only passes carry no grads -> 2x budget
MAX_NUM_BATCHED_TOKENS=16384
# gradient checkpointing MUST stay on: disabling it OOM'd at 47.3/47.4 GiB.
# gpu_memory_utilization stays low: 0.35 -> 0.6 measured zero rollout gain.
# 256-way concurrency needs ~66k KV tokens/GPU (64 seqs x 1024); 0.35 supplies only
# ~63k, so vLLM admitted ~24 of 256 requests and queued the rest. 0.5 gives ~114k.
GPU_MEM_UTIL=0.5

export VLLM_USE_V1=1

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=gae \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=2304 \
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
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPT_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_WORKERS} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.1 \
    trainer.balance_batch=True \
    trainer.critic_warmup=20 \
    trainer.critic_warmup_batch_repeat_times=1 \
    trainer.critic_warmup_batch_divide_ratio=1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='zero' \
    trainer.experiment_name='selfplay' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=30 \
    trainer.total_epochs=60 \
    trainer.val_before_train=True \
    envs.num_envs=${NUM_ENVS} \
    envs.env_name=babyai \
    envs.task=BabyAI-MixedTrainLocal-v0/pick_up_seq_go_to \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${FWD_TOKEN_LEN} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${FWD_TOKEN_LEN} \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_epochs=${PPO_EPOCHS} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    critic.use_dynamic_bsz=True \
    critic.model.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    critic.model.fsdp_config.optimizer_offload=${OPT_OFFLOAD} \
    critic.model.fsdp_config.forward_prefetch=True \
    critic.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN} \
    critic.forward_max_token_len_per_gpu=${FWD_TOKEN_LEN} \
    critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    data.train_files=$HOME/data/mmlu/test.parquet \
    data.val_files=$HOME/data/mmlu/test.parquet \
    data.val_batch_size=${BATCH_SIZE} \
    $@


