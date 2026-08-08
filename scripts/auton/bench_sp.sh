#!/bin/bash
#SBATCH --job-name=sp_bench
#SBATCH --output=/zfsauton/scratch/wentsec/Verlog_sp/logs/bench_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/Verlog_sp/logs/bench_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a6000:4

# Throughput bench: runs $STEPS steps of the self-play demo and exits.
# Knobs are env-overridable so variants can be submitted with
#   sbatch --export=ALL,VARIANT=v2,MAX_TOKEN_LEN=24576 scripts/auton/bench_sp.sh
# Baseline (v0 = run_selfplay_full.sh as shipped): 583 s/step.

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

NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-4}
unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS_PER_NODE-1)))

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

# ---- knobs (env-overridable) ----
VARIANT=${VARIANT:-v1}
STEPS=${STEPS:-8}
NUM_ENVS=${NUM_ENVS:-32}
# rollout concurrency (NUM_ENVS) is now decoupled from the Ray actor/process count
# (NUM_WORKERS); each worker hosts NUM_ENVS/NUM_WORKERS envs driven by asyncio.
NUM_WORKERS=${NUM_WORKERS:-${NUM_ENVS}}
BATCH_SIZE=${BATCH_SIZE:-256}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
PPO_EPOCHS=${PPO_EPOCHS:-1}
DYNAMIC_BSZ=${DYNAMIC_BSZ:-True}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-16384}
# forward-only passes carry no gradients, so their token budget can exceed the
# fwd+bwd budget (docs/perf/perf_tuning.rst recommends 2-4x)
FWD_TOKEN_LEN=${FWD_TOKEN_LEN:-${MAX_TOKEN_LEN}}
CRITIC_TOKEN_LEN=${CRITIC_TOKEN_LEN:-${MAX_TOKEN_LEN}}
# the critic head is a scalar projection (no vocab), so its forward pass has no
# [tokens x 151936] logits tensor and its budget can far exceed the actor's
CRITIC_FWD_TOKEN_LEN=${CRITIC_FWD_TOKEN_LEN:-${FWD_TOKEN_LEN}}
ENTROPY_CHUNKING=${ENTROPY_CHUNKING:-False}
FORWARD_PREFETCH=${FORWARD_PREFETCH:-False}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPT_OFFLOAD=${OPT_OFFLOAD:-False}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.35}

MINI_BATCH_SIZE=$((BATCH_SIZE))
FORWARD_BATCH_SIZE=$((4 * MICRO_BATCH_SIZE))

# ---- capture vLLM's own logs ----
# vLLM's logger runs inside the vLLMHttpServer Ray actor (and its EngineCore
# subprocess), so its output never reaches the job's stdout/stderr and Ray's log
# dedup swallows what does. Give vLLM an explicit FileHandler instead; this is the
# only way to see the scheduler's "Running: N reqs, Waiting: M reqs" line and the
# "GPU KV cache size" / "Maximum concurrency" init lines.
VLLM_LOG=/zfsauton/scratch/wentsec/Verlog_sp/logs/vllm_${SLURM_JOB_ID}.log
cat > $TMPDIR/vllm_logging.json <<JSON
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "plain": {"class": "logging.Formatter", "format": "%(asctime)s %(levelname)s %(name)s %(message)s"}
  },
  "handlers": {
    "vllm_file": {
      "class": "logging.FileHandler",
      "formatter": "plain",
      "filename": "$VLLM_LOG",
      "mode": "a"
    }
  },
  "loggers": {
    "vllm": {"handlers": ["vllm_file"], "level": "INFO", "propagate": false}
  },
  "root": {}
}
JSON
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH=$TMPDIR/vllm_logging.json
export VLLM_LOGGING_LEVEL=INFO
# Ray dedups repeated actor log lines ("[repeated Nx across cluster]"), which is
# exactly what the periodic scheduler stat line looks like.
export RAY_DEDUP_LOGS=0
echo "vLLM log -> $VLLM_LOG"

export VLLM_USE_V1=1

echo "===== BENCH $VARIANT ====="
echo "steps=$STEPS envs=$NUM_ENVS batch=$BATCH_SIZE micro=$MICRO_BATCH_SIZE epochs=$PPO_EPOCHS"
echo "dynamic_bsz=$DYNAMIC_BSZ max_token_len=$MAX_TOKEN_LEN"
echo "param_offload=$PARAM_OFFLOAD opt_offload=$OPT_OFFLOAD gpu_mem_util=$GPU_MEM_UTIL"
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader

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
    actor_rollout_ref.actor.use_dynamic_bsz=${DYNAMIC_BSZ} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPT_OFFLOAD} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_WORKERS} \
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
    trainer.experiment_name="bench_${VARIANT}" \
    trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=60 \
    trainer.total_training_steps=${STEPS} \
    trainer.val_before_train=False \
    envs.num_envs=${NUM_ENVS} \
    envs.env_name=babyai \
    envs.task=BabyAI-MixedTrainLocal-v0/pick_up_seq_go_to \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${FWD_TOKEN_LEN} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${FWD_TOKEN_LEN} \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=${FORWARD_PREFETCH} \
    critic.model.fsdp_config.forward_prefetch=${FORWARD_PREFETCH} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_epochs=${PPO_EPOCHS} \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    critic.use_dynamic_bsz=${DYNAMIC_BSZ} \
    critic.model.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    critic.model.fsdp_config.optimizer_offload=${OPT_OFFLOAD} \
    critic.ppo_max_token_len_per_gpu=${CRITIC_TOKEN_LEN} \
    critic.forward_max_token_len_per_gpu=${CRITIC_FWD_TOKEN_LEN} \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=${ENTROPY_CHUNKING} \
    critic.forward_micro_batch_size_per_gpu=${FORWARD_BATCH_SIZE} \
    data.train_files=$HOME/data/mmlu/test.parquet \
    data.val_files=$HOME/data/mmlu/test.parquet \
    data.val_batch_size=${BATCH_SIZE} \
    $@

echo "===== BENCH $VARIANT DONE ====="
