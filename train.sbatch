#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/my_job-%j.out
#SBATCH --error=logs/my_job-%j.err
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1024G
#SBATCH --time=12:00:00
#SBATCH --requeue

### IMPORTANT SETTINGS AND ENVIRONMENT SETUP ###
export HF_TOKEN="hf_xxxxyourtokenherexxxx"
export MODEL_CACHE_DIR="/project/flame/wentsec"

###############################################################################

set -euo pipefail
IFS=$'\n\t'

echo "===== JOB START ====="
date
hostname
pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

# =============================================================================
# CRITICAL: Unset AMD ROCm variables that conflict with CUDA
# =============================================================================
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
echo "Unset ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES to avoid conflicts"

# =============================================================================
# CRITICAL: Fast node-local storage for Ray/vLLM (this fixes the EOF error)
# =============================================================================
# On Orchard/FLAME, SLURM_TMPDIR points to fast local NVMe (/mnt/local, /tmp/local, etc.)
if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "$SLURM_TMPDIR" ]; then
    export FAST_TMP="$SLURM_TMPDIR"
    echo "Using SLURM_TMPDIR (node-local NVMe): $FAST_TMP"
else
    # Fallback locations that exist on most clusters
    for candidate in /mnt/local /tmp/local /local /scratch/local; do
        if [ -d "$candidate" ]; then
            export FAST_TMP="$candidate/ray_$SLURM_JOB_ID"
            mkdir -p "$FAST_TMP"
            echo "Falling back to node-local dir: $FAST_TMP"
            break
        fi
    done
    # Absolute worst case
    [ -z "${FAST_TMP:-}" ] && export FAST_TMP="/tmp/ray_$SLURM_JOB_ID" && mkdir -p "$FAST_TMP"
fi

# Force EVERY temp-related variable to point to the fast storage
export TMPDIR="$FAST_TMP"
export TMP="$FAST_TMP"
export TEMP="$FAST_TMP"
export TEMPDIR="$FAST_TMP"
export RAY_TMPDIR="$FAST_TMP/ray"
export VLLM_TMPDIR="$FAST_TMP/vllm"
export XDG_CACHE_HOME="$FAST_TMP/xdg_cache"

mkdir -p "$RAY_TMPDIR" "$VLLM_TMPDIR" "$XDG_CACHE_HOME"
chmod 1777 "$FAST_TMP" "$RAY_TMPDIR" "$VLLM_TMPDIR" "$XDG_CACHE_HOME"

# Default checkpoint directory on fast storage
export DEFAULT_LOCAL_DIR="/tmp/checkpoints"
mkdir -p "$DEFAULT_LOCAL_DIR"
chmod 1777 "$DEFAULT_LOCAL_DIR"

echo "Final temp directories (all on fast storage):"
echo "   FAST_TMP        = $FAST_TMP"
echo "   RAY_TMPDIR      = $RAY_TMPDIR"
echo "   VLLM_TMPDIR     = $VLLM_TMPDIR"
echo "   XDG_CACHE_HOME  = $XDG_CACHE_HOME"

# Clean any leftover Ray sessions from previous crashed jobs
rm -rf "$RAY_TMPDIR"/session_* "$RAY_TMPDIR"/plasma_store* 2>/dev/null || true

# Increase file descriptors (Ray loves sockets)
ulimit -n 65536 || true

# =============================================================================
# CUDA / GPU environment
# =============================================================================
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "CUDA_VISIBLE_DEVICES already set: $CUDA_VISIBLE_DEVICES"
elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_ON_NODE-1)))
else
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS_PER_NODE-1)))
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# =============================================================================
# HuggingFace cache (keep on shared storage)
# =============================================================================
export HF_HOME="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$MODEL_CACHE_DIR"
export HF_TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
mkdir -p "$HF_HOME"

# =============================================================================
# Conda environment
# =============================================================================
if [ -f "/home/wentsec/anaconda3/etc/profile.d/conda.sh" ]; then
    source /home/wentsec/anaconda3/etc/profile.d/conda.sh
else
    source /opt/conda/etc/profile.d/conda.sh || true
fi
conda activate verlog

# Verify GPU + environment
nvidia-smi
echo "Python executable: $(which python)"
python -c "import sys, torch; print('Python:', sys.executable); print('Torch:', torch.__version__, torch.cuda.is_available())"

# =============================================================================
# Ray + NCCL fixes (prevents 99% of Ray "End of file" errors)
# =============================================================================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Adjust if your cluster uses a different InfiniBand interface
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN   # change to INFO only for debugging

# =============================================================================
# CUDA paths (adjust if needed)
# =============================================================================
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# =============================================================================
# Training hyperparameters (unchanged)
# =============================================================================
cd /home/wentsec/Verlog
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

NUM_ENVS=32
BATCH_SIZE=512
MINI_BATCH_SIZE=$((BATCH_SIZE / 2))
MICRO_BATCH_SIZE=8
FORWARD_BATCH_SIZE=$((4 * MICRO_BATCH_SIZE))
OFFLOAD=false
PPO_EPOCHS=1
VLLM_USE_V1=1

umask 0002

# =============================================================================
# Google Cloud checkpoint sync configuration
# =============================================================================

GCS_BUCKET="gs://cmu-gpucloud-wentsec"
GCS_CHECKPOINT_PATH="${GCS_BUCKET}/checkpoints/job_${SLURM_JOB_ID}"
CHECK_INTERVAL=60  # Check for new checkpoints every 60 seconds
LOCAL_CHECKPOINT_DIR="$DEFAULT_LOCAL_DIR"

if gcloud storage ls "${GCS_CHECKPOINT_PATH}" &>/dev/null; then
    echo "Found existing checkpoints in GCS, syncing..."
    gcloud storage rsync "${GCS_CHECKPOINT_PATH}" "$LOCAL_CHECKPOINT_DIR" --recursive
else
    echo "No existing checkpoints in GCS, starting fresh"
fi

# =============================================================================
# Start background checkpoint monitoring and sync process
# =============================================================================
(
    SYNC_LOG="$LOCAL_CHECKPOINT_DIR/gcloud_sync.log"
    
    echo "[$(date)] Starting checkpoint monitor for job $SLURM_JOB_ID" | tee -a "$SYNC_LOG"
    echo "[$(date)] Watching: $LOCAL_CHECKPOINT_DIR" | tee -a "$SYNC_LOG"
    echo "[$(date)] Syncing to: $GCS_CHECKPOINT_PATH" | tee -a "$SYNC_LOG"
    
    LAST_TRACKER="$LOCAL_CHECKPOINT_DIR/.last_iteration_synced"

    # Initialize LAST if tracker doesn't exist
    if [ -f "$LAST_TRACKER" ]; then
        LAST=$(cat "$LAST_TRACKER")
    else
        LAST=-1
    fi

    while true; do
        # Read latest iteration index
        LATEST_TXT="$LOCAL_CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
        if [ -f "$LATEST_TXT" ]; then
            n=$(cat "$LATEST_TXT")
        else
            sleep "$CHECK_INTERVAL"
            continue
        fi

        # If changed, upload new global_step_n and delete old global_step_(n-1)
        if [ "$n" != "$LAST" ]; then
            ckpt_dir="$LOCAL_CHECKPOINT_DIR/global_step_${n}"

            if [ -d "$ckpt_dir" ]; then
                echo "[$(date)] New checkpoint iteration detected: $n" | tee -a "$SYNC_LOG"

                # Upload new checkpoint
                echo "[$(date)] Uploading global_step_${n}..." | tee -a "$SYNC_LOG"
                gcloud storage rsync "$ckpt_dir" "${GCS_CHECKPOINT_PATH}/global_step_${n}" --recursive 2>&1 | tee -a "$SYNC_LOG"
                status=$?

                gcloud storage cp "$LATEST_TXT" "${GCS_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt"

                if [ $status -eq 0 ]; then
                    echo "[$(date)] ✓ Successfully synced global_step_${n}" | tee -a "$SYNC_LOG"

                    # Remove previous checkpoint directory
                    old_dir="$LOCAL_CHECKPOINT_DIR/global_step_${LAST}"
                    if [ "$LAST" != "-1" ] && [ -d "$old_dir" ]; then
                        echo "[$(date)] Deleting old checkpoint global_step_${LAST}" | tee -a "$SYNC_LOG"
                        gcloud storage rm -r "${GCS_CHECKPOINT_PATH}/global_step_${LAST}"
                        rm -rf "$old_dir"
                    fi
                    echo "$n" > "$LAST_TRACKER"
                    LAST="$n"
                else
                    echo "[$(date)] ✗ Failed to sync global_step_${n}" | tee -a "$SYNC_LOG"
                fi
            else
                echo "[$(date)] global_step_${n} does not exist yet, waiting..." | tee -a "$SYNC_LOG"
            fi
        fi

        sleep "$CHECK_INTERVAL"
    done
) &

SYNC_PID=$!
echo "Started checkpoint sync process (PID: $SYNC_PID)"


# =============================================================================
# Launch training
# =============================================================================
python3 -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name='gsm8k_multiturn_grpo' \
  algorithm.adv_estimator=gae \
  data.train_batch_size=${BATCH_SIZE} \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
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
  algorithm.use_kl_in_reward=True \
  algorithm.kl_ctrl.type='linear_decay' \
  algorithm.kl_ctrl.init_kl_coef=0.003 \
  algorithm.kl_ctrl.final_kl_coef=-1.0 \
  trainer.balance_batch=False \
  trainer.critic_warmup=20 \
  trainer.critic_warmup_batch_repeat_times=50 \
  trainer.critic_warmup_batch_divide_ratio=5 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='zero' \
  trainer.experiment_name='debug' \
  trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
  trainer.test_freq=30 \
  trainer.total_epochs=150 \
  trainer.val_before_train=True \
  envs.num_envs=${NUM_ENVS} \
  envs.env_name=crafter \
  envs.task=default \
  envs.binary_reward=False \
  envs.reset_rate=0.0 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  critic.optim.lr=1e-5 \
  critic.model.use_remove_padding=True \
  critic.model.path=Qwen/Qwen2.5-7B-Instruct \
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
  "$@" 

# Trap to ensure sync process is killed when job ends
cleanup() {
    echo "Cleaning up checkpoint sync process..."
    kill $SYNC_PID 2>/dev/null || true
    wait $SYNC_PID 2>/dev/null || true
    
    # Final sync of entire checkpoint directory
    echo "Performing final checkpoint sync..."
    if [ -d "$DEFAULT_LOCAL_DIR" ]; then
        gcloud storage rsync "$DEFAULT_LOCAL_DIR" "$GCS_CHECKPOINT_PATH" --recursive
        FINAL_UPLOAD_STATUS=$?
        if [ $FINAL_UPLOAD_STATUS -eq 0 ]; then
            echo "✓ Final sync completed successfully (status: $FINAL_UPLOAD_STATUS)"
        else
            echo "✗ Final sync failed (status: $FINAL_UPLOAD_STATUS)"
        fi
    fi
}
trap cleanup EXIT INT TERM

echo "===== JOB END ====="
date


