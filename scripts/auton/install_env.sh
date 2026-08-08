#!/bin/bash
#SBATCH --job-name=verlog_install
#SBATCH --output=/zfsauton/scratch/wentsec/Verlog_sp/logs/install_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/Verlog_sp/logs/install_%j.err
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a6000:1

set -x

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/verlog
VERLOG_DIR=/zfsauton/scratch/wentsec/Verlog_sp
BALROG_DIR=/zfsauton/scratch/wentsec/BALROG

source $CONDA_ROOT/etc/profile.d/conda.sh
source /zfsauton/scratch/wentsec/.env_roll

if [ ! -d "$ENV_PATH" ]; then
  conda create -y -p $ENV_PATH python=3.10
fi
conda activate $ENV_PATH

export TMPDIR=/zfsauton/scratch/wentsec/tmp_pip_$$
mkdir -p $TMPDIR
export PIP_CACHE_DIR=/zfsauton/scratch/wentsec/pip_cache

df -h /zfsauton/scratch /zfsauton2/home/wentsec
nvidia-smi

# ---- 1. BALROG (vendored env code in verl/envs imports gym / balrog helpers) ----
if [ ! -d "$BALROG_DIR" ]; then
  git clone https://github.com/balrog-ai/BALROG.git $BALROG_DIR
fi
cd $BALROG_DIR
pip install -e .
balrog-post-install || echo "WARN: balrog-post-install failed (only needed for nle/textworld envs)"

# ---- 2. verl inference stack (pins torch 2.6.0 / vllm 0.8.5.post1) ----
cd $VERLOG_DIR
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# ---- 3. verl itself ----
pip install --no-deps -e $VERLOG_DIR

# sglang[all] drags in torchao>=0.18, which calls torch.utils._pytree.register_constant
# (torch>=2.8 only) at import time and breaks transformers.modeling_utils on torch 2.6.
pip install --no-cache-dir "torchao==0.9.0"

# ---- 4. verify ----
python -c "
import torch, vllm, transformers
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())
print('vllm', vllm.__version__)
print('transformers', transformers.__version__)
import verl; print('verl ok')
from verl.envs.sp_env import get_mmlu_env; print('sp_env ok')
"

# ---- 5. MMLU dataset ----
python3 $VERLOG_DIR/examples/data_preprocess/mmlu.py --local_save_dir $HOME/data/mmlu

df -h /zfsauton/scratch /zfsauton2/home/wentsec
echo "===== INSTALL DONE ====="
