#!/bin/bash
#SBATCH --job-name=verlog_install2
#SBATCH --output=/zfsauton/scratch/wentsec/Verlog_sp/logs/install2_%j.out
#SBATCH --error=/zfsauton/scratch/wentsec/Verlog_sp/logs/install2_%j.err
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a6000:1

# Both of the branch's own version statements are stale:
#   - scripts/install_vllm_sglang_mcore.sh pins vllm 0.8.5.post1 / torch 2.6, but
#     vllm_async_server.py imports vllm.utils.get_tcp_uri  (vllm >= 0.9)
#   - setup.py declares vllm<=0.9.1, but the same file imports
#     vllm.v1.engine.utils.CoreEngineProcManager  (vllm >= 0.10)
# Also, vllm 0.9.1 + transformers>=4.54 collide on the 'aimv2' config name.
# Landing on torch 2.8.0 + vllm 0.10.2 (the combo already working in envs/roll2).
# sglang is deliberately omitted (it hard-pins torch 2.6; rollout.name=vllm here).

set -x

CONDA_ROOT=/zfsauton/scratch/wentsec/miniconda3
ENV_PATH=/zfsauton/scratch/wentsec/envs/verlog2
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

# ---- 1. BALROG (verl/envs vendored code imports gym + balrog helpers) ----
[ -d "$BALROG_DIR" ] || git clone https://github.com/balrog-ai/BALROG.git $BALROG_DIR
cd $BALROG_DIR
pip install setuptools   # balrog-post-install needs pkg_resources
pip install -e .
balrog-post-install || echo "WARN: balrog-post-install failed (only needed for nle/textworld)"

# ---- 2. inference stack ----
pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir "vllm==0.10.2"
pip install --no-cache-dir "tensordict==0.10.0" torchdata

# ---- 3. basic packages (from scripts/install_vllm_sglang_mcore.sh, minus pins) ----
pip install --no-cache-dir "transformers[hf_xet]==4.57.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff tensorboard \
    latex2sympy2_extended math_verify \
    "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

# ---- 4. FlashAttention (torch 2.8 build) ----
cd $TMPDIR
FA_WHL=flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$FA_WHL
pip install --no-cache-dir $FA_WHL

pip install --no-cache-dir opencv-python

# ---- 5. verl ----
pip install --no-deps -e $VERLOG_DIR

# ---- 6. verify ----
python -c "
import torch, vllm, transformers, tensordict, flash_attn
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())
print('vllm', vllm.__version__, 'transformers', transformers.__version__)
print('tensordict', tensordict.__version__, 'flash_attn', flash_attn.__version__)
from vllm.utils import get_tcp_uri; print('get_tcp_uri ok')
import verl; print('verl ok')
from verl.envs.sp_env import get_mmlu_env; print('sp_env ok')
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica; print('vllm replica ok')
from vllm.v1.engine.utils import CoreEngineProcManager; print('CoreEngineProcManager ok')
from verl.trainer.main_ppo import main; print('main_ppo ok')
"

df -h /zfsauton/scratch /zfsauton2/home/wentsec
echo "===== INSTALL2 DONE ====="
