# Verlog_v2

## Installation

* create conda environment
```
conda create -n verlog python==3.10
conda activate verlog
```

* install balrog
```
cd ~
git clone https://github.com/balrog-ai/BALROG.git
cd BALROG
pip install -e .
balrog-post-install
```

* install verl
```
cd Verlog
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

## Run

```
cd Verlog
sbatch train.sbatch
```

## Important Files

* verl/trainer/ppo/ray_trainer.py
* verl/trainer/ppo/core_algos.py
* verl/experimental/agent_loop/agent_loop.py
* verl/experimental/agent_loop/tool_agent_loop.py