# Verlog_v2

## Installation

* clone Verlog
```
cd ~
git clone git@github.com:WentseChen/Verlog.git
```

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
cd ~
cd Verlog
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

* download the GSM8k dataset
```
cd ~
cd Verlog
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

> you might have to login to your own wandb account when running the code for the first time. You can create a free wandb account at https://wandb.ai/site. After that, you can set up your wandb api key by running `wandb login <your_api_key>` in the terminal.

## Run

```
cd ~
cd Verlog
sbatch train.sbatch
```

## Important Files

* verl/trainer/ppo/ray_trainer.py
* verl/trainer/ppo/core_algos.py
* verl/experimental/agent_loop/agent_loop.py
* verl/experimental/agent_loop/tool_agent_loop.py