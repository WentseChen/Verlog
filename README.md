<h1 style="text-align: center;">Verlog</h1>

[![GitHub Repo stars](https://img.shields.io/github/stars/WentseChen/verl)](https://github.com/WentseChen/verl/stargazers)
![GitHub forks](https://img.shields.io/github/forks/WentseChen/verl)
<!-- [![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project) -->
<!-- <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a> -->
<!-- <a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a> -->
<!-- ![GitHub contributors](https://img.shields.io/github/contributors/WentseChen/verl) -->
<!-- [![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/) -->
<!-- <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a> -->

Verlog is a well-tuned multi-turn RL framework built for long-horizon LLM agentic tasks. It extends [VeRL](https://github.com/volcengine/verl) and [BALROG](https://github.com/balrog-ai/BALROG), and follows the core design principles of [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail), while introducing tailored modifications for efficient multi-turn learning.

## Key features:  

⏳ Fixed-Turn Batching: For every training batch, a fixed number of turns is collected. If the episode has not terminated, we use the value function instead of relying on final rewards. 

🧠 Turn-Level Abstraction: Each turn is treated as an independent data point—no need to pack the entire history into the context window. Customize your memory mechanism as needed. 

🚀 Optimized for Long-Horizon Agentic Tasks: Verlog incorporates techniques like Dual Discounting GAE and Critic Pre-training, along with carefully tuned hyperparameters, to ensure strong performance on challenging long-horizon multi-turn benchmarks such as BALROG.

## Main Results

All the experiments are done with Qwen2.5-3B-Instruct model, PPO on 4xA40 GPUs with 48Gb memory for 24 hours.


## Installation

* create a conda environment
```bash
conda create -n verlog python=3.10
conda activate verlog
```

* install Balrog as a temporary dependency
```bash
git clone https://github.com/balrog-ai/BALROG.git
cd BALROG
pip install -e .
balrog-post-install
```

* install Verlog
```bash
# 1. Clone this repository
# 2. If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
# 3. Install Verlog
pip install --no-deps -e .
```

## Get Started

We provide a simple example to get started with Verlog in `train.sbatch`
