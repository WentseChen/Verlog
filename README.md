<h1 style="text-align: center;">Verlog</h1>

[![GitHub Repo stars](https://img.shields.io/github/stars/WentseChen/verl)](https://github.com/WentseChen/verl/stargazers)
![GitHub forks](https://img.shields.io/github/forks/WentseChen/verl)
<!-- [![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project) -->
<!-- <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a> -->
<!-- <a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a> -->
<!-- ![GitHub contributors](https://img.shields.io/github/contributors/WentseChen/verl) -->
<!-- [![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/) -->
<!-- <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a> -->

Verlog is a well-tuned multi-turn RL (PPO) framework built for long-horizon LLM agentic tasks. It extends VeRL and BALROG, and follows the core design principles of pytorch-ppo, while introducing tailored modifications for efficient multi-turn learning.

## Key features:  

⏳ Fixed-Turn Batching: For every training batch, a fixed number of turns is collected. If the episode has not terminated, we use the value function instead of relying on final rewards. 

🧠 Turn-Level Abstraction: Each turn is treated as an independent data point—no need to pack the entire history into the context window. Customize your memory mechanism as needed. 

🚀 Optimized for Long-Horizon Agentic Tasks: Verlog incorporates techniques like Dual Discounting GAE and Critic Pre-training, along with carefully tuned hyperparameters, to ensure strong performance on challenging long-horizon multi-turn benchmarks such as BALROG.

## Main Results

All the experiments are done with Qwen2.5-3B-Instruct model, PPO on 4xA40 GPUs with 48Gb memory for 24 hours.




## System Design Overview

<p align="center">

<img src="step_level_gae.gif" width="600" alt="Step Level GAE">

Similar to classic RL libraries such as [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/), LLM\_agents\_PPO provides a step-level framework tailored for LLM agents.

1. LLM\_agents\_PPO adopts a step-level approach where each step serves as an individual training data point, rather than treating entire episodes as a training data point. This approach allows for **customizing the memory mechanism** specific to each step, which is particularly beneficial for tackling long horizon tasks.

2. One key feature of LLM\_agents\_PPO is the ability to **early truncate rollouts** at any step, enhancing training efficiency and enabling better management of long horizon tasks. This method uses the value of the subsequent step as the supervised signal for the truncated rollout.

3. The framework supports **step-level Generalized Advantage Estimation (GAE)**, decoupling step-level $\lambda_{\text{step}}, \gamma_{\text{step}}$ from token level $\lambda_{\text{token}}, \gamma_{\text{token}}$. This capability improves credit assignment accuracy and overall training efficiency.

This setup is optimized specifically for LLM agents engaged in complex, long-term decision-making tasks.

</p>

## Results

<p align="center">
</p>

## Getting Started



