<h1 style="text-align: center;">Verlog</h1>

<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 24px; width: 100%;">
  <a href="https://wentsechen.github.io/Verlog_blogpost/" target="_blank" style="text-decoration: none; display: flex; align-items: center;">
    <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/github.svg" alt="GitHub" width="30" height="30" style="vertical-align: middle;">
    <span style="vertical-align: middle; font-size: 16px; margin-left: 6px;">Blogpost</span>
  </a>

  <a href="https://wandb.ai/cwz19/verlog?nw=nwusercwz19" target="_blank" style="text-decoration: none; display: flex; align-items: center;">
    <img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg" alt="W&B" width="30" height="30" style="vertical-align: middle;">
    <span style="vertical-align: middle; font-size: 16px; margin-left: 6px;">Experiment Logs</span>
  </a>
</div>

Verlog is a multi-turn reinforcement learning framework built for **long-horizon LLM-agentic tasks with highly variable episode lengths**. Extending [VeRL](https://github.com/volcengine/verl) and [BALROG](https://github.com/balrog-ai/BALROG) while following the proven design principles of [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail), it introduces specialized optimizations for stable and efficient training when episodes span from short interactions to hundreds of turns. Whereas prior frameworks like [VeRL](https://github.com/volcengine/verl) and [RAGEN](https://ragen-ai.github.io) effectively handle tasks with ~10 turns, and [verl-agent](https://github.com/langfengQ/verl-agent) scales up to ~50 turns, Verlog is designed to operate in environments reaching **up to 400 turns**, making it uniquely suited for complex, long-duration decision-making. This capability has been validated across challenging domains such as BabyAI, BabaIsAI, and Crafter, where it consistently achieves strong performance out of the box. In Crafter, for instance, episode lengths range from 70 to 400 steps with an average of about 190.

## Key features:  

🧠 Turn-Level Abstraction: To handle extremely long episodes, we treat each turn as an independent training sample. This eliminates the need to encode the entire trajectory into a single context window and allows for modular, customizable memory architectures.

🎯 Fixed-Turn Batching: To address the high variance in episode lengths across environments, we use fixed-turn batching. Each training batch contains a fixed number of turns. For incomplete episodes, we replace final rewards with value function estimates as the supervision signal.

🛠️ Tailored for Multi-Turn RL: To address the unique challenges of multi-turn RL, we introduce a set of targeted techniques such as Dual Discounting GAE and Critic Pre-training, combined with carefully tuned hyperparameters to ensure efficient and stable learning.


## Main Results

* Crafter Results:
  
  <div style="overflow-x: auto;">
    <table style="width: 100% !important; min-width: 300px !important; border-collapse: collapse !important; font-family: sans-serif !important;">
        <thead>
            <tr>
                <th style="padding: 8px !important; text-align: center !important; white-space: nowrap !important; border-bottom: 2px solid #cccccc !important; background-color: #cccccc !important; color: #000 !important;">Metric</th>
                <th style="padding: 8px !important; text-align: center !important; white-space: nowrap !important; border-bottom: 2px solid #cccccc !important; background-color: #cccccc !important; color: #000 !important;">Instruct-model</th>
                <th style="padding: 8px !important; text-align: center !important; white-space: nowrap !important; border-bottom: 2px solid #cccccc !important; background-color: #cccccc !important; color: #000 !important;">Verlog (Ours)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 8px !important; text-align: center !important; white-space: nowrap !important; border-bottom: 1px solid #cccccc !important;">Rewards</td>
                <td style="padding: 8px !important; text-align: center !important; white-space: nowrap !important; border-bottom: 1px solid #cccccc !important;">5.80</td>
                <td style="padding: 8px !important; text-align: center !important; white-space: nowrap !important; border-bottom: 1px solid #cccccc !important; font-weight: bold !important;">10.44</td>
            </tr>
            <tr>
                <td style="padding: 8px !important; text-align: center !important; white-space: nowrap !important;">Trajectory Length</td>
                <td style="padding: 8px !important; text-align: center !important; white-space: nowrap !important;">172.23</td>
                <td style="padding: 8px !important; text-align: center !important; white-space: nowrap !important; font-weight: bold !important;">196.42</td>
            </tr>
        </tbody>
    </table>
  </div>


    > Crafter's experiments are done with Qwen2.5-7B-Instruct model, using PPO algorithm, trained on 8xH100 GPUs with 82Gb memory for ~36 hours, corresponding to 170 PPO updates.


* BabaIsAI Results (win rate)

    goto_win → 🏁; 
    distr_obj → 🎁; 
    two_room → 🚪; 
    distr_obj_rule → 📏;  
    maybe_break_stop → ⚠️;

  <div style="overflow-x: auto;">
      <table style="width: 100%; border-collapse: collapse; text-align: center; font-family: sans-serif;">
          <thead>
              <tr>
                  <th style="padding: 10px; border-bottom: 2px solid #ddd; background-color: #cccccc !important; color: #000;">Model</th>
                  <th style="padding: 10px; border-bottom: 2px solid #ddd; background-color: #cccccc !important; color: #000;">🏁+🎁</th>
                  <th style="padding: 10px; border-bottom: 2px solid #ddd; background-color: #cccccc !important; color: #000;">🚪+🏁</th>
                  <th style="padding: 10px; border-bottom: 2px solid #ddd; background-color: #cccccc !important; color: #000;">🚪+🏁+📏</th>
                  <th style="padding: 10px; border-bottom: 2px solid #ddd; background-color: #cccccc !important; color: #000;">🚪+⚠️+🏁</th>
              </tr>
          </thead>
          <tbody>
              <tr>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;">Instruct-model</td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;">0.66 &plusmn; 0.08</td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;">0.03 &plusmn; 0.03</td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;">0.22 &plusmn; 0.07</td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd;">0.19 &plusmn; 0.07</td>
              </tr>
              <tr>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd; font-weight: bold;">Verlog (Ours)</td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd; font-weight: bold;">1.00 &plusmn; 0.00</td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd; font-weight: bold;">1.0</td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd; font-weight: bold;">0.89 &plusmn; 0.11</td>
                  <td style="padding: 10px; border-bottom: 1px solid #ddd; font-weight: bold;">0.69</td>
              </tr>
          </tbody>
      </table>
  </div>
  

  > BabaIsAI's experiments are done with Qwen2.5-3B-Instruct model, using PPO algorithm, trained on 4xA40 GPUs with 48Gb memory for ~24 hours, corresponding to 300 PPO updates.


* BabyAI Results (win rate)
  <div style="overflow-x: auto;">
    <table style="width: 100%; border-collapse: collapse; text-align: center; font-family: sans-serif;">
        <thead>
            <tr>
                <th style="padding: 10px; border-bottom: 2px solid #ddd; background-color: #cccccc !important; color: #000;">Model</th>
                <th style="padding: 10px; border-bottom: 2px solid #ddd; background-color: #cccccc !important; color: #000;">goto</th>
                <th style="padding: 10px; border-bottom: 2px solid #ddd; background-color: #cccccc !important; color: #000;">pickup</th>
                <th style="padding: 10px; border-bottom: 2px solid #ddd; background-color: #cccccc !important; color: #000;">pick_up_seq_go_to</th>
                <th style="padding: 10px; border-bottom: 2px solid #ddd; background-color: #cccccc !important; color: #000;">open</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">Instruct-model</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">0.88 &plusmn; 0.06</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">0.41 &plusmn; 0.09</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">0.22 &plusmn; 0.07</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">0.09 &plusmn; 0.05</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd; font-weight: bold;">Verlog (Ours)</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd; font-weight: bold;">1.00 &plusmn; 0.00</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd; font-weight: bold;">1.00 &plusmn; 0.00</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd; font-weight: bold;">0.65 &plusmn; 0.16</td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd; font-weight: bold;">0.94 &plusmn; 0.07</td>
            </tr>
        </tbody>
    </table>
  </div>

    > BabyAI's experiments are done with Qwen2.5-3B-Instruct model, using PPO algorithm, trained on 4xA40 GPUs with 48Gb memory for ~24 hours, corresponding to 300 PPO updates.

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

We provide training examples and fine-tuned hyper-parameters list in `Verlog/examples/Verlog`. 


