<h1 style="text-align: center;">Verlog</h1>

[![GitHub Repo stars](https://img.shields.io/github/stars/WentseChen/verl)](https://github.com/WentseChen/verl/stargazers)
![GitHub forks](https://img.shields.io/github/forks/WentseChen/verl)
<!-- [![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project) -->
<!-- <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a> -->
<!-- <a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a> -->
<!-- ![GitHub contributors](https://img.shields.io/github/contributors/WentseChen/verl) -->
<!-- [![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/) -->
<!-- <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a> -->

Verlog is a well-tuned multi-turn RL framework built for long-horizon LLM agentic tasks. It extends RLHF framework [VeRL](https://github.com/volcengine/verl) to the multi-turn setting and is evaluated on the long horizon LLM agents benchmark [BALROG](https://github.com/balrog-ai/BALROG). Its workflow follows the core design principles of [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail), while introducing tailored modifications for efficient multi-turn learning.

## Key features:  

⏳ Fixed-Turn Batching: For every training batch, a fixed number of turns is collected. If the episode has not terminated, we use the value function instead of relying on final rewards. 

🧠 Turn-Level Abstraction: Each turn is treated as an independent data point—no need to pack the entire history into the context window. Customize your memory mechanism as needed. 

🚀 Optimized for Long-Horizon Agentic Tasks: Verlog incorporates techniques like Dual Discounting GAE and Critic Pre-training, along with carefully tuned hyperparameters, to ensure strong performance on challenging long-horizon multi-turn benchmarks such as BALROG.

## Main Results

All the experiments are done with Qwen2.5-3B-Instruct model, PPO on 4xA40 GPUs with 48Gb memory for 24 hours.



## Technical Report

We will explain the design choice and introduce the implementation details of the proposed framework in the following sections:

### Model

* **Instruct Model:**
    For simplicity, we start with the Instruct model, **Qwen-2.5-3B-Instruct**. We chose this over the base model because it allows us to leverage [BALROG](https://github.com/balrog-ai/BALROG) for debugging and to use the benchmark's prompts with minimal modifications.

* **Memory Mechanism:**
    Rather than placing the entire trajectory into the context window, we include only the latest $n+1$ turns. Each turn, i.e., data = $(\text{history}_t, s_t, \text{think}_t, a_t)$, with $\text{history}_t = \{s_{t-n}, \text{think}_{t-n}, a_{t-n}, ..., s_{t-1}, \text{think}_{t-1}, a_{t-1}\}$, is treated as an individual training data point. As a result, each training batch consists of `batch_size` individual turns, not `batch_size` full trajectories. We detail our memory design choices below:

    * **(1) Choice of $n$:**
    For the 3B Qwen model, performance peaks at $n = 1$ or $2$ and degrades as $n$ increases to $4$ or $8$. We hypothesize that this decline is due to the 3B model’s limited capacity to handle long contexts—for example, $n = 8$ yields a prompt of approximately 4.6k tokens. Whether this trend holds for larger models is an open question. Notably, the tasks we evaluate can be framed as Markov Decision Processes (MDPs). In more complex or partially observable tasks, a larger $n$ may help.

    * **(2) Including reasoning paths in context:**
    We tested a variant that includes only the final action in history:
    data = $(\text{history}_t, s_t, \text{think}_t, a_t)$, with
    $\text{history}_t = \{s_{t-n}, a_{t-n}, ..., s_{t-1}, a_{t-1}\}$.
    This version failed to learn an effective policy.

    Qualitative results show that increasing $n$ leads to: (1) longer responses, as LLMs often spend additional tokens rephrasing or reiterating previous plans; (2) less diverse reasoning paths, with the model tending to mimic reasoning patterns from earlier turns; and (3) more hallucinations, where the LLM struggles to distinguish between internally generated thoughts and actual events in the environment. Future work is needed to investigate how memory design influences these behaviors in LLM agents.

* **Prompt Template:**
    Belows is the prompt template used for BabyAI. The prompts are adapted from [BALROG](https://github.com/balrog-ai/BALROG).
    ```
    [SYSTEM] You are an agent playing a simple navigation game. 
    Your goal is to {MISSION}. 
    The following are the possible actions you can take in the game, 
    followed by a short description of each action: {AVAILABLE ACTIONS}. 
    In a moment I will present you an observation. Tips: {TIPS}.
    PLAY!
    ```
    ```
    [USER] {OBSERVATION t-1}
    ```
    ```
    [ASSISTANT]
    THINK: {THINK t-1}
    ACTION: {ACTION t-1}
    ```
    ```
    [USER] {OBSERVATION t}
    What will you do next? Please respond in the following format:
    THINK: step-by-step reasoning
    ACTION: One valid action from the allowed set.
    ```
    We recommend always examining the model’s zero-shot outputs before training. Specifically, evaluate: (1) Whether reasoning paths are diverse, (2) whether the model reasons sufficiently before selecting an action, (3) the ratio of valid actions, and (4) the types of failure cases. These checks ensure the model understands the environment from the prompt. If not, revise the prompt before fine-tuning.

### Environment

* **Valid Action:**
    Improving the valid action ratio via prompt engineering is the simplest way to boost the performance. In our setup, we ensure the model produces valid actions over 95% of the time.

    We find that truncating the entire trajectory upon encountering an invalid action reduces performance. Replacing the invalid action with a default one works better. With a high valid-action ratio, applying a format penalty has little impact.

* **Reward:**
    Rewards are rule-based and provided by the environment. In BabyAI and BabaIsAI, the reward (ranging from 0 to 1) is only available at the end of the trajectory. Longer trajectories generally get lower rewards.

    Here, We adopt a **binary trajectory-level reward** scheme: 1 for success trajectory, 0 for failure trajectory. Combined with dual-discount GAE, this ensures earlier steps in suboptimal trajectories get lower credit than those in optimal ones.

* **Batch Environment:**
    Our framework supports asynchronous rollouts and works with any environment using the OpenAI Gym interface. Each training batch size is: `n_env` × `e_len`, where:
    * `n_env` = number of parallel environments
    * `e_len` = episode length per rollout

    Note: `e_len` can be smaller than the environment's trajectory length. For example, we set `e_len = 8` and max trajectory length = 128 in BabyAI. For early truncated trajectories, we leverage the value function to guide the training process. We find that using `e_len = 8`, `n_env = 32` performs better than `e_len = 16`, `n_env = 16`.

    

### Algorithm

* **Dual Discounting GAE:**
    We decouple token-level discounting $(\gamma_{\text{token}}, \lambda_{\text{token}})$ and step-level $(\gamma_{\text{step}}, \lambda_{\text{step}})$. We set:

    * $\gamma_{\text{step}} = 0.99$, $\lambda_{\text{step}} = 0.95$
    * $\gamma_{\text{token}} = 1.0$, $\lambda_{\text{token}} = 1.0$

    The GAE is computed recursively:

    $$
    \hat{A}_t = \gamma\lambda \hat{A}_{t+1} + \delta_t^V
    $$

    where:

    * $\gamma\lambda = \gamma_{\text{step}} \lambda_{\text{step}}$ if tokens are from different turns
    * $\gamma\lambda = \gamma_{\text{token}} \lambda_{\text{token}}$ otherwise
    * $\delta_t^V = -V(s_t) + r_t + \gamma V(s_{t+1})$

    The recursion starts from the last token of the final turn and proceeds backward. Once all tokens in the final turn are processed, we move to the last token of the second-to-last turn, and continue this process recursively. 
    If a trajectory is truncated at step $T$, we store the next state $s_{T+1}$ but do not sample $a_{T+1}$. Instead, we use the final token of $s_{T+1}$ to estimate $V(s_{T+1})$, used as the bootstrap value in GAE.

* **Value Function Estimation:**
    * When both $\gamma$ and $\lambda$ are set to 1.0, the value function serves purely as a baseline in PPO’s advantage estimation. Specifically, the advantage for the $t$-th token in the last turn is defined as $A_{-1,t} = r - V_{-1,t}$, where $r$ is the trajectory reward and $V_{-1,t}$ is the value estimate for the $t$-th token in the last turn.

    * When $\lambda$ are less than 1.0, the value function contributes to the GAE objective beyond serving as a simple baseline. For instance, in our setting with $\gamma_{\text{step}} = 0.99$, $\lambda_{\text{step}} = 0.95$, and $\gamma_{\text{token}} = 1.0$, $\lambda_{\text{token}} = 1.0$, where the reward $r$ is non-zero only at the end of the trajectory, the advantage for the $t$-th token in the second-to-last turn is given by: $A_{-2,t} = \gamma_{\text{step}}[\lambda_{\text{step}} r + (1-\lambda_{\text{step}}) V_{-1,0}] - V_{-2,t}$. In fact, in our setting, the value function of the first token in each turn is used to bootstrap the GAE objective for the preceding turn.

    * Since the first token of each turn carries more semantic significance than the subsequent tokens, we assign it a higher weight when training the critic network.

* **Critic Warmup:**
    In our setting, we warm up the critic before fine-tuning, as it is used both for bootstrapping truncated trajectories and for computing GAE. That is, we freeze the actor and update only the critic at the beginning of training. Specifically, We collect `w_epoch × batch_size` turns of data at the beginning. For each warmup iteration, we compute the GAE objective with current critic, sample one tenth of the collected data, train the critic, and repeat this process for `w_iter` iterations. We select `w_epoch = 40` and `w_iter = 5` in our experiments, and make sure that the critic loss converges to a small value before fine-tuning the actor.
    
* **KL-Divergence in Reward:**
    Adding a KL-divergence term $KL(\pi||\pi_0)$ in reward stabilizes training. Without it, the policy quickly drifts from \$\pi\_0\$ and converges to poor solutions. KL rewards encourage local exploration around \$\pi\_0\$ before divergence.

    For entorpy term in the PPO loss, we experimented with higher entropy coefficients (e.g., 3e-3, 1e-2), which caused instability. We set both entropy and KL reward coefficients to **1e-3**. 

## Conclusion

In summary, when training an LLM agent using multi-turn RL, the following aspects are critical (in order of importance):

1. Prompt design
2. Value loss optimization
3. Exploration



