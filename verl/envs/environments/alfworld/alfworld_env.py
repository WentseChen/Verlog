import os
from typing import Optional

import gymnasium as gym
import yaml

from verl.envs.environments.alfworld.llm_agents_wrapper import ALFWorldLLMAgentsWrapper


def _load_config_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ALFWorld config not found: {path}")
    with open(path, encoding="utf-8") as reader:
        return yaml.safe_load(reader)


class ALFWorldSingleEnv(gym.Env):
    def __init__(self, alf_config_path: str, eval_dataset: str = "eval_in_distribution"):
        super().__init__()
        try:
            from alfworld.agents.environment import get_environment
        except ImportError as exc:
            raise ImportError(
                "ALFWorld support requires the `alfworld` package. "
                "Please install it in this environment."
            ) from exc

        self.config = _load_config_file(alf_config_path)
        env_type = self.config["env"]["type"]
        base_env = get_environment(env_type)(self.config, train_eval=eval_dataset)
        self.env = base_env.init_env(batch_size=1)

        self.max_steps = int(self.config.get("dagger", {}).get("training", {}).get("max_nb_steps_per_episode", 50))
        self._admissible_commands = ["look"]
        self._progression = 0.0
        self.action_space = gym.spaces.Space()
        self.observation_space = gym.spaces.Space()

    @property
    def default_action(self):
        return "look"

    @property
    def language_action_space(self):
        return self._admissible_commands

    def get_text_action(self, action):
        return action

    def _process_text_obs(self, text_obs):
        return {"text": {"long_term_context": text_obs, "short_term_context": ""}, "image": None}

    def reset(self, **kwargs):
        obs, infos = self.env.reset()
        info = {k: v[0] for k, v in infos.items()}
        self._admissible_commands = info.get("admissible_commands", ["look"])
        self._progression = float(info.get("goal_condition_success_rate", 0.0))
        return self._process_text_obs(obs[0]), info

    def step(self, action):
        obs, scores, dones, infos = self.env.step([action])
        info = {k: v[0] for k, v in infos.items()}
        self._admissible_commands = info.get("admissible_commands", ["look"])

        won = float(info.get("won", 0.0))
        progression = float(info.get("goal_condition_success_rate", 0.0))
        self._progression = progression
        reward = 10.0 * won + progression

        terminated = bool(dones[0])
        truncated = False
        return self._process_text_obs(obs[0]), reward, terminated, truncated, info

    def get_stats(self):
        return {"progression": self._progression}


def make_alfworld_env(env_name, task, config, render_mode: Optional[str] = None):
    env_type = config.envs.alfworld_kwargs.get("env_type", "AlfredTWEnv")
    config_name = "config_thor.yaml" if env_type == "AlfredThorEnv" else "config_tw.yaml"
    config_path = os.path.join(os.path.dirname(__file__), "configs", config_name)

    eval_dataset = config.envs.alfworld_kwargs.get("eval_dataset", "eval_in_distribution")
    env = ALFWorldSingleEnv(config_path, eval_dataset=eval_dataset)
    env = ALFWorldLLMAgentsWrapper(env, **config.envs)
    return env

