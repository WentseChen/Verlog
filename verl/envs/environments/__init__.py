# Here we should have an environment manager function that can be used to instantiate
# environments with the correct wrappers.
from gym import spaces

from verl.envs.environments.env_wrapper import EnvWrapper


def make_env(env_name, task, config, render_mode=None, tokenizer=None):
    """Create an environment instance with the appropriate wrapper based on the environment name.

    Args:
        env_name (str): The name of the environment to create.
        task (str): The specific task within the environment.
        config (dict): Configuration settings for the environment.
        render_mode (str, optional): Rendering mode for the environment. Defaults to None.

    Returns:
        EnvWrapper: A wrapped environment instance (or raw env for dummy_openai).

    Raises:
        ValueError: If the environment name is not recognized.
    """
    if env_name == "nle":
        from verl.envs.environments.nle.nle_env import make_nle_env

        base_env = make_nle_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "minihack":
        from verl.envs.environments.minihack.minihack_env import make_minihack_env

        base_env = make_minihack_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "babyai":
        from verl.envs.environments.babyai_text.babyai_env import make_babyai_env
        base_env = make_babyai_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "crafter":
        from verl.envs.environments.crafter.crafter_env import make_crafter_env

        base_env = make_crafter_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "textworld":
        from verl.envs.environments.textworld.textworld_env import make_textworld_env

        base_env = make_textworld_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "babaisai":
        from verl.envs.environments.babaisai.babaisai_env import make_babaisai_env
        base_env = make_babaisai_env(env_name, task, config, render_mode=render_mode)
    elif env_name == "dummy_openai":
        from verl.envs.environments.dummy_openai_env import DummyOpenAIEnv

        dummy_prompt = getattr(config.envs, "dummy_prompt", "what's 1+1?")
        return DummyOpenAIEnv(prompt=dummy_prompt)
    elif env_name == "dummy_openai_multi":
        from verl.envs.environments.dummy_openai_multi_env import DummyOpenAIMultiEnv

        dummy_prompt = getattr(config.envs, "dummy_prompt", "what's 1+1?")
        num_agents = getattr(config.envs, "dummy_num_agents", 2)
        return DummyOpenAIMultiEnv(prompt=dummy_prompt, num_agents=num_agents)
    elif env_name == "async_ticker_admissions":
        from omegaconf import OmegaConf
        from verl.envs.async_ticker_env import AsyncTickerAdmissionsEnv, AsyncTickerEnvWrapper

        env_config = getattr(config.envs, "env_config", config.envs)
        env_config_dict = OmegaConf.to_container(env_config, resolve=True)
        # Inject prompt_length so the env can enforce a token-based trim budget on
        # the user observation, keeping the critical header (student table, vote tally)
        # always visible.
        try:
            env_config_dict["prompt_length"] = config.actor_rollout_ref.rollout.prompt_length
        except Exception:
            pass
        return AsyncTickerEnvWrapper(AsyncTickerAdmissionsEnv(env_config_dict, tokenizer=tokenizer))
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    return EnvWrapper(base_env, env_name, task)


class Strings(spaces.Space):
    """A custom Gym space for managing discrete string-based actions."""

    def __init__(self, values, seed=None):
        super().__init__((len(values),), str, seed)
        self._dict = {value: i for i, value in enumerate(values)}
        self._values = values

    def sample(self):
        return self.np_random.choice(self._values)

    def map(self, action):
        return self._dict[action]

    def contains(self, value):
        return value in self._dict

    def __iter__(self):
        return self._values.__iter__()
