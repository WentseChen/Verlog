
from verl.envs.environments import make_env
from verl.envs.captioners import make_captioner

class BalrogEnv:
    def __init__(self, env_name, config, env, captioner):
        self.env_name = env_name
        self.config = config
        self.env = env
        self.captioner = captioner
        self.image = None
        self.last_msg = None
        self.last_info = None
        
    def get_last_obs(self):
        return self.last_msg, self.last_info
    
    def step(self, action):
        full_action, executed_action, is_valid, metrics = self.env.extract_action(action)
        env_obs, reward, terminated, truncated, info = self.env.step(executed_action, is_valid)
        self.image = env_obs.get("image", None)
        instructions = env_obs["mission"] if self.env_name == "babyai" else None
        inst_prompt = self.env.get_instruction_prompt(instructions=instructions, info=info)
        self.captioner.prompt_builder.update_instruction_prompt(inst_prompt)
        self.captioner.update_action(full_action, executed_action)
        info["metrics"] = {**info.get("metrics", {}), **metrics}
        obs = self.captioner.get_obs(env_obs)
        # Auto-reset if episode ends
        if terminated or truncated:
            obs, last_info = self._reset_internal()
        self.last_msg = obs
        self.last_info = info if not (terminated or truncated) else last_info
        
        return obs, reward, terminated, truncated, info
    
    def reset(self):
        obs, info = self._reset_internal()
        self.last_msg = obs
        self.last_info = info
        return obs, info
    
    def _reset_internal(self):
        self.captioner.reset()
        env_obs, info = self.env.reset()
        self.image = env_obs.get("image", None)
        instructions = env_obs["mission"] if self.env_name == "babyai" else None
        inst_prompt = self.env.get_instruction_prompt(instructions=instructions)
        self.captioner.prompt_builder.update_instruction_prompt(inst_prompt)
        obs = self.captioner.get_obs(env_obs)
        return obs, info
    
    def render(self):
        return self.image
    
    def close(self):
        self.env.close()

def get_balrog_env(config):
    env = make_env(config.envs.env_name, config.envs.task, config)
    captioner = make_captioner(config)
    return BalrogEnv(config.envs.env_name, config, env, captioner)