import re

import gymnasium as gym

ALFWORLD_ACTIONS = [
    "pass",
    "goto",
    "pick",
    "put",
    "open",
    "close",
    "toggle",
    "heat",
    "clean",
    "cool",
    "slice",
    "inventory",
    "examine",
    "look",
]


class ALFWorldLLMAgentsWrapper(gym.Wrapper):
    def __init__(self, env, vlm=False, **kwargs):
        super().__init__(env)
        self.env = env
        self.format_penalty = kwargs.get("format_penalty", 0.0)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, is_valid=True):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not is_valid:
            reward = -self.format_penalty
        return obs, reward * 1.0, terminated, truncated, info

    def extract_action(self, action):
        full_action = str(action)
        parsed_action = full_action.strip()
        is_valid = 1.0

        start_tag = parsed_action.lower().find("<action>")
        end_tag = parsed_action.lower().find("</action>")
        if start_tag != -1 and end_tag != -1 and end_tag > start_tag:
            parsed_action = parsed_action[start_tag + len("<action>") : end_tag].strip()
        else:
            is_valid = 0.0

        # ALFWorld commands are lowercase.
        parsed_action = parsed_action.lower()

        think_start = full_action.find("<think>")
        think_end = full_action.find("</think>")
        if think_start == -1 or think_end == -1:
            is_valid = 0.0

        # Keep consistent with prior implementation constraints.
        if re.search(r"[\u4e00-\u9fff]", full_action):
            is_valid = 0.0

        valid_action = parsed_action if parsed_action in self.language_action_space else self.default_action
        if parsed_action not in self.language_action_space:
            is_valid = 0.0

        metrics = {"behavior/valid_action_ratio": is_valid}
        return full_action, valid_action, bool(is_valid), metrics

