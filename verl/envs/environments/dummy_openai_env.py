from __future__ import annotations


class DummyOpenAIEnv:
    """Minimal env that emits a fixed prompt and ends after one step."""

    def __init__(self, prompt: str = "what's 1+1?"):
        self.prompt = prompt
        self.possible_agents = ["agent_0"]
        self.max_steps = 1
        self._step = 0

    def reset(self, **kwargs):
        self._step = 0
        observations = {self.possible_agents[0]: {"prompt": self.prompt}}
        infos = {self.possible_agents[0]: {}}
        return observations, infos

    def step(self, action):
        self._step += 1
        observations = {self.possible_agents[0]: {"prompt": ""}}
        infos = {self.possible_agents[0]: {}}
        reward = 0.0
        terminated = self._step >= self.max_steps
        truncated = False
        return observations, reward, terminated, truncated, infos

    def render(self):
        return None

    def close(self):
        return None
