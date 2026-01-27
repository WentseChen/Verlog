from __future__ import annotations


class DummyOpenAIMultiEnv:
    """Minimal multi-agent env that emits fixed prompts and ends after one step."""

    def __init__(self, prompt: str = "what's 1+1?", num_agents: int = 2):
        self.prompt = prompt
        self.num_agents = max(1, int(num_agents))
        self.possible_agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.max_steps = 1
        self._step = 0

    def reset(self, **kwargs):
        self._step = 0
        observations = {
            agent_id: {"prompt": self.prompt} for agent_id in self.possible_agents
        }
        infos = {agent_id: {} for agent_id in self.possible_agents}
        return observations, infos

    def step(self, action):
        self._step += 1
        observations = {
            agent_id: {"prompt": ""} for agent_id in self.possible_agents
        }
        infos = {agent_id: {} for agent_id in self.possible_agents}
        reward = {agent_id: 0.0 for agent_id in self.possible_agents}
        terminated = {agent_id: self._step >= self.max_steps for agent_id in self.possible_agents}
        truncated = {agent_id: False for agent_id in self.possible_agents}
        return observations, reward, terminated, truncated, infos

    def render(self):
        return None

    def close(self):
        return None
