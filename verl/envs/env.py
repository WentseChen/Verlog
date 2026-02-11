from __future__ import annotations

import re
import sys
from pathlib import Path

UNSCRIPTED_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "unscripted_llms"
if UNSCRIPTED_ROOT.exists() and str(UNSCRIPTED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNSCRIPTED_ROOT))

from src.wrappers.gym_llm_wrapper import AdmissionsLLMGymEnv


class Env:
    def __init__(self, env_name, config, env=None, captioner=None):
        self.env_name  = env_name
        self.config = config
        self.env = env if env is not None else self._build_env()
        self.last_msg = None
        self.last_info = None
        self.last_msg_by_agent = {}
        self.last_info_by_agent = {}

    def _build_env(self):
        game_config = None
        if hasattr(self.config, "envs"):
            game_config = getattr(self.config.envs, "game_config", None)
        if not game_config:
            game_config = {
                "professor_ids": ["prof_1", "prof_2", "prof_3"],
                "students_per_batch": 10,
                "num_rounds": 1,
                "feature_dim": 5,
                "vote_threshold": 0.5,
                "seed": None,
            }
        return AdmissionsLLMGymEnv(game_config=game_config)

    def _build_messages(self, obs_text: str, agent_id: str | None):
        system_prompt = None
        if hasattr(self.env, "build_system_prompt"):
            try:
                system_prompt = self.env.build_system_prompt(agent_id)
            except Exception:
                system_prompt = None
        if system_prompt is None:
            try:
                prof_config = self.env._get_professor_config(agent_id)
                system_prompt = self.env.prompt_builder.build_system_prompt(
                    professor_id=agent_id,
                    game=self.env.game,
                    personality=prof_config.personality,
                    personality_prompts=self.env.personality_prompts,
                    runner_mode="async",
                    token_budget=self.env.total_token_budget,
                    shared_token_budget=True,
                )
            except Exception:
                system_prompt = None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": obs_text})
        return messages

    def _extract_obs_text(self, observations: dict, agent_id: str | None) -> str:
        if agent_id is None:
            return ""
        obs_item = observations.get(agent_id, "")
        if isinstance(obs_item, str):
            return obs_item
        if isinstance(obs_item, dict):
            prompt_text = obs_item.get("prompt", "")
            return prompt_text if isinstance(prompt_text, str) else str(prompt_text)
        return str(obs_item)

    def _extract_agent_id_from_action(self, action) -> str | None:
        if action is None:
            return None
        if isinstance(action, list):
            for msg in action:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "system":
                    agent_id = self._extract_agent_id_from_text(msg.get("content", ""))
                    if agent_id:
                        return agent_id
            return self._extract_agent_id_from_text(" ".join(str(item) for item in action))
        if isinstance(action, dict):
            if "messages" in action and isinstance(action["messages"], list):
                return self._extract_agent_id_from_action(action["messages"])
            if "system" in action and isinstance(action["system"], str):
                return self._extract_agent_id_from_text(action["system"])
            if len(action) == 1:
                only_key = next(iter(action.keys()))
                if isinstance(only_key, str):
                    if getattr(self.env, "possible_agents", None) and only_key in self.env.possible_agents:
                        return only_key
                    return only_key
        if isinstance(action, str):
            return self._extract_agent_id_from_text(action)
        return None

    def _extract_agent_id_from_text(self, text: str) -> str | None:
        match = re.search(r"You are Professor\s+([^,\s]+)", text, re.IGNORECASE)
        if not match:
            return None
        return match.group(1)

    def get_last_obs(self, agent_id: str | None = None):
        if agent_id is not None:
            return self.last_msg_by_agent.get(agent_id), self.last_info_by_agent.get(agent_id)
        return self.last_msg, self.last_info

    def step(self, action):
        observations, reward, terminated, truncated, infos = self.env.step(action)
        info = dict(next(iter(infos.values()))) if infos else {}
        agent_id = self._extract_agent_id_from_action(action)
        if agent_id is None and getattr(self.env, "possible_agents", None):
            agent_id = self.env.possible_agents[0]
        if isinstance(reward, dict) and agent_id is not None:
            reward = reward.get(agent_id, 0.0)
        if isinstance(terminated, dict) and agent_id is not None:
            terminated = terminated.get(agent_id, False)
        if isinstance(truncated, dict) and agent_id is not None:
            truncated = truncated.get(agent_id, False)
        if isinstance(infos, dict) and agent_id is not None:
            info = dict(infos.get(agent_id, info))
        obs_text = self._extract_obs_text(observations, agent_id)
        info["agent_id"] = agent_id
        info["raw_infos"] = infos
        messages = self._build_messages(obs_text, agent_id)
        self.last_msg = messages
        self.last_info = info
        if agent_id is not None:
            self.last_msg_by_agent[agent_id] = messages
            self.last_info_by_agent[agent_id] = info
        return messages, reward, terminated, truncated, info

    def reset(self, agent_id: str | None = None):
        observations, infos = self.env.reset()
        if agent_id is None:
            agent_id = self.env.possible_agents[0] if self.env.possible_agents else None
        obs_text = self._extract_obs_text(observations, agent_id)
        info = dict(infos.get(agent_id, {}))
        info["agent_id"] = agent_id
        info["raw_infos"] = infos
        messages = self._build_messages(obs_text, agent_id)
        self.last_msg = messages
        self.last_info = info
        if agent_id is not None:
            self.last_msg_by_agent[agent_id] = messages
            self.last_info_by_agent[agent_id] = info
        return messages, info

    def render(self):
        return None

    def close(self):
        self.env.close()
