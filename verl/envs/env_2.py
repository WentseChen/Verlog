from __future__ import annotations

import sys
from pathlib import Path

UNSCRIPTED_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "unscripted_llms"
if UNSCRIPTED_ROOT.exists() and str(UNSCRIPTED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNSCRIPTED_ROOT))

from src.wrappers.pettingzoo_wrapper import AdmissionsPettingZooEnv


class Env:
    def __init__(self, env_name, config, captioner=None):
        self.env_name = env_name
        self.config = config
        self.env = self._build_env()
        self.agent_id = None
        self.history = HistoryManager(self.env)
        self.last_msg = None
        self.last_info = None

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
        return AdmissionsPettingZooEnv(game_config=game_config)

    def _build_messages(self, obs_text: str):
        return self.history.add_user_message(self.agent_id, obs_text)

    def _normalize_actions(self, action_text: str, agent_id: str | None):
        if agent_id is None:
            agent_id = self.env.possible_agents[0] if self.env.possible_agents else None

        actions = {agent: "" for agent in self.env.possible_agents}
        if agent_id:
            actions[agent_id] = action_text
        return actions

    def get_last_obs(self):
        return self.last_msg, self.last_info

    def step(self, action):
        # action is the agent's response text; env returns prompts/messages
        if self.agent_id is None and self.env.possible_agents:
            self.agent_id = self.env.possible_agents[0]

        action_text = action if action is not None else ""
        actions = self._normalize_actions(action_text, self.agent_id)

        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        obs_text = observations.get(self.agent_id, "") if observations else ""
        reward = rewards.get(self.agent_id, 0.0) if rewards else 0.0
        terminated = terminations.get(self.agent_id, False) if terminations else False
        truncated = truncations.get(self.agent_id, False) if truncations else False
        info = dict(infos.get(self.agent_id, {})) if infos else {}
        info["agent_id"] = self.agent_id
        info["raw_infos"] = infos

        if action_text:
            self.history.add_assistant_message(self.agent_id, action_text)
        messages = self._build_messages(obs_text)
        self.last_msg = messages
        self.last_info = info
        return messages, reward, terminated, truncated, info

    def reset(self):
        observations, infos = self.env.reset()
        self.agent_id = self.env.possible_agents[0] if self.env.possible_agents else None
        obs_text = observations.get(self.agent_id, "") if observations else ""
        info = dict(infos.get(self.agent_id, {})) if infos else {}
        info["agent_id"] = self.agent_id
        info["raw_infos"] = infos
        messages = self._build_messages(obs_text)
        self.last_msg = messages
        self.last_info = info
        return messages, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


class HistoryManager:
    def __init__(self, env):
        self.env = env
        self._messages_by_agent: dict[str, list[dict]] = {}

    def _ensure_history(self, agent_id: str | None) -> list[dict]:
        if agent_id is None:
            return []
        if agent_id not in self._messages_by_agent:
            self._messages_by_agent[agent_id] = []
        return self._messages_by_agent[agent_id]

    def _get_system_prompt(self, agent_id: str | None) -> str | None:
        if agent_id is None:
            return None
        if not hasattr(self.env, "prompt_builder") or not hasattr(self.env, "game"):
            return None
        try:
            return self.env.prompt_builder.build_system_prompt(
                professor_id=agent_id,
                game=self.env.game,
                personality="",
                personality_prompts={},
                runner_mode="sync",
            )
        except Exception:
            return None

    def add_assistant_message(self, agent_id: str | None, content: str):
        if agent_id is None or not content:
            return
        history = self._ensure_history(agent_id)
        history.append({"role": "assistant", "content": content})

    def add_user_message(self, agent_id: str | None, content: str) -> list[dict]:
        history = self._ensure_history(agent_id)
        if not history:
            system_prompt = self._get_system_prompt(agent_id)
            if system_prompt:
                history.append({"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": content})
        return list(history)
