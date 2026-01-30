from __future__ import annotations

from typing import Any, List

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register


@register("multi_tool_agent_loop")
class MultiAgentToolAgentLoop(ToolAgentLoop):
    """Multi-agent wrapper over ToolAgentLoop that selects one agent per sample."""

    async def run(
        self,
        env,
        counter,
        env_idx: int,
        sampling_params: dict[str, Any],
        is_val: bool,
        **kwargs,
    ) -> List[AgentLoopOutput]:
        agent_id = kwargs.get("agent_id")
        if agent_id is None:
            possible_agents = (
                getattr(env, "possible_agents", None)
                or getattr(getattr(env, "env", None), "possible_agents", None)
            )
            if possible_agents:
                agent_id = possible_agents[env_idx % len(possible_agents)]

        return await super().run(
            env,
            counter,
            env_idx,
            sampling_params,
            is_val,
            agent_id=agent_id,
            **{k: v for k, v in kwargs.items() if k != "agent_id"},
        )
