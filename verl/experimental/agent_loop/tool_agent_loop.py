# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional, List
from uuid import uuid4
import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

    @rollout_trace_op
    async def run(self, env, data_buffer, env_idx: int, sampling_params: dict[str, Any], is_val: bool, **kwargs) -> List[AgentLoopOutput]:
        
        outputs = []
        last_prompt_ids = None
        last_info = None
        
        total_count = 0
        
        while True:
            # Pop data point from buffer
            data_point = await data_buffer.pop.remote()
            if data_point is None:
                print(f"Total count: {total_count} len(outputs): {len(outputs)}")
                break
            total_count += 1
            # Reset environment with data point as sample
            messages, info = env.reset(sample=data_point)
            
            metrics = {}
            request_id = uuid4().hex
            
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            
            num_turns = 0
            reward = 0.0
            
            # Inner loop for environment steps
            while True:
                
                with simple_timer("generate_sequences", metrics):
                    output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        image_data=None,
                    )
                
                # truncate response_ids to response_length
                response_ids = output.token_ids[: self.response_length]
                response_mask = [1] * len(response_ids)
                
                assert len(prompt_ids) <= self.prompt_length
                assert len(response_ids) <= self.response_length
                
                if output.log_probs:
                    response_logprobs = output.log_probs[: self.response_length]
                else:
                    response_logprobs = None
                
                actions = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
                )
                
                last_prompt_ids = copy.deepcopy(prompt_ids)
                last_phase = info.get("phase")
                
                assert last_phase in ["dreaming", "answering"], f"last_phase: {last_phase}"
                assert (num_turns == 0 and last_phase == "dreaming") or (num_turns == 1 and last_phase == "answering"), f"num_turns: {num_turns}, last_phase: {last_phase}"
                
                messages, reward, terminated, truncated, info = env.step(actions)
                done = np.logical_or(terminated, truncated)
                
                if last_phase == "dreaming":
                    turn_data = AgentLoopOutput(
                        prompt_ids=prompt_ids,
                        response_ids=response_ids,
                        response_mask=response_mask,
                        response_logprobs=response_logprobs,
                        metrics=metrics,
                        rewards=reward,
                        done=done,
                        num_turns=num_turns,
                        env_idx=env_idx,
                        info=info.get("metrics", info),
                    )
                    outputs.append(turn_data)
                else:
                    outputs[-1].rewards = reward
                
                # assert (last_phase == "answering" and done) or (last_phase == "dreaming" and not done)
                
                if done:
                    break
                
                num_turns += 1
                
                prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        messages,
                        tools=self.tool_schemas,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
            
        # Add final turn data for this episode
        turn_data = AgentLoopOutput(
            prompt_ids=last_prompt_ids,
            response_ids=[outputs[-1].response_ids[0]] if outputs else [151645],
            response_mask=[1],
            metrics=dict(),
            rewards=reward if is_val else 0.0,
            done=True,
            num_turns=1,
            env_idx=env_idx,
            info=dict(), # last_info["metrics"],
        )
        outputs.append(turn_data)
        
        return outputs

