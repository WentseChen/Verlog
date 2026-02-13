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

    def _calculate_z_index(self, answering_prompt_ids, z_content_tokens):
        """
        Calculate the token index range where z_content_tokens appear in answering_prompt_ids.
        """
        if not z_content_tokens:
            return None
        
        z_len = len(z_content_tokens)
        prompt_len = len(answering_prompt_ids)
        
        # Find where z_tokens appear in answering_prompt_ids
        for i in range(prompt_len - z_len + 1):
            if answering_prompt_ids[i:i+z_len] == z_content_tokens:
                return (i, i + z_len)
        
        return None

    def _calculate_z_index2(self, response_ids, z_content_tokens):
        """
        Calculate which tokens in response_ids correspond to z_content_tokens.
        """
        if not z_content_tokens:
            return [False] * len(response_ids)
        
        z_len = len(z_content_tokens)
        response_len = len(response_ids)
        
        # Create boolean mask
        z_index2 = [False] * response_len
        
        # Find where z_tokens appear in response_ids
        for i in range(response_len - z_len + 1):
            if response_ids[i:i+z_len] == z_content_tokens:
                # Mark these positions as True
                for j in range(i, i+z_len):
                    z_index2[j] = True
                break
        
        # pad to self.response_length
        z_index2 = z_index2 + [False] * (self.response_length - len(z_index2))
        
        return z_index2

    @rollout_trace_op
    async def run(self, env, data_buffer, env_idx: int, sampling_params: dict[str, Any], is_val: bool, **kwargs) -> List[AgentLoopOutput]:
        
        outputs = []
        last_prompt_ids = None
        last_info = None
        
        while True:
            # Pop data point from buffer
            data_point = await data_buffer.pop.remote()
            if data_point is None:
                break
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
                
                # Ensure logprobs is enabled to get log probabilities
                sampling_params_with_logprobs = copy.deepcopy(sampling_params)
                if info.get("phase") == "answering":
                    sampling_params_with_logprobs["logprobs"] = True
                
                with simple_timer("generate_sequences", metrics):
                    output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params_with_logprobs,
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
                last_info = copy.deepcopy(info)
                
                messages, reward, terminated, truncated, info = env.step(actions)
                done = np.logical_or(terminated, truncated)
                
                # Check if we're in the answering phase
                if last_phase == "answering":
                    if response_logprobs is not None and len(response_logprobs) > 0 and not is_val:
                        avg_log_prob = np.mean(response_logprobs)
                        prob = np.exp(avg_log_prob)
                        adjusted_reward = reward / prob # * avg_log_prob / prob
                        adjusted_reward = np.clip(adjusted_reward, -5.0, 5.0)
                        reward = adjusted_reward
                    outputs[-1].rewards = reward
                else:
                    # Persist env-provided info (including phase-transition metadata) with the turn.
                    extra_fields = {}
                    if isinstance(info, dict):
                        # MMLUEnv (sp_env) provides answering_messages and z_content as strings
                        if "answering_messages" in info and "z_content" in info:
                            answering_messages = info["answering_messages"]
                            z_content = info["z_content"]
                            
                            # Tokenize answering_messages to get token sequence
                            answering_prompt_ids = await self.loop.run_in_executor(
                                None,
                                lambda: self.tokenizer.apply_chat_template(
                                    answering_messages,
                                    tools=self.tool_schemas,
                                    add_generation_prompt=True,
                                    tokenize=True,
                                    **self.apply_chat_template_kwargs,
                                ),
                            )
                            
                            # Try z_content first, then " " + z_content if needed
                            # First attempt: use z_content
                            z_content_tokens = self.tokenizer.encode(z_content, add_special_tokens=False)
                            z_index2 = self._calculate_z_index2(response_ids, z_content_tokens)
                            
                            # Check if z_index2 found a match (any True values in the response_ids part)
                            z_index2_matched = any(z_index2)
                            
                            if not z_index2_matched:
                                z_content_with_space = " " + z_content
                                z_content_tokens = self.tokenizer.encode(z_content_with_space, add_special_tokens=False)
                                z_index2 = self._calculate_z_index2(response_ids, z_content_tokens)
                            
                            # Calculate z_index: where z_content tokens appear in answering_prompt_ids
                            z_index = self._calculate_z_index(answering_prompt_ids, z_content_tokens)
                            
                            len_z_index = z_index[1] - z_index[0] if z_index is not None else 0
                            len_z_index2 = sum(z_index2)
                            # assert len_z_index == len_z_index2, f"z_index and z_index2 must have the same length, got {len_z_index} and {len_z_index2}, z_content: {z_content}, answering_messages: {answering_messages}, response_text: {actions}"
                            # warning only
                            if len_z_index != len_z_index2:
                                print(f"Warning: z_index and z_index2 must have the same length, got {len_z_index} and {len_z_index2}, len(z_content_tokens): {len(z_content_tokens)}")
                                print("---")
                            
                            # Store tokenized versions in extra_fields
                            extra_fields["answering_messages"] = answering_prompt_ids  # List[int]
                            extra_fields["z_content"] = z_content_tokens  # List[int]
                            extra_fields["z_index"] = z_index  # (start, end) or None
                            extra_fields["z_index2"] = z_index2  # List[bool]
                    
                    turn_data = AgentLoopOutput(
                        prompt_ids=prompt_ids,
                        response_ids=response_ids,
                        response_mask=response_mask,
                        response_logprobs=response_logprobs,
                        extra_fields=extra_fields,
                        metrics=metrics,
                        rewards=reward,
                        done=True,
                        num_turns=num_turns,
                        env_idx=env_idx,
                        info=info.get("metrics", info),
                    )
                    outputs.append(turn_data)
                
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
            
        # # Add final turn data for this episode
        # turn_data = AgentLoopOutput(
        #     prompt_ids=last_prompt_ids,
        #     response_ids=[outputs[-1].response_ids[0]] if outputs else [151645],
        #     response_mask=[1],
        #     metrics=dict(),
        #     rewards=0.0,
        #     done=True,
        #     num_turns=1,
        #     env_idx=env_idx,
        #     info=last_info["metrics"],
        # )
        # outputs.append(turn_data)
        
        return outputs