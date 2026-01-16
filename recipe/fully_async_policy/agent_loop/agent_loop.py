# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import logging
import os
from typing import Any, Optional

import hydra
import numpy as np
import ray
from omegaconf import DictConfig

from recipe.fully_async_policy.vllm_rollout.vllm_async_server import FullyAsyncvLLMReplica
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorkerBase,
    AsyncLLMServerManager,
    _agent_loop_registry,
    _DummyConfig,
    get_trajectory_info,
)
from verl.protocol import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.rollout_trace import rollout_trace_attr
from verl.workers.rollout.replica import TokenOutput

from verl.envs.env import Env
from verl.envs.environments import make_env
from verl.envs.captioners import make_captioner

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class WorkerPoolQueue:
    """Manages worker pool using asyncio.Queue"""
    
    def __init__(self, workers: list):
        self.workers = workers
        self.queue = asyncio.Queue()
        # Initialize queue with all workers
        for idx, worker in enumerate(workers):
            self.queue.put_nowait((idx, worker))
    
    async def acquire(self) -> tuple[int, any]:
        """Acquire a worker (waits if none available)"""
        return await self.queue.get()
    
    async def release(self, worker_index: int, worker: any):
        """Release worker back to pool"""
        await self.queue.put((worker_index, worker))
    
    def qsize(self) -> int:
        """Return number of available workers"""
        return self.queue.qsize()
    
class FullyAsyncLLMServerManager(AsyncLLMServerManager):
    async def generate_for_partial(self, request_id, prompt_ids, sampling_params, **kwargs_extra) -> TokenOutput:
        """Generate tokens from prompt ids. with partial rollout function"""
        server = self._choose_server(request_id)
        output = await server.generate_for_partial.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            **kwargs_extra,
        )
        return output


class FullyAsyncAgentLoopOutput(AgentLoopOutput):
    """Agent loop output."""

    is_cancel: bool = False
    """Indicates whether the request was interrupted"""
    log_probs: list[float] = None
    """Response token log probs including LLM generated token, tool response token."""
    param_version_start: int = 0
    """Indicate start parameter version when this response is generated"""
    param_version_end: int = 0
    """Indicate end parameter version when this response is generated, used for partial rollout"""
    rewards: float = 0.0
    """Cumulative rewards, for RL agent loop."""
    dones: bool = False
    """Whether the episode is done, for RL agent loop."""
    env_idx: int = -1
    """Environment index, for parallel environment handling."""
    turn_idx: int = -1
    """Turn index in the episode, for GAE calculation."""
    env_info: dict = None
    """Additional environment info returned after step."""

@ray.remote
class FullyAsyncAgentLoopWorker(AgentLoopWorkerBase):
    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        self.server_manager = FullyAsyncLLMServerManager(config, server_handles)
        super().__init__(config, server_handles, reward_router_address)
        
        env_name = config.envs.env_name
        env_task = config.envs.task
        base_env = make_env(env_name, env_task, config)
        captioner = make_captioner(config)
        self.env = Env(env_name, config, base_env, captioner)
        self.env.reset()

    async def generate_sequences_no_post(
        self, batch: DataProto, partial_output_list: Optional[list[AgentLoopOutput]], worker_index: Optional[int] = 0
    ) -> list[AgentLoopOutput]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[FullyAsyncAgentLoopOutput]: List of agent loop outputs, one per sample in the batch.
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        )

        if not partial_output_list:
            partial_output_list = [None] * len(batch)

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs["output"] = partial_output_list[i]
            
            kwargs["env_actor"] = self.env
            kwargs["env_idx"] = worker_index
            
            tasks.append(
                asyncio.create_task(self._partial_run_agent_loop(sampling_params, trajectory_info[i], **kwargs))
            )
        return await asyncio.gather(*tasks)

    async def _partial_run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> AgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            return await agent_loop.run(sampling_params, **kwargs)


class FullyAsyncAgentLoopManager(AgentLoopManager):
    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        self.agent_loop_workers_class = FullyAsyncAgentLoopWorker
        self.rollout_replica_class = FullyAsyncvLLMReplica

        self.rm_wg = rm_wg
        self.rollout_replicas = None
        self.server_handles = None
        self.server_addresses = None
        self.agent_loop_workers = None
        
        self._worker_pool_lock = asyncio.Lock()
        
    @classmethod
    async def create(cls, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        instance = cls(config, worker_group, rm_wg)
        await instance._async_init()
        return instance

    async def _async_init(self):
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward import RewardModelManager

            self.reward_model_manager = RewardModelManager(self.config.reward_model, self.rm_wg)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        await self._initialize_llm_servers_async()
        self._init_agent_loop_workers()

    async def _initialize_llm_servers_async(self):
        rollout_world_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group:
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    async def generate_single_sample_async(
        self,
        sample: DataProto,
        partial_output_list: Optional[list[AgentLoopOutput]],
    ) -> list[AgentLoopOutput]:
        """
        Asynchronously process a single sample

        Args:
            sample: Single sample data
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[AgentLoopOutput]: Processing results
        """
        async with self._worker_pool_lock:
            if not hasattr(self, "worker_pool"):
                self.worker_pool = WorkerPoolQueue(self.agent_loop_workers)
        
        worker_index, worker = await self.worker_pool.acquire()
        
        try:
            output_future = worker.generate_sequences_no_post.remote(
                sample, partial_output_list, worker_index
            )
            result = await asyncio.wrap_future(output_future.future())
            return result
        finally:
            await self.worker_pool.release(worker_index, worker)
        
        # worker_index, worker = self._select_best_worker()
        # output_future = worker.generate_sequences_no_post.remote(sample, partial_output_list, worker_index)
        # return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        """Select the best worker, simple round-robin load balancing"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        selected_work_index = self._worker_index
        return selected_work_index, worker

    async def cancel(self):
        await asyncio.gather(*[replica.cancel() for replica in self.rollout_replicas])

    async def resume(self):
        await asyncio.gather(*[replica.resume() for replica in self.rollout_replicas])

    async def wake_up(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    async def sleep(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])

    async def reset_prefix_cache(self):
        await asyncio.gather(*[replica.reset_prefix_cache() for replica in self.rollout_replicas])
