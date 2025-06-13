import numpy as np
import torch
from multiprocessing import Process, Pipe
import random

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VecEnv:
    
    def __init__(self, env_name, config, env_fns, captioner_fns):
        
        self.config = config
        self.n_rollouts = config.envs.n_rollouts
        assert len(env_fns) == self.n_rollouts, "Number of env_fns must match n_rollouts"
        
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_rollouts)])
        self.processes = []
        for rank, (work_remote, remote, env_fn, captioner_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns, captioner_fns)):
            p = Process(
                target=worker,
                args=(rank, work_remote, remote, env_name, CloudpickleWrapper(env_fn), CloudpickleWrapper(captioner_fn)),
            )
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
            self.processes.append(p)
        
        for remote in self.work_remotes:
            remote.close()
            
    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, terminated, truncated, infos = zip(*results)
        
        merged_infos = {}
        for info in infos:
            for key, value in info["metrics"].items():
                if key not in merged_infos.keys():
                    merged_infos[key] = []
                merged_infos[key].append(value)
        for key in merged_infos.keys():
            merged_infos[key] = np.mean(merged_infos[key])
        
        return obs, np.stack(rews), np.stack(terminated), np.stack(truncated), merged_infos
    
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        observations, infos = zip(*[remote.recv() for remote in self.remotes])
        return observations, infos
    
    def action(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('action', action))
        results = [remote.recv() for remote in self.remotes]
        actions = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        
def worker(rank, remote, parent_remote, env_name, env_fn_wrapper, captioner_fn_wrapper):
    random.seed(rank)
    np.random.seed(rank)
    parent_remote.close()
    env = env_fn_wrapper.x()
    captioner = captioner_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        
        if cmd == 'step':
            reasoning, action, valid_action, metrics = env.extract_action(data)
            env_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            captioner.update_action(reasoning, valid_action)
            obs = captioner.get_obs(env_obs)
            if done:
                captioner.reset()
                env_obs, _ = env.reset()
                # TODO: move this part of code to the proper place ====
                instructions = None
                if env_name == "babyai":
                    instructions = env_obs["mission"]
                inst_prompt = env.get_instruction_prompt(instructions=instructions)
                captioner.prompt_builder.update_instruction_prompt(inst_prompt)
                # =====================================================
                obs = captioner.get_obs(env_obs)
            
            # TODO: better way to handle metrics #######
            info["metrics"] = metrics
            ############################################
            
            remote.send((obs, reward, terminated, truncated, info))
            
        elif cmd == 'reset':
            env_obs, info = env.reset(seed=rank)
            captioner.reset()
            # TODO: move this part of code to the proper place ====
            instructions = None
            if env_name == "babyai":
                instructions = env_obs["mission"]
            inst_prompt = env.get_instruction_prompt(instructions=instructions)
            captioner.prompt_builder.update_instruction_prompt(inst_prompt)
            # =====================================================
            obs = captioner.get_obs(env_obs)
            remote.send((obs, info))
            
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        else:
            raise NotImplementedError

        
  
    
