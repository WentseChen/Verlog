import itertools

import crafter
import gym
import numpy as np
from PIL import Image

from verl.envs.environments import Strings
from verl.envs.environments.minihack import get_available_actions

class MiniHackLLMAgentsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.all_posible_default_action = ["north", "east", "south", "west"]
        self.default_action = self.all_posible_default_action[np.random.randint(4)]
        self.language_action_space = get_available_actions(env)
        self.format_penalty = kwargs.get("format_penalty", 0.0)
        self.binary_reward = kwargs.get("binary_reward", False)
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def step(self, action, is_valid=True):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not is_valid:
            reward = -self.format_penalty
        if self.binary_reward:
            reward = 1.0 if reward > 0 else reward
        return obs, reward*1.0, terminated, truncated, info
    
    def extract_action(self, action):
        
        self.default_action = self.all_posible_default_action[np.random.randint(4)]
        
        full_action = str(action)
        
        if "ACTION:" in action:
            action = action.split("ACTION:")[-1].strip()
        elif "action:" in action:
            action = action.split("action:")[-1].strip()
        elif "Action" in action:
            action = action.split("Action")[-1].strip()
            
        lower_action = action.lower()
            
        is_valid = lower_action in self.language_action_space
        valid_action = lower_action if is_valid else self.default_action
        
        total_but_occurrences = 0
        for word in ["However", "different", "but", "wait", "won't", "can't", "cannot", "another"]:
            total_but_occurrences += full_action.lower().count(word.lower())
        
        metrics = {
            "behavior/valid_action_ratio": is_valid * 1.0,
            "behavior/backtrack_length": total_but_occurrences
        }
        
        return full_action, valid_action, is_valid, metrics
