import itertools

import crafter
import gym
import numpy as np
from PIL import Image

from verl.envs.environments import Strings
from verl.envs.environments.minihack import get_available_actions

class MiniHackActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.all_posible_default_action = ["north", "east", "south", "west"]
        self.default_action = self.all_posible_default_action[np.random.randint(4)]
        self.language_action_space = get_available_actions(env)
    
    def extract_action(self, action):
        
        self.default_action = self.all_posible_default_action[np.random.randint(4)]
        
        reasoning = str(action)
        
        if "ACTION:" in action:
            action = action.split("ACTION:")[-1].strip()
        elif "action:" in action:
            action = action.split("action:")[-1].strip()
        elif "Action" in action:
            action = action.split("Action")[-1].strip()
            
        lower_pred_action = action.lower()
        
        valid_action = action if action in self.language_action_space else self.default_action
        
        return reasoning, action, valid_action
