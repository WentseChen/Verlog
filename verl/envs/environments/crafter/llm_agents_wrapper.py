import gymnasium as gym
from PIL import Image
import numpy as np
import re

POSSIBLE_ACTIONS = [
    "Move West",
    "Move East",
    "Move North",
    "Move South",
    "Do ",
    "Collect ",
    "Drink ",
    "Hit ",
    "Attack ",
    "Sleep",
    "Place ",
    "Make ",
    "Craft ",
    "Moving West",
    "Moving East",
    "Moving North",
    "Moving South",
    "Collecting ",
    "Drinking ",
    "Hitting ",
    "Attacking ",
    "Sleeping",
    "Placing ",
    "Making ",
    "Crafting ",
]

class CrafterLLMAgentsWrapper(gym.Wrapper):
    def __init__(self, env, vlm=False, **kwargs):
        super().__init__(env)
        self.env = env
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
        
        full_action = str(action)
        
        if "ACTION:" in action:
            action = action.split("ACTION:")[-1].strip()
        elif "action:" in action:
            action = action.split("action:")[-1].strip()
        elif "Action" in action:
            action = action.split("Action")[-1].strip()
            
        lower_pred_action = action.lower()
        action = lower_pred_action.title()
        if "\n" in action:
            action = action.split("\n")[0].strip()
        if action.endswith(".") or action.endswith(","):
            action = action[:-1]
        
        if action == "Move North West" or action == "Move NorthWest" or action == "Move North-West" or action == "Move Northwest":
            action = np.random.choice(["Move North", "Move West"])
        elif action == "Move South West" or action == "Move SouthWest" or action == "Move South-West" or action == "Move Southwest":
            action = np.random.choice(["Move South", "Move West"])
        elif action == "Move North East" or action == "Move NorthEast" or action == "Move North-East" or action == "Move Northeast":
            action = np.random.choice(["Move North", "Move East"])
        elif action == "Move South East" or action == "Move SouthEast" or action == "Move South-East" or action == "Move Southeast":
            action = np.random.choice(["Move South", "Move East"])
        elif action == "Move North 1 Step" or action == "Move North 2 Steps" or action == "Move North 3 Steps":
            action = "Move North"
        elif action == "Move South 1 Step" or action == "Move South 2 Steps" or action == "Move South 3 Steps":
            action = "Move South"
        elif action == "Move East 1 Step" or action == "Move East 2 Steps" or action == "Move East 3 Steps":
            action = "Move East"
        elif action == "Move West 1 Step" or action == "Move West 2 Steps" or action == "Move West 3 Steps":
            action = "Move West"
        elif action == "Collect Stone" or action == "Mine Stone":
            action = "Do"
        elif action == "Chop Tree" or action == "Chop" or action == "Chop Trees" or action == "Cut Tree" or action == "Chop_Tree" or action == "Cut" or action == "Collect Wood":
            action = "Do"
        elif action == "Drink" or action == "Drink From Water" or action == "Drink From Lakes":
            action = "Do"
        elif action == "Mine" or action == "Collect Coal" or action == "Collect Iron" or action == "Mine Coal":
            action = "Do"
        elif action == "Defeat Zombie" or action == "Defeat Skeleton" or action == "Eat Cow" or action == "Attack" or action == "Collect Sapling" :
            action = "Do"
        elif action == "Make Table" or action == "Make Crafting Table":
            action = "Place Table"
        elif action == "Make Furnace":
            action = "Place Furnace"
            
        if "Do" in action and "(" in action:
            pattern = r'\([^)]*\)'
            action = re.sub(pattern, '', action).strip()
            
        is_valid = action in self.language_action_space
        valid_action = action if is_valid else self.default_action
        
        if "ACTION:" in full_action:
            full_action = full_action.split("ACTION:")[0].strip()
        elif "action:" in action:
            full_action = full_action.split("action:")[0].strip()
        elif "Action" in full_action:
            full_action = full_action.split("Action")[0].strip()
        full_action += "\nACTION: " + valid_action + ".\n"
        
        total_action_occurrences = 0
        for p_action in POSSIBLE_ACTIONS:
            total_action_occurrences += full_action.lower().count(p_action.lower())
        
        total_but_occurrences = 0
        for word in ["However", "different", "but", "wait", "won't", "can't", "cannot", "another"]:
            total_but_occurrences += full_action.lower().count(word.lower())
        metrics = {
            "behavior/valid_action_ratio": is_valid * 1.0,
            "behavior/plan_length": total_action_occurrences,
            "behavior/backtrack_length": total_but_occurrences
        }
        
        return full_action, valid_action, is_valid, metrics
