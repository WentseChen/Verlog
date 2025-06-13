import gymnasium as gym
from PIL import Image

BABYAI_ACTION_SPACE = [
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
]

POSSIBLE_ACTIONS = [
    "turn left",
    "turn to the left",
    "turn right",
    "turn to the right",
    "go forward",
    "move forward",
    "pick up",
    "pick it up",
    "drop",
    "toggle",
    "open",
    "turning left",
    "turning right",
    "moving forward",
    "picking up",
    "dropping",
    "toggling",
    "opening",
]
    

class BabyAITextCleanLangWrapper(gym.Wrapper):
    def __init__(self, env, vlm=False, **kwargs):
        super().__init__(env)
        self.language_action_space = BABYAI_ACTION_SPACE[:]
        self._mission = None
        self.progression = 0.0
        self.format_penalty = kwargs.get("format_penalty", 0.0)
        self.binary_reward = kwargs.get("binary_reward", False)
        # self.early_truncate = kwargs.get("early_truncate", False)

    @property
    def max_steps(self):
        return self.env.unwrapped.max_steps

    @property
    def interleaving_token(self):
        return self._interleaving_token

    @property
    def default_action(self):
        return "go forward"

    def get_text_action(self, action):
        return self.language_action_space[action.value]

    def get_prompt(self, obs, infos):
        image = Image.fromarray(self.env.unwrapped.get_pov_render(tile_size=16)).convert("RGB")

        def _form_prompt(description):
            return "\n".join([d.replace("You see ", "") for d in description])

        prompt = _form_prompt(infos["descriptions"])
        return prompt, image

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        prompt, image = self.get_prompt(obs, info)
        self._mission = obs["mission"]
        self.progression = 0.0
        # Following the convention from NetHack Language Wrapper for specifying
        # short term vs long term context here. There is no equivalent long term
        # context like e.g. inventory in BabyAI-Text.
        obs["text"] = {"long_term_context": prompt, "short_term_context": ""}
        obs["image"] = image
        return obs, info
    
    def extract_action(self, action):
        
        reasoning = str(action)
        
        if "ACTION:" in action:
            action = action.split("ACTION:")[-1].strip()
        elif "action:" in action:
            action = action.split("action:")[-1].strip()
        elif "Action" in action:
            action = action.split("Action")[-1].strip()
            
        lower_pred_action = action.lower()
        
        lower_pred_action = lower_pred_action.replace("_", " ")
        if lower_pred_action == "turnleft":
            lower_pred_action = "turn left"
        elif lower_pred_action == "turnright":
            lower_pred_action = "turn right"
        elif lower_pred_action == "goforward":
            lower_pred_action = "go forward"
        elif lower_pred_action == "pickup":
            lower_pred_action = "pick up"
            
        action = lower_pred_action
        
        valid_action = action if action in self.language_action_space else self.default_action
        
        total_action_occurrences = 0
        for p_action in POSSIBLE_ACTIONS:
            total_action_occurrences += reasoning.lower().count(p_action.lower())
        valid_count = 1 if valid_action == action else 0
        total_but_occurrences = 0
        for word in ["However", "different", "but", "wait", "won't", "can't", "cannot", "another"]:
            total_but_occurrences += reasoning.lower().count(word.lower())
        metrics = {
            "behavior/valid_action_ratio": valid_count,
            "behavior/plan_length": total_action_occurrences,
            "behavior/backtrack_length": total_but_occurrences
        }
        
        return reasoning, action, valid_action, metrics

    def step(self, action):
        
        valid_actions = False
        action_int = self.language_action_space.index(self.default_action)
        for a_idx, a in enumerate(self.language_action_space):
            lower_gt_action = a.lower()
            if lower_gt_action == action:
                action_int = a_idx
                valid_actions = True
                break
            
        obs, reward, terminated, truncated, infos = self.env.step(action_int)
        if reward > 0:
            self.progression = 1.0
        prompt, image = self.get_prompt(obs, infos)
        obs["text"] = {"long_term_context": prompt, "short_term_context": ""}
        obs["image"] = image
        
        if not valid_actions:
            reward = -self.format_penalty
            # terminated = True
            # truncated = True
        
        if self.binary_reward:
            reward = 1.0 if reward > 0 else reward
        
        return obs, reward*1.0, terminated, truncated, infos

    def get_stats(self):
        # No special stats tracking implemented for now
        return {"mission": self._mission, "progression": self.progression}
