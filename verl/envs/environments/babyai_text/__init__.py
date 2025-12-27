from .clean_lang_wrapper import BabyAITextCleanLangWrapper
from .llm_agents_wrapper import BabyAILLMAgentsWrapper

ACTIONS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object one step in front of you",
    "drop": "drop the object that you are holding",
    "toggle": "manipulate the object one step in front of you",
}


def get_instruction_prompt(env, mission="BabyAI-MixedTrainLocal-v0"):
    action_strings = ",\n".join(f"{action}: {description}" for action, description in ACTIONS.items())

    instruction_prompt = f"""
You are an agent playing a simple navigation game. Your goal is to {mission}. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

In a moment I will present you an observation.

Tips:
- Once the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.
- It doesn't make sense to repeat the same action over and over if the observation doesn't change.

PLAY!
""".strip()

    return instruction_prompt


# You are in a game. Your goal is to reach the red box. You can do these actions:

# * turn left: turn left
# * turn right: turn right
# * go forward: move forward one step
# * pick up: pick up the object under you
# * drop: drop the object you hold
# * toggle: use the object in front of you

# Next, you will see a view.

# Tips:

# * Use **toggle** if the object you want is in front.
# * Do not do the same action if nothing changes.

# Go!