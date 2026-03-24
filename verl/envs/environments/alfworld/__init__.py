from .llm_agents_wrapper import ALFWORLD_ACTIONS, ALFWorldLLMAgentsWrapper


def get_instruction_prompt(env, task=None):
    action_strings = ",\n".join(f"{action}: ALFWorld admissible command" for action in ALFWORLD_ACTIONS)
    instruction_prompt = f"""
You are an expert agent operating in the ALFWorld environment.
Always reason first, then output a single final action.
Use only admissible commands from the current observation.

Common command families are:
{action_strings}

Output format requirements:
- Put your reasoning inside <think>...</think>
- Put exactly one final environment command inside <action>...</action>
""".strip()
    return instruction_prompt

