import copy
import re

from verl.envs.captioners.base import BaseCaptioner

class COTCaptioner(BaseCaptioner):
    """A captioner that performs actions using a chain-of-thought reasoning process."""

    def __init__(self, prompt_builder):
        """Initialize the ChainOfThoughtAgent with a client, prompt builder, and configuration.

        Args:
            prompt_builder (PromptBuilder): Object to build prompts for the agent.
        """
        super().__init__(prompt_builder)

    def get_obs(self, obs):
        """Convert environment observation to a prompt for the LLM.

        Args:
            obs (dict): The current observation in the environment.

        Returns:
            prompts: The prompt for the LLM, formatted as a list of dictionaries with 'role' and 'content'.
        """

        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()

        # Add CoT-specific instructions to the prompt
        cot_instructions = """
What will you do next? Please respond in the following format:
THINK: step-by-step reasoning
ACTION: One valid action from the allowed set
        """.strip()

        messages[-1].content += "\n\n" + cot_instructions

        # convert messages to dict format
        prompts = []
        for message in messages:
            role = message.role
            content = message.content
            prompts.append({"role": role, "content": content})

        return prompts
    
    def update_action(self, full_action, executed_action):
        self.prompt_builder.update_reasoning(full_action)
        self.prompt_builder.update_action(executed_action)
