import copy
import re

from verl.envs.captioners.base import BaseCaptioner

class COTCaptioner(BaseCaptioner):
    """An agent that performs actions using a chain-of-thought reasoning process."""

    def __init__(self, prompt_builder):
        """Initialize the ChainOfThoughtAgent with a client, prompt builder, and configuration.

        Args:
            client_factory (LLMClientWrapper): A factory for creating the LLM client instance.
            prompt_builder (PromptBuilder): Object to build prompts for the agent.
            config: Configuration object containing settings for the agent.
        """
        super().__init__(prompt_builder)

    def get_obs(self, obs, prev_action=None):
        """Generate the next action using chain-of-thought reasoning based on the current observation.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the final selected action.
        """
        # if prev_action:
        #     self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        # Add CoT-specific instructions to the prompt
        cot_instructions = """
First think about what's the best course of action step by step (in free form).
Finally, provide a single output action (the action you should take given the current observation) at the end of the message in the form of: ACTION: <action>
        """.strip()

        messages[-1].content += "\n\n" + cot_instructions

        # TODO: remove the transformation
        new_messages = []
        for message in messages:
            role = message.role
            content = message.content
            new_messages.append({"role": role, "content": content})

        return new_messages
    
    def update_action(self, reasoning, prev_action):
        self.prompt_builder.update_reasoning(reasoning)
        self.prompt_builder.update_action(prev_action)
