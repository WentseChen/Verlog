import copy
import re

from verl.envs.captioners.base import BaseCaptioner


class NaiveCaptioner(BaseCaptioner):
    """A captioner that generates actions based on observations without complex reasoning."""

    def __init__(self, prompt_builder):
        """Initialize the NaiveCaptioner with a client and prompt builder."""
        super().__init__(prompt_builder)

    def get_obs(self, obs):
        """Generate the next action based on the observation.

        Args:
            obs (dict): The current observation in the environment.

        Returns:
            prompt: The prompt for the LLM, formatted as a list of dictionaries with 'role' and 'content'.
        """

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        naive_instruction = """
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction
            
        # TODO: remove the transformation
        prompt = []
        for message in messages:
            role = message.role
            content = message.content
            prompt.append({"role": role, "content": content})

        return prompt
    
    def update_action(self, full_action, executed_action):
        # self.prompt_builder.update_reasoning(full_action)
        self.prompt_builder.update_action(executed_action)
