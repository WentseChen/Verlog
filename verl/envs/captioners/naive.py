import copy
import re

from verl.envs.captioners.base import BaseCaptioner


class NaiveCaptioner(BaseCaptioner):
    """An agent that generates actions based on observations without complex reasoning."""

    def __init__(self, prompt_builder):
        """Initialize the NaiveCaptioner with a client and prompt builder."""
        super().__init__(prompt_builder)

    def get_obs(self, obs, prev_action=None):
        """Generate the next action based on the observation and previous action.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            str: The selected action from the LLM response.
        """
        
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        naive_instruction = """
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
        """.strip()

        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + naive_instruction
            
        # TODO: remove the transformation
        new_messages = []
        for message in messages:
            role = message.role
            content = message.content
            new_messages.append({"role": role, "content": content})

        return new_messages

    def get_action(self, answer):
        """Sanitize the final answer, keeping only alphabetic characters.

        Args:
            answer (LLMResponse): The response from the LLM.

        Returns:
            LLMResponse: The sanitized response.
        """
        

        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        final_answer = copy.deepcopy(answer)
        final_answer = filter_letters(final_answer)
        self.prompt_builder.update_action(final_answer)

        return final_answer
