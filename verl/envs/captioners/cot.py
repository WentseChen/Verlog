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
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        # Add CoT-specific instructions to the prompt
        cot_instructions = """
First think about what's the best course of action step by step.
Finally, provide a single output action at the end of the message in the form of: ACTION: <action>
        """.strip()

        messages[-1].content += "\n\n" + cot_instructions

        # TODO: remove the transformation
        new_messages = []
        for message in messages:
            role = message.role
            content = message.content
            new_messages.append({"role": role, "content": content})

        return new_messages
    
        # # Generate the CoT reasoning
        # cot_reasoning = self.client.generate(messages)

        # # Extract the final answer from the CoT reasoning
        # final_answer = self._extract_final_answer(cot_reasoning)

        # return final_answer

    def get_action(self, reasoning):
        """Extract the final action from the chain-of-thought reasoning response.

        Args:
            reasoning (LLMResponse): The response containing CoT reasoning and action.

        Returns:
            LLMResponse: The response with the extracted final action.
        """

        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        answer = copy.deepcopy(reasoning)
        self.prompt_builder.update_reasoning(reasoning)
        answer = filter_letters(answer)
        answer = answer.split("ACTION:")[-1].strip()

        return answer
