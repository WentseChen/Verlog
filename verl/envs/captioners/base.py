class BaseCaptioner:
    """Base class for captioners using prompt-based interactions."""

    def __init__(self, prompt_builder, env_name=None):
        """Initialize the agent with a client and prompt builder."""
        self.prompt_builder = prompt_builder

    def get_obs(self, observation, action):
        """Update the prompt with the observation and action."""
        self.prompt_builder.update_observation(observation)
        self.prompt_builder.update_action(action)

    def reset(self):
        """Reset the prompt builder."""
        self.prompt_builder.reset()
        
    def get_action(self, action):
        """Get the action from the prompt builder."""
        return self.prompt_builder.get_action(action)
