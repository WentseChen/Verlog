from ..prompt_builder import create_prompt_builder
from .naive import NaiveCaptioner


def make_captioner(config):
    """Create a captioner agent based on the provided configuration.

    Args:
        config: Configuration object containing settings for the agent and client.

    Returns:
        Agent: An instance of the selected agent type, configured with the client and prompt builder.
    """
    prompt_builder = create_prompt_builder(config.envs.captioner)

    if config.envs.captioner.type == "naive":
        return NaiveCaptioner(prompt_builder)
    else:
        raise ValueError(f"Unknown captioner type: {config.captioner.type}")
