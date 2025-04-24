from ..prompt_builder import create_prompt_builder


def make_captioner(config):
    """Create a captioner agent based on the provided configuration.

    Args:
        config: Configuration object containing settings for the agent and client.

    Returns:
        Agent: An instance of the selected agent type, configured with the client and prompt builder.
    """
    prompt_builder = create_prompt_builder(config.envs.captioner)

    if config.envs.captioner.type == "naive":
        from .naive import NaiveCaptioner
        return NaiveCaptioner(prompt_builder)
    elif config.envs.captioner.type == "cot":
        from .cot import COTCaptioner
        return COTCaptioner(prompt_builder)
    else:
        raise ValueError(f"Unknown captioner type: {config.captioner.type}")
