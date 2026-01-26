from __future__ import annotations

import argparse
from types import SimpleNamespace

import requests

from verl.envs.environments.dummy_openai_env import DummyOpenAIEnv


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dummy env against a vLLM OpenAI-compatible server.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="vLLM server base URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model name exposed by vLLM")
    parser.add_argument("--api-key", default="EMPTY", help="API key for vLLM server (if required)")
    parser.add_argument("--prompt", default="what's 1+1?", help="User prompt for dummy env")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    env = DummyOpenAIEnv(prompt=args.prompt)
    observations, _ = env.reset()
    agent_id = env.possible_agents[0]
    prompt = observations.get(agent_id, {}).get("prompt", args.prompt)
    messages = [{"role": "user", "content": prompt}]
    base_url = _normalize_base_url(args.base_url)
    url = f"{base_url}/chat/completions"

    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    headers = {"Authorization": f"Bearer {args.api_key}"}

    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"]
    print(f"assistant: {content}")

    env.step(content)


if __name__ == "__main__":
    main()
