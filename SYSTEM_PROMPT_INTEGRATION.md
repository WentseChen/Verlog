# System Prompt Integration for AsyncTickerAdmissionsEnv

## Summary of Changes

The AsyncTickerAdmissionsEnv has been updated to:
1. Accept system prompts in the config
2. Log system prompt token lengths using the tokenizer
3. Return observations in VERL-compatible format

## What Changed

### 1. VERL-Compatible Observation Format

**Before:**
```python
observations = {
    "prof_1": "obs_text",  # Raw string
    "prof_2": "",
    "prof_3": ""
}
```

**After:**
```python
observations = {
    "prof_1": {"prompt": "obs_text"},  # Dict with "prompt" key
    "prof_2": {"prompt": ""},
    "prof_3": {"prompt": ""}
}
```

This format is compatible with the VERL `Env` wrapper which expects:
```python
obs_text = observations.get(agent_id, {}).get("prompt", "")
```

### 2. System Prompt Support

The environment now accepts a `system_prompt` parameter in the config:

**Single prompt for all agents:**
```python
config = {
    "professor_ids": ["prof_1", "prof_2", "prof_3"],
    "students_per_batch": 5,
    "token_budget": 500,
    "system_prompt": "You are a professor in an admissions committee...",
}
```

**Per-agent prompts:**
```python
config = {
    "professor_ids": ["prof_1", "prof_2", "prof_3"],
    "students_per_batch": 5,
    "token_budget": 500,
    "system_prompt": {
        "prof_1": "You are Professor Smith, an expert in ML...",
        "prof_2": "You are Professor Jones, an expert in NLP...",
        "prof_3": "You are Professor Lee, an expert in CV...",
    },
}
```

### 3. System Prompt Token Length Logging

The environment now accepts an optional `tokenizer` parameter and logs token lengths:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
env = AsyncTickerAdmissionsEnv(config, tokenizer=tokenizer)
```

Token lengths are computed during initialization and included in the info dict:

```python
observations, infos = env.reset()
info = infos["prof_1"]

# Access system prompt
system_prompt = info["system_prompt"]["prof_1"]

# Access token length (computed using actual tokenizer, not word count!)
token_length = info["system_prompt_token_length"]["prof_1"]
```

## Usage Example

```python
from transformers import AutoTokenizer
from verl.envs.async_ticker_env import AsyncTickerAdmissionsEnv

# Load tokenizer for token counting
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Configure environment with system prompt
config = {
    "professor_ids": ["prof_1", "prof_2", "prof_3"],
    "students_per_batch": 5,
    "token_budget": 500,
    "feature_dim": 5,
    "vote_threshold": 0.5,
    "seed": 42,
    "system_prompt": "You are a professor participating in a student admissions committee.",
}

# Create environment with tokenizer
env = AsyncTickerAdmissionsEnv(config, tokenizer=tokenizer)

# Reset and check info
observations, infos = env.reset()
info = infos["prof_1"]

print(f"System prompt: {info['system_prompt']['prof_1']}")
print(f"Token length: {info['system_prompt_token_length']['prof_1']} tokens")

# Access observation
obs_text = observations["prof_1"]["prompt"]
```

## Integration with VERL

The environment now follows VERL conventions:

1. **Multi-agent dict structure**: Returns dicts for observations, rewards, terminations, truncations, infos
2. **Observation format**: `observations[agent_id] = {"prompt": obs_text}`
3. **System prompt in info**: Available for wrapper to build chat messages
4. **Token length logging**: Helps track prompt sizes for buffer management

## Backward Compatibility

**Breaking Change**: The observation format has changed from strings to dicts. Code that expects:
```python
obs_text = observations[agent_id]
```

Must be updated to:
```python
obs_text = observations[agent_id]["prompt"]
```

All unit tests have been updated to match the new format.

## Testing

Run the test suite to verify:
```bash
python -m pytest tests/envs/test_async_ticker_env.py -v
```

All 28 tests should pass, including new tests for:
- System prompt with single string
- System prompt with per-agent prompts
- System prompt token length logging

## Standalone Test

Run the standalone test to see system prompts in action:
```bash
python test_env_standalone.py
```

This will:
1. Load a tokenizer
2. Create environment with system prompt
3. Log system prompt token lengths
4. Run inference with a small LLM
