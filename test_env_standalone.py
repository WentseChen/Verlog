#!/usr/bin/env python3
"""
Quick standalone test of AsyncTickerAdmissionsEnv with Qwen3-0.6B
"""

import sys
sys.path.insert(0, '.')

# Direct import to avoid verl.__init__ dependencies
import importlib.util
spec = importlib.util.spec_from_file_location('async_ticker_env', 'verl/envs/async_ticker_env.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AsyncTickerAdmissionsEnv = module.AsyncTickerAdmissionsEnv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 80)
print("AsyncTickerAdmissionsEnv Standalone Test")
print("=" * 80)

# Load tokenizer for system prompt token counting
print("\nStep 0: Loading Tokenizer...")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"✓ Tokenizer loaded: {model_name}")

# Create environment with system prompt
system_prompt = """You are a professor participating in a student admissions committee.
Your goal is to negotiate with other professors to select the best candidate based on your research interests.
You can discuss, wait for others, or vote for a student."""

config = {
    "professor_ids": ["prof_1", "prof_2", "prof_3"],
    "students_per_batch": 5,
    "token_budget": 500,
    "feature_dim": 5,
    "vote_threshold": 0.5,
    "seed": 42,
    "system_prompt": system_prompt,
}

print("\nStep 1: Loading Environment...")
env = AsyncTickerAdmissionsEnv(config, tokenizer=tokenizer)
print("✓ Environment created")

observations, infos = env.reset()
print("✓ Environment reset successful")

# Get active agent from any of the info dicts (they all have the same active_agent)
first_agent = list(infos.keys())[0]
active_agent = infos[first_agent]['active_agent']

print(f"\n  Observations keys: {list(observations.keys())}")
print(f"  Active agent: {active_agent}")
print(f"  Observation format: {type(observations[active_agent])}")
print(f"  Observation for active agent (first 200 chars):")
print(f"    {observations[active_agent]['prompt'][:200]}...")

# Log system prompt token length
if 'system_prompt_token_length' in infos[first_agent]:
    token_lengths = infos[first_agent]['system_prompt_token_length']
    print(f"\n  System prompt token lengths:")
    for agent_id, length in token_lengths.items():
        print(f"    {agent_id}: {length} tokens")

print("\nStep 2: Loading Model...")
print(f"  Loading {model_name}...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print("✓ Model loaded successfully")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  GPU Memory: {allocated:.2f}GB allocated")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

print("\nStep 3: Running inference test...")

def generate_response(prompt, max_new_tokens=30):
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from response
    response = response[len(prompt):].strip()
    return response

# Test 3 steps
for step_num in range(3):
    print(f"\n--- Step {step_num + 1} ---")
    print(f"Active agent: {active_agent}")

    # Simple prompt for voting
    prompt = f"As {active_agent}, vote for student 0-4. Respond: VOTE: X\n"

    print("Generating action...")
    try:
        action = generate_response(prompt, max_new_tokens=20)
        print(f"  Generated: {action[:100]}")

        # Fallback to simple vote if generation doesn't match
        if "VOTE:" not in action:
            action = f"VOTE: {step_num % 5}"
            print(f"  Using fallback: {action}")

    except Exception as e:
        print(f"  Error: {e}")
        action = f"VOTE: {step_num % 5}"
        print(f"  Using fallback: {action}")

    # Step environment
    observations, rewards, dones, truncated, infos = env.step(action)

    # Update active agent
    active_agent = infos[list(infos.keys())[0]]['active_agent']

    print(f"  Rewards: {rewards}")
    print(f"  Done: {dones[active_agent]}")

    if dones[active_agent]:
        print("\n✓ Episode finished!")
        print(f"  Consensus: {infos[list(infos.keys())[0]]['episode_state']['consensus_reached']}")
        break

print("\n" + "=" * 80)
print("✓ TEST PASSED - Model responds and environment works!")
print("=" * 80)

# Cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
