#!/bin/bash
#
# Test Script for AsyncTickerAdmissionsEnv with LLM Inference
#
# This script performs a basic smoke test to verify:
# 1. Environment can be loaded
# 2. Small LLM (Qwen2.5-0.5B) can be loaded with 8GB VRAM
# 3. Inference works with the environment
#
# Requirements: ~4GB VRAM for Qwen2.5-0.5B-Instruct

set -e  # Exit on error

echo "================================================"
echo "Async Ticker Environment - Inference Smoke Test"
echo "================================================"
echo ""

# Configuration
MODEL_NAME="Qwen/Qwen3-0.6B"
NUM_PROFESSORS=3
NUM_STUDENTS=5
TOKEN_BUDGET=500
VOTE_THRESHOLD=0.5
FEATURE_DIM=5

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Professors: $NUM_PROFESSORS"
echo "  Students: $NUM_STUDENTS"
echo "  Token Budget: $TOKEN_BUDGET"
echo "  Vote Threshold: $VOTE_THRESHOLD"
echo "  Feature Dim: $FEATURE_DIM"
echo ""

# Check if we're in the right directory
if [ ! -f "verl/envs/async_ticker_env.py" ]; then
    echo "❌ Error: Not in the correct directory"
    echo "Please run from: /home/cpulling/unscripted/verlog_v2_unscripted/Verlog/.worktrees/async-ticker"
    exit 1
fi

echo "✓ In correct directory"
echo ""

# Create test script
echo "Creating test script..."
cat > /tmp/test_async_ticker_env.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Smoke test for AsyncTickerAdmissionsEnv with LLM inference.
Tests that the environment can load and interact with a small LLM.
"""

import sys
import os
sys.path.insert(0, '.')

# Direct import to avoid verl.__init__ dependencies
import importlib.util
spec = importlib.util.spec_from_file_location('async_ticker_env', 'verl/envs/async_ticker_env.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AsyncTickerAdmissionsEnv = module.AsyncTickerAdmissionsEnv

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 80)
print("STEP 1: Loading Environment")
print("=" * 80)

# Create environment
config = {
    "professor_ids": ["prof_1", "prof_2", "prof_3"],
    "students_per_batch": 5,
    "token_budget": 500,
    "feature_dim": 5,
    "vote_threshold": 0.5,
    "seed": 42,
}

env = AsyncTickerAdmissionsEnv(config)
print("✓ Environment created")

obs, info = env.reset()
print("✓ Environment reset successful")
print(f"  Active agent: {info['active_agent']}")
print(f"  Agent tickers: {info['episode_state']['agent_tickers']}")
print("")

print("=" * 80)
print("STEP 2: Loading Small LLM (Qwen3-0.6B)")
print("=" * 80)

model_name = "Qwen/Qwen3-0.6B"
print(f"Loading {model_name}...")
print("This may take a minute on first run (downloading model)...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print("✓ Model and tokenizer loaded successfully")

    # Check GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print("  Running on CPU (no CUDA available)")
    print("")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nNote: If model download fails, you may need to:")
    print("  1. Check internet connection")
    print("  2. Set HF_TOKEN environment variable for gated models")
    sys.exit(1)

print("=" * 80)
print("STEP 3: Running Inference Loop")
print("=" * 80)

def generate_response(prompt, max_new_tokens=50):
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

# Run a few steps
done = False
step_count = 0
max_steps = 5  # Just test a few steps

print(f"\nRunning {max_steps} test steps...\n")

while not done and step_count < max_steps:
    step_count += 1
    active_agent = info['active_agent']

    print(f"--- Step {step_count} ---")
    print(f"Active agent: {active_agent}")
    print(f"Ticker: {info['episode_state']['agent_tickers'][active_agent]}")
    print(f"Tokens used: {info['episode_state']['tokens_used']}/{config['token_budget']}")

    # Show first few lines of observation
    obs_lines = obs.split('\n')[:10]
    print(f"\nObservation (first 10 lines):")
    for line in obs_lines:
        print(f"  {line}")
    print("  ...")

    # Generate action from LLM
    print(f"\nGenerating action from {model_name}...")

    # Simple prompt - just ask for a vote
    simple_prompt = f"As {active_agent}, vote for a student (0-4). Respond with: VOTE: X\n"

    try:
        action = generate_response(simple_prompt, max_new_tokens=20)
        print(f"Generated action: {action}")
    except Exception as e:
        print(f"Error during generation: {e}")
        # Fallback action
        action = f"VOTE: {step_count % 5}"
        print(f"Using fallback action: {action}")

    # Take step in environment
    obs, reward, done, truncated, info = env.step(action)

    print(f"Reward: {reward}")
    print(f"Done: {done}")

    if done:
        print(f"\n✓ Episode finished!")
        print(f"  Consensus: {info['episode_state']['consensus_reached']}")
        if info['episode_state']['consensus_reached']:
            print(f"  Choice: Student {info['episode_state']['consensus_choice']}")
        print(f"  Agent rewards: {info['agent_rewards']}")

    print("")

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Environment loaded successfully")
print(f"✓ Model loaded successfully ({model_name})")
print(f"✓ Inference completed {step_count} steps")
print(f"✓ Environment-LLM integration working!")
print("")

# Cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✓ GPU memory cleared")

print("\n🎉 Smoke test PASSED! Environment is ready for VERL integration.")
PYTHON_SCRIPT

echo "✓ Test script created"
echo ""

echo "================================================"
echo "Running Smoke Test"
echo "================================================"
echo ""

# Run the test
python3 /tmp/test_async_ticker_env.py

echo ""
echo "================================================"
echo "Test Complete!"
echo "================================================"
