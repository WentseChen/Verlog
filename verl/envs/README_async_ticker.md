# AsyncTickerAdmissionsEnv

Multi-agent negotiation environment with async time ticker mechanics.

## Overview

Agents negotiate over student admissions with:
- **Time tickers**: Turn order based on accumulated token counts
- **Wait actions**: Agents can wait for specific events
- **Consensus detection**: Episode ends when all agents vote consecutively for same choice
- **Individual rewards**: Outcome quality + preference alignment

## Usage

```python
from verl.envs.async_ticker_env import AsyncTickerAdmissionsEnv

config = {
    "professor_ids": ["prof_1", "prof_2", "prof_3"],
    "students_per_batch": 10,
    "token_budget": 10000,
    "preference_weight": 0.5,
    "seed": 42
}

env = AsyncTickerAdmissionsEnv(config)
obs, info = env.reset()

done = False
while not done:
    # Get action from policy for active agent
    active_agent = info["active_agent"]
    action = get_action(obs, active_agent)  # Your policy

    obs, reward, done, truncated, info = env.step(action)

# Retrieve per-agent rewards
agent_rewards = info["agent_rewards"]
```

## Action Format

### Discussion
Any text without special keywords:
```
"I think Student 5 has strong qualifications in research."
```

### Voting
Use `VOTE: <student_id>` format:
```
"I've made my decision. VOTE: 5"
```

### Waiting
Use `WAIT_FOR: <condition>` format:
```
"WAIT_FOR: prof_2"           # Wait for specific agent
"WAIT_FOR: any_response"     # Wait for any agent
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `professor_ids` | List of agent IDs | Required |
| `students_per_batch` | Number of students to choose from | Required |
| `token_budget` | Max total tokens before forced voting | Required |
| `preference_weight` | Weight of preference bonus in rewards | 0.5 |
| `seed` | Random seed | None |

## Episode Termination

Episodes end when:
1. **Early consensus**: N consecutive matching votes (N = number of agents)
2. **Budget exhausted**: Total tokens across all agents ≥ token_budget

## Rewards

- **No consensus**: All agents get 0.0
- **Consensus reached**: `outcome_reward + (preference * preference_weight)`
  - `outcome_reward`: 1.0 if choice matches ground truth, else 0.0
  - `preference`: Agent's preference value for chosen student

## Info Dict

The `info` dict returned by `step()` and `reset()` contains:

```python
{
    "active_agent": str,              # Which agent speaks next
    "agent_idx": int,                 # Index in professor_ids
    "episode_state": {
        "agent_tickers": dict,
        "message_history": list,
        "tokens_used": int,
        "token_budget": int,
        "waiting_agents": dict,
        "consensus_reached": bool,
        "consensus_choice": int | None
    },
    "agent_rewards": dict,            # Populated when done=True
    "ground_truth": dict,
    "student_batch": list
}
```

## Integration with VERL

The environment exposes `active_agent` and `agent_idx` in the info dict for agent loop integration. One episode produces N agent-specific trajectories by tracking which agent generated each message in `message_history`.
