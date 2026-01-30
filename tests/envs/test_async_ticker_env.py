# tests/envs/test_async_ticker_env.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest
import numpy as np
import importlib.util

# Import the module directly without going through verl.__init__
spec = importlib.util.spec_from_file_location(
    "async_ticker_env",
    os.path.join(os.path.dirname(__file__), '../../verl/envs/async_ticker_env.py')
)
async_ticker_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(async_ticker_env)
AsyncTickerAdmissionsEnv = async_ticker_env.AsyncTickerAdmissionsEnv


def test_env_creation():
    """Test basic environment instantiation."""
    config = {
        "professor_ids": ["prof_1", "prof_2", "prof_3"],
        "students_per_batch": 10,
        "token_budget": 10000,
        "preference_weight": 0.5,
        "seed": 42,
    }
    env = AsyncTickerAdmissionsEnv(config)
    assert env is not None
    assert len(env.professor_ids) == 3
    assert env.token_budget == 10000


def test_reset_initializes_state():
    """Test that reset creates proper episode state."""
    config = {
        "professor_ids": ["prof_1", "prof_2"],
        "students_per_batch": 5,
        "token_budget": 1000,
        "preference_weight": 0.5,
        "seed": 42,
    }
    env = AsyncTickerAdmissionsEnv(config)
    obs, info = env.reset()

    # Check episode state structure
    assert "episode_state" in info
    state = info["episode_state"]

    assert "agent_tickers" in state
    assert state["agent_tickers"]["prof_1"] == 0
    assert state["agent_tickers"]["prof_2"] == 0

    assert "message_history" in state
    assert state["message_history"] == []

    assert state["tokens_used"] == 0
    assert state["token_budget"] == 1000
    assert state["waiting_agents"] == {}
    assert state["consensus_reached"] is False
    assert state["consensus_choice"] is None

    # Check active agent is set
    assert info["active_agent"] in ["prof_1", "prof_2"]
    assert "agent_idx" in info


def test_parse_vote_action():
    """Test parsing vote format."""
    config = {"professor_ids": ["prof_1"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5}
    env = AsyncTickerAdmissionsEnv(config)

    parsed = env._parse_action("I think Student 3 is best. VOTE: 3")
    assert parsed["type"] == "vote"
    assert parsed["choice"] == 3
    assert "VOTE: 3" in parsed["text"]


def test_parse_wait_agent_action():
    """Test parsing wait for specific agent."""
    config = {"professor_ids": ["prof_1"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5}
    env = AsyncTickerAdmissionsEnv(config)

    parsed = env._parse_action("WAIT_FOR: prof_2")
    assert parsed["type"] == "wait"
    assert parsed["condition"] == "prof_2"


def test_parse_wait_any_action():
    """Test parsing wait for any response."""
    config = {"professor_ids": ["prof_1"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5}
    env = AsyncTickerAdmissionsEnv(config)

    parsed = env._parse_action("WAIT_FOR: any_response")
    assert parsed["type"] == "wait"
    assert parsed["condition"] == "any_response"


def test_parse_discuss_action():
    """Test parsing discussion (default)."""
    config = {"professor_ids": ["prof_1"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5}
    env = AsyncTickerAdmissionsEnv(config)

    parsed = env._parse_action("I think Student 5 has strong qualifications.")
    assert parsed["type"] == "discuss"
    assert "qualifications" in parsed["text"]


def test_count_tokens():
    """Test token counting (whitespace split)."""
    config = {"professor_ids": ["prof_1"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5}
    env = AsyncTickerAdmissionsEnv(config)

    # Simple whitespace tokenization for now
    assert env._count_tokens("Hello world") == 2
    assert env._count_tokens("I think Student 5 is best.") == 6
    assert env._count_tokens("VOTE: 3") == 2


def test_select_next_agent_min_ticker():
    """Test selecting agent with minimum ticker."""
    config = {"professor_ids": ["prof_1", "prof_2", "prof_3"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5}
    env = AsyncTickerAdmissionsEnv(config)
    env.reset()

    # Set different ticker values
    env.episode_state["agent_tickers"] = {
        "prof_1": 10,
        "prof_2": 5,
        "prof_3": 15
    }

    next_agent = env._select_next_agent()
    assert next_agent == "prof_2"  # Minimum ticker


def test_select_next_agent_tiebreak():
    """Test lexicographic tiebreak when tickers equal."""
    config = {"professor_ids": ["prof_3", "prof_1", "prof_2"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5}
    env = AsyncTickerAdmissionsEnv(config)
    env.reset()

    # All same ticker
    env.episode_state["agent_tickers"] = {
        "prof_3": 0,
        "prof_1": 0,
        "prof_2": 0
    }

    next_agent = env._select_next_agent()
    assert next_agent == "prof_1"  # Lexicographic first


def test_step_discussion_updates_ticker():
    """Test that discussion action updates ticker correctly."""
    config = {"professor_ids": ["prof_1", "prof_2"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    obs, info = env.reset()

    active_agent = info["active_agent"]
    initial_ticker = info["episode_state"]["agent_tickers"][active_agent]

    # Perform discussion action
    action = "I think Student 2 is strong"  # 6 tokens
    obs, reward, done, truncated, info = env.step(action)

    # Check ticker updated
    new_ticker = info["episode_state"]["agent_tickers"][active_agent]
    assert new_ticker == initial_ticker + 6

    # Check message added to history
    assert len(info["episode_state"]["message_history"]) == 1
    msg = info["episode_state"]["message_history"][0]
    assert msg["agent_id"] == active_agent
    assert msg["text"] == action
    assert msg["ticker_time"] == new_ticker
    assert msg["token_count"] == 6
    assert msg["message_type"] == "discuss"

    # Check tokens_used updated
    assert info["episode_state"]["tokens_used"] == 6

    # Check not done yet
    assert done is False
    assert reward == 0.0


def test_wait_condition_satisfied_any_response():
    """Test checking if wait condition for any_response is satisfied."""
    config = {"professor_ids": ["prof_1", "prof_2"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    env.reset()

    # Create wait info
    wait_info = {
        "condition_type": "any_response",
        "wait_issued_at": 10
    }

    # No messages yet
    assert env._is_wait_satisfied(wait_info) is False

    # Add message after wait issued
    env.episode_state["message_history"].append({
        "agent_id": "prof_2",
        "ticker_time": 15,
        "text": "Hello",
        "token_count": 1,
        "message_type": "discuss"
    })

    assert env._is_wait_satisfied(wait_info) is True


def test_wait_condition_satisfied_specific_agent():
    """Test checking if wait condition for specific agent is satisfied."""
    config = {"professor_ids": ["prof_1", "prof_2"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    env.reset()

    # Wait for prof_2
    wait_info = {
        "condition_type": "agent_specific",
        "target_agent": "prof_2",
        "wait_issued_at": 10
    }

    # Message from wrong agent
    env.episode_state["message_history"].append({
        "agent_id": "prof_1",
        "ticker_time": 15,
        "text": "Hello",
        "token_count": 1,
        "message_type": "discuss"
    })
    assert env._is_wait_satisfied(wait_info) is False

    # Message from correct agent
    env.episode_state["message_history"].append({
        "agent_id": "prof_2",
        "ticker_time": 20,
        "text": "Hi there",
        "token_count": 2,
        "message_type": "discuss"
    })
    assert env._is_wait_satisfied(wait_info) is True


def test_step_wait_action_marks_agent_waiting():
    """Test that wait action marks agent as waiting."""
    config = {"professor_ids": ["prof_1", "prof_2", "prof_3"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    obs, info = env.reset()

    active_agent = info["active_agent"]

    # Issue wait action
    action = "WAIT_FOR: prof_2"
    obs, reward, done, truncated, info = env.step(action)

    # Check agent marked as waiting
    assert active_agent in info["episode_state"]["waiting_agents"]
    wait_info = info["episode_state"]["waiting_agents"][active_agent]
    assert wait_info["condition_type"] == "agent_specific"
    assert wait_info["target_agent"] == "prof_2"

    # Check ticker jumped because wait not satisfied (min(others) + 1 = 0 + 1 = 1)
    assert info["episode_state"]["agent_tickers"][active_agent] == 1


def test_step_wait_any_response_marks_agent_waiting():
    """Test that wait for any_response marks agent as waiting."""
    config = {"professor_ids": ["prof_1", "prof_2"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    obs, info = env.reset()

    active_agent = info["active_agent"]

    # Issue wait for any_response action
    action = "WAIT_FOR: any_response"
    obs, reward, done, truncated, info = env.step(action)

    # Check agent marked as waiting
    assert active_agent in info["episode_state"]["waiting_agents"]
    wait_info = info["episode_state"]["waiting_agents"][active_agent]
    assert wait_info["condition_type"] == "any_response"
    assert "target_agent" not in wait_info

    # Check ticker jumped because wait not satisfied (min(others) + 1 = 0 + 1 = 1)
    assert info["episode_state"]["agent_tickers"][active_agent] == 1


def test_select_next_agent_skips_unsatisfied_wait():
    """Test that waiting agents with unsatisfied conditions get ticker jumped."""
    config = {"professor_ids": ["prof_1", "prof_2", "prof_3"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    env.reset()

    # Set tickers: prof_1 has min but is waiting
    env.episode_state["agent_tickers"] = {
        "prof_1": 5,
        "prof_2": 10,
        "prof_3": 15
    }

    # Mark prof_1 as waiting for prof_2
    env.episode_state["waiting_agents"]["prof_1"] = {
        "condition_type": "agent_specific",
        "target_agent": "prof_2",
        "wait_issued_at": 5
    }

    # No new messages, so condition not satisfied
    next_agent = env._select_next_agent()

    # prof_1's ticker should have jumped to min(others) + 1 = 11
    assert env.episode_state["agent_tickers"]["prof_1"] == 11

    # Next agent should be prof_2 (ticker=10, now minimum)
    assert next_agent == "prof_2"


def test_select_next_agent_satisfied_wait_removes_from_waiting():
    """Test that satisfied wait conditions remove agent from waiting."""
    config = {"professor_ids": ["prof_1", "prof_2"], "students_per_batch": 5, "token_budget": 1000, "preference_weight": 0.5, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    env.reset()

    env.episode_state["agent_tickers"] = {"prof_1": 5, "prof_2": 10}
    env.episode_state["waiting_agents"]["prof_1"] = {
        "condition_type": "agent_specific",
        "target_agent": "prof_2",
        "wait_issued_at": 5
    }

    # Add message from prof_2 that satisfies wait
    env.episode_state["message_history"].append({
        "agent_id": "prof_2",
        "ticker_time": 10,
        "text": "Hello",
        "token_count": 1,
        "message_type": "discuss"
    })

    next_agent = env._select_next_agent()

    # prof_1 should be removed from waiting
    assert "prof_1" not in env.episode_state["waiting_agents"]

    # prof_1 should be selected (ticker=5 < 10)
    assert next_agent == "prof_1"


def test_check_consensus_consecutive_votes():
    """Test consensus detection with threshold voting (all professors vote for same student)."""
    config = {"professor_ids": ["prof_1", "prof_2", "prof_3"], "students_per_batch": 5, "token_budget": 1000, "vote_threshold": 0.5, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    env.reset()

    # Simulate votes (threshold voting needs votes dict)
    env.episode_state["votes"] = {
        "prof_1": 2,
        "prof_2": 2,
        "prof_3": 2,
    }

    consensus, choice = env._check_consensus()
    assert consensus is True
    assert choice == 2


def test_check_consensus_discussion_resets():
    """Test that insufficient votes (below threshold) don't trigger consensus."""
    config = {"professor_ids": ["prof_1", "prof_2", "prof_3"], "students_per_batch": 5, "token_budget": 1000, "vote_threshold": 0.67, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    env.reset()

    # Only 2 out of 3 professors vote for same student (below 67% threshold)
    env.episode_state["votes"] = {
        "prof_1": 2,
        "prof_2": 2,
    }

    consensus, choice = env._check_consensus()
    assert consensus is False


def test_step_detects_early_consensus():
    """Test that step function detects consensus when threshold is reached."""
    config = {"professor_ids": ["prof_1", "prof_2", "prof_3"], "students_per_batch": 5, "token_budget": 10000, "vote_threshold": 0.67, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    obs, info = env.reset()

    # All agents vote for same student (reaches threshold)
    for i in range(3):
        obs, reward, done, truncated, info = env.step("VOTE: 2")

    # Should be done with consensus (3/3 = 100% >= 67%)
    assert done is True
    assert info["episode_state"]["consensus_reached"] is True
    assert info["episode_state"]["consensus_choice"] == 2


def test_calculate_rewards_no_consensus():
    """Test that no consensus results in 0 rewards for all."""
    config = {"professor_ids": ["prof_1", "prof_2"], "students_per_batch": 5, "token_budget": 100, "preference_weight": 0.5, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    env.reset()

    env.episode_state["consensus_reached"] = False

    rewards = env._calculate_rewards()
    assert rewards["prof_1"] == 0.0
    assert rewards["prof_2"] == 0.0


def test_calculate_rewards_with_consensus():
    """Test reward calculation with consensus using direct utility."""
    config = {"professor_ids": ["prof_1", "prof_2"], "students_per_batch": 3, "token_budget": 1000, "feature_dim": 3, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    env.reset()

    # Set interests and student profiles manually for predictability
    # Using normalized vectors
    env.professor_interests = {
        "prof_1": np.array([1.0, 0.0, 0.0]),  # Aligned with first dimension
        "prof_2": np.array([0.0, 1.0, 0.0]),  # Aligned with second dimension
    }

    # Student 1 has strong first dimension alignment
    env.student_batch[1]["profile_vector"] = [0.9, 0.1, 0.0]

    # Consensus on student 1
    env.episode_state["consensus_reached"] = True
    env.episode_state["consensus_choice"] = 1

    rewards = env._calculate_rewards()

    # prof_1 should have higher utility (dot product closer to 1.0)
    # prof_1: 1.0*0.9 + 0.0*0.1 + 0.0*0.0 + 1.0 = 1.9
    # prof_2: 0.0*0.9 + 1.0*0.1 + 0.0*0.0 + 1.0 = 1.1
    assert rewards["prof_1"] > rewards["prof_2"]
    assert rewards["prof_1"] == pytest.approx(1.9)
    assert rewards["prof_2"] == pytest.approx(1.1)


def test_step_returns_rewards_when_done():
    """Test that rewards are calculated and returned when episode ends."""
    config = {"professor_ids": ["prof_1", "prof_2", "prof_3"], "students_per_batch": 5, "token_budget": 10000, "vote_threshold": 0.5, "feature_dim": 3, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    env.reset()

    # Set predictable interests - all aligned with first dimension but different magnitudes
    env.professor_interests = {
        "prof_1": np.array([1.0, 0.0, 0.0]),
        "prof_2": np.array([0.8, 0.6, 0.0]) / np.linalg.norm([0.8, 0.6, 0.0]),
        "prof_3": np.array([0.6, 0.8, 0.0]) / np.linalg.norm([0.6, 0.8, 0.0]),
    }

    # Student 1 strongly aligned with first dimension
    env.student_batch[1]["profile_vector"] = [1.0, 0.0, 0.0]

    # Get to consensus
    for i in range(3):
        obs, reward, done, truncated, info = env.step("VOTE: 1")

    # Check rewards calculated
    assert done is True
    rewards = info["agent_rewards"]
    assert rewards["prof_1"] > 0
    assert rewards["prof_2"] > 0
    assert rewards["prof_3"] > 0

    # prof_1 should have highest reward (best alignment with student 1)
    assert rewards["prof_1"] > rewards["prof_2"]
    assert rewards["prof_2"] > rewards["prof_3"]


def test_build_observation_chronological_order():
    """Test that observations show messages in chronological ticker order."""
    config = {"professor_ids": ["prof_1", "prof_2"], "students_per_batch": 3, "token_budget": 1000, "preference_weight": 0.5, "seed": 42}
    env = AsyncTickerAdmissionsEnv(config)
    env.reset()

    # Add messages out of generation order but with ticker times
    env.episode_state["message_history"] = [
        {"agent_id": "prof_1", "ticker_time": 5, "text": "First", "token_count": 1, "message_type": "discuss"},
        {"agent_id": "prof_2", "ticker_time": 20, "text": "Third", "token_count": 1, "message_type": "discuss"},
        {"agent_id": "prof_1", "ticker_time": 10, "text": "Second", "token_count": 1, "message_type": "discuss"},
    ]

    obs = env._build_observation("prof_2")

    # Should be ordered by ticker_time
    assert "First" in obs
    assert "Second" in obs
    assert "Third" in obs

    # Check chronological ordering (First before Second before Third)
    first_pos = obs.index("First")
    second_pos = obs.index("Second")
    third_pos = obs.index("Third")
    assert first_pos < second_pos < third_pos


def test_full_episode_with_consensus():
    """Integration test: Full episode with discussion, waiting, and consensus."""
    config = {
        "professor_ids": ["prof_1", "prof_2", "prof_3"],
        "students_per_batch": 5,
        "token_budget": 10000,
        "vote_threshold": 0.67,
        "seed": 42
    }
    env = AsyncTickerAdmissionsEnv(config)
    obs, info = env.reset()

    assert info["active_agent"] is not None
    assert not info["episode_state"]["consensus_reached"]

    # Agent 1 discusses
    obs, reward, done, truncated, info = env.step("I think Student 2 is strong")
    assert not done
    assert reward == 0.0

    # Agent 2 discusses
    obs, reward, done, truncated, info = env.step("I agree Student 2 looks good")
    assert not done

    # Agent 3 waits for more input
    obs, reward, done, truncated, info = env.step("WAIT_FOR: any_response")
    assert "prof_3" in info["episode_state"]["waiting_agents"]

    # Agent 1 votes
    obs, reward, done, truncated, info = env.step("VOTE: 2")
    assert not done

    # Agent 2 votes
    obs, reward, done, truncated, info = env.step("VOTE: 2")
    assert not done

    # Agent 3 should be unblocked now, can vote
    obs, reward, done, truncated, info = env.step("VOTE: 2")

    # Should reach consensus
    assert done is True
    assert info["episode_state"]["consensus_reached"] is True
    assert info["episode_state"]["consensus_choice"] == 2

    # All agents should have rewards
    assert all(r > 0 for r in info["agent_rewards"].values())


def test_full_episode_budget_exhausted():
    """Integration test: Episode ends when budget exhausted without consensus."""
    config = {
        "professor_ids": ["prof_1", "prof_2"],
        "students_per_batch": 3,
        "token_budget": 20,  # Very small budget
        "vote_threshold": 0.5,
        "seed": 42
    }
    env = AsyncTickerAdmissionsEnv(config)
    obs, info = env.reset()

    # Keep discussing until budget runs out
    done = False
    steps = 0
    while not done and steps < 100:  # Safety limit
        obs, reward, done, truncated, info = env.step("I think we should discuss more options here")
        steps += 1

    # Should be done due to budget
    assert done is True
    assert info["episode_state"]["tokens_used"] >= config["token_budget"]

    # No consensus, so all rewards should be 0
    assert all(r == 0.0 for r in info["agent_rewards"].values())
