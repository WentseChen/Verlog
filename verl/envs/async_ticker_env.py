# verl/envs/async_ticker_env.py
import gym
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import re


class AsyncTickerAdmissionsEnv(gym.Env):
    """
    Async ticker-based multi-agent negotiation environment.

    Agents take turns based on ticker values (accumulated token counts).
    Supports wait actions, voting, and consensus detection.
    """

    def __init__(self, config: Dict[str, Any], tokenizer=None):
        super().__init__()

        # Configuration
        self.professor_ids = config["professor_ids"]
        self.students_per_batch = config["students_per_batch"]
        self.token_budget = config["token_budget"]
        self.feature_dim = config.get("feature_dim", 5)
        self.vote_threshold = config.get("vote_threshold", 0.5)
        self.seed_value = config.get("seed", None)

        # System prompt configuration
        # Can be a single string (used for all agents) or dict mapping agent_id -> prompt
        self.system_prompt = config.get("system_prompt", None)
        self.tokenizer = tokenizer

        # Cache system prompt token lengths
        self.system_prompt_token_lengths = {}
        if self.system_prompt and self.tokenizer:
            if isinstance(self.system_prompt, str):
                # Single prompt for all agents
                tokens = self.tokenizer.encode(self.system_prompt, add_special_tokens=False)
                token_length = len(tokens)
                for agent_id in self.professor_ids:
                    self.system_prompt_token_lengths[agent_id] = token_length
            elif isinstance(self.system_prompt, dict):
                # Per-agent prompts
                for agent_id in self.professor_ids:
                    prompt = self.system_prompt.get(agent_id, "")
                    if prompt:
                        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                        self.system_prompt_token_lengths[agent_id] = len(tokens)

        # Multi-agent support: possible_agents attribute
        self.possible_agents = self.professor_ids

        # Episode state (initialized in reset)
        self.episode_state = None
        self.student_batch = None
        self.professor_interests = None

        # Set random seed
        if self.seed_value is not None:
            np.random.seed(self.seed_value)

    def get_system_prompt(self, agent_id: str) -> Optional[str]:
        """Get system prompt for a specific agent."""
        if self.system_prompt is None:
            return None
        if isinstance(self.system_prompt, str):
            return self.system_prompt
        return self.system_prompt.get(agent_id, None)

    def reset(self) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """Reset environment and return dict observations and infos for all agents."""
        # Initialize episode state
        self.episode_state = {
            "agent_tickers": {agent_id: 0 for agent_id in self.professor_ids},
            "message_history": [],
            "tokens_used": 0,
            "token_budget": self.token_budget,
            "waiting_agents": {},
            "consensus_reached": False,
            "consensus_choice": None,
            "active_agent": None,
            "votes": {},  # Track votes: agent_id -> student_index
        }

        # Generate professor interest vectors (normalized Gaussian)
        self.professor_interests = {}
        for prof_id in self.professor_ids:
            interest_vector = np.random.randn(self.feature_dim)
            interest_vector = interest_vector / np.linalg.norm(interest_vector)
            self.professor_interests[prof_id] = interest_vector

        # Generate student batch with normalized Gaussian profile vectors
        self.student_batch = []
        for i in range(self.students_per_batch):
            profile_vector = np.random.randn(self.feature_dim)
            profile_vector = profile_vector / np.linalg.norm(profile_vector)

            self.student_batch.append({
                "index": i,
                "id": f"student_{i}",
                "name": f"Student {i}",
                "profile_vector": profile_vector.tolist(),
            })

        # Select first agent (lexicographic order when all at 0)
        active_agent = min(self.professor_ids)
        self.episode_state["active_agent"] = active_agent

        # Build observation for active agent
        obs_text = self._build_observation(active_agent)

        # Build observations dict (only active agent gets observation)
        # Format: {"prompt": obs_text} for VERL compatibility
        observations = {
            agent_id: {"prompt": obs_text} if agent_id == active_agent else {"prompt": ""}
            for agent_id in self.professor_ids
        }

        # Build info dict for all agents
        base_info = {
            "active_agent": active_agent,
            "agent_idx": self.professor_ids.index(active_agent),
            "episode_state": self.episode_state.copy(),
            "agent_rewards": {agent_id: 0.0 for agent_id in self.professor_ids},
            "student_batch": self.student_batch,
            "professor_interests": self.professor_interests,
        }

        # Add system prompt info if available
        if self.system_prompt:
            base_info["system_prompt"] = {
                agent_id: self.get_system_prompt(agent_id) for agent_id in self.professor_ids
            }
            if self.system_prompt_token_lengths:
                base_info["system_prompt_token_length"] = self.system_prompt_token_lengths

        infos = {agent_id: base_info.copy() for agent_id in self.professor_ids}

        return observations, infos

    def step(self, action: str) -> Tuple[Dict[str, Dict], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict]]:
        """
        Execute one step with active agent's action.

        Returns:
            observations: Dict mapping agent_id to observation dict {"prompt": obs_text}
            rewards: Dict mapping agent_id to reward value
            terminations: Dict mapping agent_id to termination bool
            truncations: Dict mapping agent_id to truncation bool (always False)
            infos: Dict mapping agent_id to info dict
        """
        active_agent = self.episode_state["active_agent"]

        # Parse action
        parsed = self._parse_action(action)

        # Count tokens
        token_count = self._count_tokens(action)

        # Update active agent's ticker
        old_ticker = self.episode_state["agent_tickers"][active_agent]
        new_ticker = old_ticker + token_count
        self.episode_state["agent_tickers"][active_agent] = new_ticker

        # Add message to history
        message = {
            "agent_id": active_agent,
            "text": action,
            "ticker_time": new_ticker,
            "token_count": token_count,
            "message_type": parsed["type"]
        }

        # Store vote choice if applicable
        if parsed["type"] == "vote":
            message["choice"] = parsed["choice"]
            # Track vote in episode state for threshold voting
            self.episode_state["votes"][active_agent] = parsed["choice"]

        self.episode_state["message_history"].append(message)

        # Handle wait actions
        if parsed["type"] == "wait":
            condition = parsed["condition"]

            # Determine condition type
            if condition == "any_response":
                condition_type = "any_response"
                wait_info = {
                    "condition_type": condition_type,
                    "wait_issued_at": new_ticker
                }
            else:
                # Specific agent
                condition_type = "agent_specific"
                wait_info = {
                    "condition_type": condition_type,
                    "target_agent": condition,
                    "wait_issued_at": new_ticker
                }

            self.episode_state["waiting_agents"][active_agent] = wait_info

        # Update total tokens used
        self.episode_state["tokens_used"] += token_count

        # Check for consensus
        consensus_reached, consensus_choice = self._check_consensus()
        if consensus_reached:
            self.episode_state["consensus_reached"] = True
            self.episode_state["consensus_choice"] = consensus_choice
            done = True
        else:
            # Check if budget exhausted
            done = self.episode_state["tokens_used"] >= self.token_budget

        # Select next agent
        next_agent = self._select_next_agent()
        self.episode_state["active_agent"] = next_agent

        # Build observation for next agent
        obs_text = self._build_observation(next_agent)

        # Calculate rewards if episode done
        if done:
            agent_rewards = self._calculate_rewards()
        else:
            agent_rewards = {agent_id: 0.0 for agent_id in self.professor_ids}

        # Build base info dict
        base_info = {
            "active_agent": next_agent,
            "agent_idx": self.professor_ids.index(next_agent),
            "episode_state": self.episode_state.copy(),
            "agent_rewards": agent_rewards,  # Use calculated rewards
            "student_batch": self.student_batch,
            "professor_interests": self.professor_interests,
        }

        # Add system prompt info if available
        if self.system_prompt:
            base_info["system_prompt"] = {
                agent_id: self.get_system_prompt(agent_id) for agent_id in self.professor_ids
            }
            if self.system_prompt_token_lengths:
                base_info["system_prompt_token_length"] = self.system_prompt_token_lengths

        # Build multi-agent return dicts
        # Format: {"prompt": obs_text} for VERL compatibility
        observations = {
            agent_id: {"prompt": obs_text} if agent_id == next_agent else {"prompt": ""}
            for agent_id in self.professor_ids
        }
        rewards = {agent_id: agent_rewards[agent_id] for agent_id in self.professor_ids}
        terminations = {agent_id: done for agent_id in self.professor_ids}
        truncations = {agent_id: False for agent_id in self.professor_ids}
        infos = {agent_id: base_info.copy() for agent_id in self.professor_ids}

        return observations, rewards, terminations, truncations, infos

    def _build_observation(self, agent_id: str) -> str:
        """
        Build observation string for agent.

        Shows messages in chronological ticker order.
        Includes game context: students with utilities, budget.
        """
        current_ticker = self.episode_state["agent_tickers"][agent_id]

        # Get visible messages (all generated so far)
        visible_messages = self.episode_state["message_history"].copy()

        # Sort by ticker_time (chronological order)
        visible_messages.sort(key=lambda m: m["ticker_time"])

        # Format conversation
        conversation_lines = []
        for msg in visible_messages:
            conversation_lines.append(
                f"[{msg['agent_id']} at t={msg['ticker_time']}]: {msg['text']}"
            )

        conversation_text = "\n".join(conversation_lines) if conversation_lines else "(No messages yet)"

        # Calculate utilities for this professor
        student_utilities = []
        for student in self.student_batch:
            utility = self._calculate_utility_for_student(
                self.professor_interests[agent_id],
                student["profile_vector"]
            )
            student_utilities.append({
                "index": student["index"],
                "name": student["name"],
                "utility": utility
            })

        # Format student list with utilities
        student_lines = []
        for su in student_utilities:
            student_lines.append(
                f"  Student {su['index']}: {su['name']} (utility={su['utility']:.3f})"
            )
        students_text = "\n".join(student_lines)

        # Build full observation
        obs = f"""=== YOUR TURN (t={current_ticker}) ===

STUDENTS (Your utility if selected):
{students_text}

Your interest vector: {self.professor_interests[agent_id].tolist()}
Token budget: {self.episode_state['tokens_used']}/{self.token_budget}

CONVERSATION HISTORY (chronological by ticker time):
{conversation_text}

Your turn:"""

        return obs

    def _parse_action(self, action_text: str) -> Dict[str, Any]:
        """
        Parse action text to determine type and extract data.

        Priority: Vote > Wait > Discuss
        """
        # Check for vote format: "VOTE: <number>"
        vote_match = re.search(r'VOTE:\s*(\d+)', action_text)
        if vote_match:
            return {
                'type': 'vote',
                'choice': int(vote_match.group(1)),
                'text': action_text
            }

        # Check for wait format: "WAIT_FOR: <agent_id or any_response>"
        wait_match = re.search(r'WAIT_FOR:\s*(\w+)', action_text)
        if wait_match:
            return {
                'type': 'wait',
                'condition': wait_match.group(1),
                'text': action_text
            }

        # Default to discussion
        return {
            'type': 'discuss',
            'text': action_text
        }

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Uses simple whitespace splitting for now.
        Can be replaced with actual tokenizer later.
        """
        return len(text.split())

    def _select_next_agent(self) -> str:
        """
        Select next agent to speak.

        Handles waiting agents:
        - If wait condition satisfied: remove from waiting, make eligible
        - If wait condition not satisfied: jump ticker to min(others) + 1

        Returns agent with minimum ticker value.
        Tiebreak: lexicographic order.
        """
        candidates = []

        for agent_id, ticker in self.episode_state["agent_tickers"].items():
            if agent_id in self.episode_state["waiting_agents"]:
                # Check if wait condition satisfied
                wait_info = self.episode_state["waiting_agents"][agent_id]

                if self._is_wait_satisfied(wait_info):
                    # Condition met - remove from waiting and make candidate
                    del self.episode_state["waiting_agents"][agent_id]
                    candidates.append((ticker, agent_id))
                else:
                    # Still waiting - jump ticker ahead
                    other_tickers = [
                        t for aid, t in self.episode_state["agent_tickers"].items()
                        if aid != agent_id
                    ]
                    if other_tickers:
                        min_other_ticker = min(other_tickers)
                        self.episode_state["agent_tickers"][agent_id] = min_other_ticker + 1
                    # Don't add to candidates this round
            else:
                # Not waiting - normal candidate
                candidates.append((ticker, agent_id))

        # Sort by (ticker, agent_id) for min ticker + lexicographic tiebreak
        candidates.sort()
        return candidates[0][1]

    def _is_wait_satisfied(self, wait_info: Dict[str, Any]) -> bool:
        """
        Check if wait condition has been met.

        Args:
            wait_info: Dict with condition_type, target_agent (optional), wait_issued_at

        Returns:
            True if condition satisfied, False otherwise
        """
        if not self.episode_state["message_history"]:
            return False

        last_message = self.episode_state["message_history"][-1]

        if wait_info["condition_type"] == "any_response":
            # Any message posted after wait was issued
            return last_message["ticker_time"] > wait_info["wait_issued_at"]

        elif wait_info["condition_type"] == "agent_specific":
            # Specific agent posted after wait was issued
            return (last_message["agent_id"] == wait_info["target_agent"] and
                    last_message["ticker_time"] > wait_info["wait_issued_at"])

        return False

    def _check_consensus(self) -> Tuple[bool, Optional[int]]:
        """
        Check if consensus has been reached via threshold voting.

        Counts votes collected in episode_state["votes"].
        Consensus requires vote_threshold fraction of professors to vote for same student.

        Returns:
            (consensus_reached, choice) tuple
        """
        if not self.episode_state["votes"]:
            return False, None

        # Count votes for each student index
        from collections import Counter
        vote_counts = Counter(self.episode_state["votes"].values())

        # Check if any student has enough votes
        n_professors = len(self.professor_ids)
        for student_index, count in vote_counts.items():
            vote_fraction = count / n_professors
            if vote_fraction >= self.vote_threshold:
                return True, student_index

        return False, None

    def _calculate_utility_for_student(
        self,
        interest_vector: np.ndarray,
        profile_vector: List[float]
    ) -> float:
        """
        Calculate utility from selecting a student.

        Uses cosine similarity (dot product of normalized vectors) + 1.
        Returns value in [0, 2]:
        - 2.0 = perfect alignment
        - 1.0 = orthogonal interests
        - 0.0 = opposite interests

        Args:
            interest_vector: Professor's normalized interest vector
            profile_vector: Student's normalized profile vector

        Returns:
            Utility value in range [0, 2]
        """
        return float(np.dot(interest_vector, profile_vector) + 1.0)

    def _calculate_rewards(self) -> Dict[str, float]:
        """
        Calculate per-agent rewards at episode end.

        No consensus: All get 0.0
        Consensus: Direct utility based on alignment with selected student

        Utility = dot_product(professor_interest, student_profile) + 1.0
        Range: [0, 2] where:
            - 2.0 = perfect alignment
            - 1.0 = orthogonal interests
            - 0.0 = opposite interests

        Returns:
            Dict mapping agent_id to reward value
        """
        if not self.episode_state["consensus_reached"]:
            return {agent_id: 0.0 for agent_id in self.professor_ids}

        consensus_choice = self.episode_state["consensus_choice"]
        selected_student = self.student_batch[consensus_choice]

        # Calculate utility-based rewards for all professors
        agent_rewards = {}
        for agent_id in self.professor_ids:
            utility = self._calculate_utility_for_student(
                self.professor_interests[agent_id],
                selected_student["profile_vector"]
            )
            agent_rewards[agent_id] = utility

        return agent_rewards
