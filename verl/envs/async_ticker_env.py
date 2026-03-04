# verl/envs/async_ticker_env.py
import gym
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import re
from copy import deepcopy

# from history_manager import HistoryManager
# from time_manager import TimeManager

from verl.envs.history_manager import HistoryManager
from verl.envs.time_manager import TimeManager


class AsyncTickerAdmissionsEnv(gym.Env):
    """
    Async ticker-based multi-agent negotiation environment.

    Agents take turns based on ticker values (accumulated token counts).
    Supports wait actions, voting, and consensus detection.

    Scoring:
    - Student ability vectors are N-dimensional, values sum to 1 (public info).
    - Professor preference vectors are N-dimensional integers in {0,1,2,3,4} (private).
    - Utility = dot product of professor preference and student ability vectors.

    Observations returned from reset() and step() are OpenAI-compatible chat
    message lists:
        [
            {"role": "system", "content": "<system prompt>"},
            {"role": "user",   "content": "<current game state / turn prompt>"},
        ]
    This allows the caller to pass `observations` directly as the `messages`
    argument to openai.ChatCompletion.create (or any compatible client).

    Think-before-act protocol:
    - Every turn MUST start with <THINK>...</THINK> followed by an action tag.
    - <THINK> tokens count toward the agent's personal ticker (turn order) but
      NOT toward the shared token budget.
    - <THINK> content is private: each agent sees only their own thoughts in
      their conversation history; other agents' <THINK> blocks are hidden.
    - <GROUP> tokens count toward both ticker and the shared budget.
    - Combined format: <THINK>reasoning</THINK><GROUP>message</GROUP>
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
                tokens = self.tokenizer.encode(self.system_prompt, add_special_tokens=False)
                token_length = len(tokens)
                for agent_id in self.professor_ids:
                    self.system_prompt_token_lengths[agent_id] = token_length
            elif isinstance(self.system_prompt, dict):
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

        # Managers for history and time/ticker scheduling (initialized in reset)
        self.history_manager: Optional[HistoryManager] = None
        self.time_manager: Optional[TimeManager] = None

        # Build a lowercase lookup map for fast agent_name resolution
        # Maps lower-stripped name -> canonical professor_id
        self._professor_id_lookup: Dict[str, str] = {
            pid.strip().lower(): pid for pid in self.professor_ids
        }

        # Set random seed
        if self.seed_value is not None:
            np.random.seed(self.seed_value)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def build_system_prompt(self, professor_id: str | None, preference_vector: List[int] | None = None) -> str:
        """Build the system prompt used by the LLM agent."""
        topic_order = "AI/ML, Systems, Theory, HCI, Computational Biology"
        if professor_id is None:
            professor_id = "unknown_professor"

        pref_str = (
            f"{preference_vector}"
            if preference_vector is not None
            else "(not yet assigned)"
        )
        
        other_professors_str = ", ".join([p for p in self.professor_ids if p != professor_id])

        return (
            f"You are Professor {professor_id}, a strategic negotiator participating in an "
            f"admissions committee. Your goal is to maximize your personal utility while "
            f"achieving consensus with other professors.\n\n"
            f"GAME RULES:\n"
            f"- Each round, you review a batch of {self.students_per_batch} students.\n"
            f"- You deliberate with other professors to select one student.\n"
            f"- named {other_professors_str} are the other professors."
            f"- Voting requires {int(self.vote_threshold * 100)}% agreement for consensus. "
            f"If consensus is reached, you receive utility based on how well the selected "
            f"student matches your research preferences. If NO consensus is reached, EVERYONE "
            f"gets 0 utility.\n"
            f"- Student ability vectors are {self.feature_dim}-dimensional; each entry is a "
            f"non-negative value. They represent how much of each "
            f"research topic the student specialises in. These vectors are PUBLIC — all professors "
            f"can see them.\n"
            f"- Professor preference vectors are a permutation of {{0, 1, …, {self.feature_dim - 1}}}. "
            f"Each priority level (0 = lowest, {self.feature_dim - 1} = highest) appears exactly once, "
            f"so you have a strict ranking over the {self.feature_dim} topics. These are PRIVATE "
            f"— only you know your own preference vector.\n"
            f"- Topic order (dimensions 0–{self.feature_dim - 1}): {topic_order}.\n"
            f"- Utility = dot product of YOUR preference vector and the selected student's "
            f"ability vector. Higher preference values on topics where a student scores highly "
            f"gives more utility.\n"
            f"- All professors share a single token budget of {self.token_budget} total tokens "
            f"(only <GROUP> message tokens consume this budget).\n\n"
            f"YOUR PRIVATE PREFERENCE VECTOR (keep this secret):\n"
            f"  {pref_str}\n"
            f"  (dimensions correspond to: {topic_order})\n\n"
            f"ACTION RULES (CRITICAL):\n"
            f"Every turn MUST begin with <THINK>...</THINK> reasoning, followed by EXACTLY ONE action tag.\n"
            f"Required format: <THINK>your private reasoning</THINK> followed by one of:\n"
            f"1. <GROUP>your message</GROUP> - Send a message visible to all (uses token budget)\n"
            f"2. <WAIT_FOR>prof_name</WAIT_FOR> - Wait for another professor to respond\n"
            f"3. <VOTE>student_index</VOTE> - Cast your vote for a student\n\n"
            f"ALL VAILD ACTIONS: (no two group, wait, or vote are allowed in one turn)\n"
            f"1. <THINK>your reasoning</THINK><GROUP>your message</GROUP>\n"
            f"2. <THINK>your reasoning</THINK><WAIT_FOR>prof_name</WAIT_FOR>\n"
            f"3. <THINK>your reasoning</THINK><VOTE>student_index</VOTE>\n\n"
            f"TOKEN BUDGET RULES:\n"
            f"- <THINK> tokens do NOT count against the shared token budget.\n"
            f"- <THINK> tokens DO count against your personal ticker time (turn order).\n"
            f"- Only <GROUP> message tokens consume the shared budget.\n\n"
            f"VISIBILITY RULES:\n"
            f"- <THINK> content is PRIVATE — only you can see your own thoughts in your history.\n"
            f"- <GROUP> content is PUBLIC — all professors see these messages.\n"
            f"- Other professors' <THINK> blocks are NEVER shown to you.\n\n"
            f"STRICT FORMATTING RESTRICTION: You MUST always output <THINK>...</THINK> before your action. "
            f"Plain text or responses missing <THINK> will be treated as discuss-type actions. "
            f"Never add text outside the tags.\n\n"
            f"STRATEGIC CONSIDERATIONS:\n"
            f"- Your preference vector is private; DO NOT reveal it explicitly to others.\n"
            f"- Instead, describe your opinions qualitatively (e.g., 'I find this student very strong in AI/ML and Systems').\n"
            f"- You must balance self-interest with compromise to avoid 0 utility.\n"
            f"- The student ability vectors are public, so focus your negotiation on which student best serves the group's combined interests.\n"
            f"- Use <THINK> freely for deep reasoning — it doesn't cost budget, so think carefully before speaking.\n"
        )

    def get_system_prompt(self, agent_id: str) -> str | None:
        """Get system prompt for a specific agent."""
        if self.system_prompt is None:
            return None
        if isinstance(self.system_prompt, str):
            return self.system_prompt
        return self.system_prompt.get(agent_id, None)

    # ------------------------------------------------------------------
    # OpenAI message list builder
    # ------------------------------------------------------------------

    def _build_chat_messages(self, agent_id: str) -> List[Dict[str, str]]:
        """
        Return an OpenAI-compatible chat messages list for *agent_id*.

        Structure:
            [
                {"role": "system", "content": "<system prompt>"},
                {"role": "user",   "content": "<observation>"},
            ]

        If a custom system_prompt is configured for this agent it is used;
        otherwise the default build_system_prompt() is called (with the
        agent's private preference vector embedded).
        """
        sys_prompt = self.get_system_prompt(agent_id)
        if sys_prompt is None:
            # Embed the private preference vector into the system prompt
            pref_vector = (
                self.professor_interests[agent_id].tolist()
                if self.professor_interests and agent_id in self.professor_interests
                else None
            )
            sys_prompt = self.build_system_prompt(agent_id, preference_vector=pref_vector)

        # Build user observation content via HistoryManager
        if self.time_manager is not None:
            current_ticker = self.time_manager.agent_tickers.get(agent_id, 0)
        else:
            current_ticker = 0

        tokens_used = 0
        if self.episode_state is not None:
            tokens_used = self.episode_state.get("tokens_used", 0)

        if self.history_manager is None:
            # Fallback for safety; should not happen in normal usage.
            raise RuntimeError("HistoryManager is not initialized.")

        professor_interest_vec = self.professor_interests[agent_id]

        user_content = self.history_manager.get_obs(
            agent_id=agent_id,
            current_ticker=current_ticker,
            token_budget_used=tokens_used,
            token_budget=self.token_budget,
            student_batch=self.student_batch,
            professor_interest_vector=professor_interest_vec,
            feature_dim=self.feature_dim,
        )

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(self, agent_id: str | None = None) -> Tuple[List[Dict[str, str]], Dict[str, Dict]]:
        """
        Reset environment and return observations and infos.

        Returns
        -------
        observations : list[dict]
            OpenAI chat messages list for the first active agent.
        infos : dict[str, dict]
            Per-agent info dicts (keyed by agent_id).
        """
        # Generate professor preference vectors: a random permutation of [0, 1, ..., feature_dim-1]
        # Each value appears exactly once, so no two topics share the same priority level.
        self.professor_interests = {}
        for prof_id in self.professor_ids:
            preference_vector = np.random.permutation(self.feature_dim)
            self.professor_interests[prof_id] = preference_vector

        # Generate student batch: ability vectors on the probability simplex
        # (non-negative, sum to 1) via Dirichlet distribution
        self.student_batch = []
        for i in range(self.students_per_batch):
            profile_vector = self._sample_discrete_simplex(self.feature_dim)
            self.student_batch.append({
                "index": i,
                "id": f"student_{i}",
                "name": f"Student {i}",
                "profile_vector": profile_vector,
            })

        # Initialize managers
        self.history_manager = HistoryManager(self.professor_ids)
        self.time_manager = TimeManager(self.professor_ids)

        # Initialize episode state
        self.episode_state = {
            "agent_tickers": self.time_manager.get_public_tickers(),
            "message_history": self.history_manager.get_episode_message_history_snapshot(),
            "tokens_used": 0,
            "token_budget": self.token_budget,
            "waiting_agents": self.time_manager.get_public_waiting_state(),
            "consensus_reached": False,
            "consensus_choice": None,
            "active_agent": None,
            "votes": {},
        }

        # Select first agent (lexicographic order when all at 0) unless an explicit
        # starting agent is provided.
        if agent_id is not None and agent_id in self.professor_ids:
            active_agent = agent_id
        else:
            active_agent = min(self.professor_ids)
        self.episode_state["active_agent"] = active_agent

        # Build OpenAI-compatible observations for the active agent
        observations = self._build_chat_messages(active_agent)

        # Build info dict
        base_info = {
            "active_agent": active_agent,
            "agent_idx": self.professor_ids.index(active_agent),
            "episode_state": deepcopy(self.episode_state),
            "agent_rewards": {agent_id: 0.0 for agent_id in self.professor_ids},
            "student_batch": self.student_batch,
            "professor_interests": self.professor_interests,
        }

        if self.system_prompt:
            base_info["system_prompt"] = {
                agent_id: self.get_system_prompt(agent_id) for agent_id in self.professor_ids
            }
            if self.system_prompt_token_lengths:
                base_info["system_prompt_token_length"] = self.system_prompt_token_lengths

        infos = {agent_id: deepcopy(base_info) for agent_id in self.professor_ids}

        return observations, infos

    def step(
        self, action: str | Dict[str, Any]
    ) -> Tuple[
        List[Dict[str, str]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
        """
        Execute one step with the active agent's action.

        Parameters
        ----------
        action : str | dict
            The action produced by the active agent. Expected format:
            <THINK>private reasoning</THINK><GROUP>public message</GROUP>
            (or <THINK>...</THINK><VOTE>N</VOTE> / <THINK>...</THINK><WAIT_FOR>name</WAIT_FOR>)

        Returns
        -------
        observations : list[dict]
            OpenAI chat messages list for the *next* active agent.
        rewards : dict[str, float]
        terminations : dict[str, bool]
        truncations : dict[str, bool]
        infos : dict[str, dict]
        """
        active_agent = self.episode_state["active_agent"]

        # Normalise action to a string
        action_text = action
        if isinstance(action, dict):
            if active_agent in action:
                action_text = action.get(active_agent, "")
            elif len(action) == 1:
                action_text = next(iter(action.values()))
            else:
                action_text = ""
        if action_text is None:
            action_text = ""
        if not isinstance(action_text, str):
            action_text = str(action_text)

        # Parse action — splits into think_text and action payload
        parsed = self._parse_action(action_text)

        # Count tokens separately for think and action portions.
        # Think tokens → ticker only (not budget).
        # Action tokens → ticker + budget.
        think_tokens = self._count_tokens(parsed.get("think_text", ""))
        action_tokens = self._count_tokens(parsed.get("action_text", ""))
        total_ticker_tokens = think_tokens + action_tokens

        # Update active agent's ticker (think + action both advance time)
        if self.time_manager is None:
            raise RuntimeError("TimeManager is not initialized.")
        new_ticker = self.time_manager.record_action_advance(
            active_agent, total_ticker_tokens
        )

        # Record the think block in history (private, only visible to owner)
        if parsed.get("think_text"):
            if self.history_manager is None:
                raise RuntimeError("HistoryManager is not initialized.")
            self.history_manager.add_think(
                agent_id=active_agent,
                text=parsed["think_text"],
                ticker_time=new_ticker,
                token_count=think_tokens,
            )

        # Record the action message in history (public or type-specific)
        if self.history_manager is None:
            raise RuntimeError("HistoryManager is not initialized.")

        extra_fields: Dict[str, Any] = {}
        if parsed["type"] == "vote":
            extra_fields["choice"] = parsed["choice"]
            self.episode_state["votes"][active_agent] = parsed["choice"]
            # Voted agents can no longer take actions; inform the time manager.
            self.time_manager.record_vote(active_agent)

        if parsed["type"] == "wait":
            # Attach the interpreted wait condition so it can be rendered
            # in conversation history (e.g., "waiting for prof_bob").
            extra_fields["condition"] = parsed["condition"]

        self.history_manager.add_public(
            agent_id=active_agent,
            text=parsed.get("action_text", action_text),
            ticker_time=new_ticker,
            token_count=action_tokens,
            message_type=parsed["type"],
            extra_fields=extra_fields or None,
        )

        # Handle wait actions (update time manager)
        if parsed["type"] == "wait":
            condition = parsed["condition"]
            if condition == "any_response":
                wait_info = {
                    "condition_type": "any_response",
                    "wait_issued_at": new_ticker,
                }
            else:
                wait_info = {
                    "condition_type": "agent_specific",
                    "target_agent": condition,
                    "wait_issued_at": new_ticker,
                }
            self.time_manager.set_wait(active_agent, wait_info)

        # If the agent chose to only think (no public action tag), add an
        # explicit public status message so other agents see that they
        # decided not to speak this turn.
        if parsed["type"] == "think" and not parsed.get("action_text"):
            self.history_manager.add_public(
                agent_id=active_agent,
                text="decided not to talk this turn.",
                ticker_time=new_ticker,
                token_count=0,
                message_type="status",
                extra_fields=None,
            )

        # Update total tokens used — only action tokens count against budget
        self.episode_state["tokens_used"] += action_tokens

        # Check for consensus
        consensus_reached, consensus_choice = self._check_consensus()
        if consensus_reached:
            self.episode_state["consensus_reached"] = True
            self.episode_state["consensus_choice"] = consensus_choice
            done = True
        else:
            done = self.episode_state["tokens_used"] >= self.token_budget

        # Refresh episode_state snapshots from managers
        self.episode_state["agent_tickers"] = self.time_manager.get_public_tickers()
        self.episode_state["waiting_agents"] = self.time_manager.get_public_waiting_state()
        self.episode_state["message_history"] = (
            self.history_manager.get_episode_message_history_snapshot()
        )

        # Select next agent
        public_history = self.history_manager.get_public_history()
        next_agent = self.time_manager.get_next_agent(public_history)
        self.episode_state["active_agent"] = next_agent

        # Build OpenAI-compatible observations for the next active agent
        observations = self._build_chat_messages(next_agent)

        # Calculate rewards if episode done
        if done:
            agent_rewards = self._calculate_rewards()
        else:
            agent_rewards = {agent_id: 0.0 for agent_id in self.professor_ids}

        # Build info dict
        base_info = {
            "active_agent": next_agent,
            "agent_idx": self.professor_ids.index(next_agent),
            "episode_state": deepcopy(self.episode_state),
            "agent_rewards": agent_rewards,
            "student_batch": self.student_batch,
            "professor_interests": self.professor_interests,
        }

        if self.system_prompt:
            base_info["system_prompt"] = {
                agent_id: self.get_system_prompt(agent_id) for agent_id in self.professor_ids
            }
            if self.system_prompt_token_lengths:
                base_info["system_prompt_token_length"] = self.system_prompt_token_lengths

        rewards = {agent_id: agent_rewards[agent_id] for agent_id in self.professor_ids}
        terminations = {agent_id: done for agent_id in self.professor_ids}
        truncations = {agent_id: False for agent_id in self.professor_ids}
        infos = {agent_id: deepcopy(base_info) for agent_id in self.professor_ids}

        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_discrete_simplex(self, n: int) -> List[float]:
        """
        Sample an n-dimensional vector whose entries are multiples of 0.1
        (i.e. each entry ∈ {0.0, 0.1, …, 1.0}) and sum to exactly 1.0.

        Uses the stars-and-bars trick on 10 indistinguishable units:
        randomly place (n-1) dividers among positions 0..10 to split 10
        units across n bins, then divide by 10.
        """
        # Draw (n-1) cut points from {0, 1, ..., 10}, sort them, compute gaps
        cuts = np.sort(np.random.choice(np.arange(11), size=n - 1, replace=True))
        boundaries = np.concatenate(([0], cuts, [10]))
        counts = np.diff(boundaries)          # integer counts, sum = 10
        return (counts / 10.0).tolist()

    def _parse_action(self, action_text: str) -> Dict[str, Any]:
        """
        Parse action text into a think block + action payload.

        Expected format (mandatory):
            <THINK>private reasoning</THINK><ACTION_TAG>...</ACTION_TAG>

        The think block is extracted and stored separately. The remainder
        is parsed as the action. If no <THINK> block is present, think_text
        will be an empty string.

        Action types (from the action portion after <THINK>):
          - "vote"          : <VOTE>index</VOTE>
          - "wait"          : <WAIT_FOR>agent_name</WAIT_FOR>
          - "communication" : <GROUP>...</GROUP>
          - "discuss"       : anything else (plain text / unrecognised tag)

        Note: standalone <THINK>-only responses (without a following action
        tag) are recorded as type "think" for backward compatibility, but
        agents are encouraged to always pair <THINK> with an action.

        Returns a dict with:
          - "type": action type string
          - "think_text": extracted think content (may be empty string)
          - "action_text": the action tag text (for history recording)
          - type-specific fields (e.g. "choice" for vote, "condition" for wait)
        """
        # Extract <THINK>...</THINK> block (greedy=False to get first block only)
        think_text = ""
        remainder = action_text
        think_match = re.search(r"<THINK>(.*?)</THINK>", action_text, re.DOTALL)
        if think_match:
            think_text = think_match.group(0)  # full <THINK>...</THINK> tag
            # Remove the think block from remainder to parse the action
            remainder = action_text[think_match.end():].strip()

        # If nothing remains after the think block, treat as a think-only turn
        if not remainder:
            return {
                "type": "think",
                "think_text": think_text,
                "action_text": "",
            }

        # Parse the action from the remainder
        vote_match = re.search(r"<VOTE>\s*(\d+)\s*</VOTE>", remainder)
        if vote_match:
            return {
                "type": "vote",
                "choice": int(vote_match.group(1)),
                "think_text": think_text,
                "action_text": vote_match.group(0),
            }

        wait_match = re.search(r"<WAIT_FOR>(.*?)</WAIT_FOR>", remainder, re.DOTALL)
        if wait_match:
            raw_name = wait_match.group(1).strip()
            matched_agent = self._professor_id_lookup.get(raw_name.lower())
            condition = matched_agent if matched_agent is not None else "any_response"
            return {
                "type": "wait",
                "condition": condition,
                "think_text": think_text,
                "action_text": wait_match.group(0),
            }

        communication_match = re.search(r"<GROUP>(.*?)</GROUP>", remainder, re.DOTALL)
        if communication_match:
            return {
                "type": "communication",
                "think_text": think_text,
                "action_text": communication_match.group(0),
            }

        # Fallback: plain text or unrecognised tag
        return {
            "type": "discuss",
            "think_text": think_text,
            "action_text": remainder,
        }

    def _count_tokens(self, text: str) -> int:
        """Count tokens using whitespace splitting (or tokenizer if available)."""
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text.split())

    def _check_consensus(self) -> Tuple[bool, Optional[int]]:
        """
        Check whether threshold voting consensus has been reached.

        Returns (consensus_reached, chosen_student_index).
        """
        if not self.episode_state["votes"]:
            return False, None

        valid_votes = [
            idx
            for idx in self.episode_state["votes"].values()
            if 0 <= idx < len(self.student_batch)
        ]
        if not valid_votes:
            return False, None

        from collections import Counter

        vote_counts = Counter(valid_votes)
        n_professors = len(self.professor_ids)

        for student_index, count in vote_counts.items():
            if count / n_professors >= self.vote_threshold:
                return True, student_index

        return False, None

    def _calculate_utility_for_student(
        self,
        preference_vector: np.ndarray,
        profile_vector: List[float],
    ) -> float:
        """
        Dot-product utility.

        preference_vector : integer array, permutation of [0, 1, ..., feature_dim-1]
        profile_vector    : float array, values in [0, 1] summing to 1

        Result range: [0, 4] (0 when prefs are all 0 or ability is all 0,
        4 when the highest-preference dimension gets all of the ability mass).
        """
        return float(np.dot(preference_vector, profile_vector))

    def _calculate_rewards(self) -> Dict[str, float]:
        """
        Calculate per-agent rewards at episode end.

        No consensus  → all 0.0.
        Consensus     → dot(professor_preference, selected_student_ability).
        """
        if not self.episode_state["consensus_reached"]:
            return {agent_id: 0.0 for agent_id in self.professor_ids}

        consensus_choice = self.episode_state["consensus_choice"]
        if (
            consensus_choice is None
            or consensus_choice < 0
            or consensus_choice >= len(self.student_batch)
        ):
            return {agent_id: 0.0 for agent_id in self.professor_ids}

        selected_student = self.student_batch[consensus_choice]

        return {
            agent_id: self._calculate_utility_for_student(
                self.professor_interests[agent_id],
                selected_student["profile_vector"],
            )
            for agent_id in self.professor_ids
        }