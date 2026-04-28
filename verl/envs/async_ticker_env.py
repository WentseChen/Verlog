# verl/envs/async_ticker_env.py
import gym
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import re
from copy import deepcopy
from collections import Counter

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
        self.max_steps = config.get("max_steps", None)
        self.seed_value = config.get("seed", None)
        self.max_prompt_words = config.get("max_prompt_words", None)
        self.prompt_length = config.get("prompt_length", None)

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
        self.last_prompt_truncated = False
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

        other_professors_str = ", ".join([p for p in self.professor_ids if p != professor_id])

        return (
            f"You are Professor {professor_id}, a strategic negotiator participating in an "
            f"admissions committee. Your goal is to maximize your personal utility while "
            f"achieving consensus with other professors.\n\n"
            f"GAME RULES:\n"
            f"- Each round, you review a batch of {self.students_per_batch} students.\n"
            f"- You deliberate with other professors to select one student.\n"
            f"- The other professors are: {other_professors_str}.\n"
            f"- Voting requires {int(self.vote_threshold * 100)}% agreement for consensus. "
            f"If consensus is reached, you receive utility based on how well the selected "
            f"student matches your research preferences. If NO consensus is reached, EVERYONE "
            f"gets 0 utility.\n"
            f"- Student ability vectors are {self.feature_dim}-dimensional, representing expertise "
            f"across research topics. These vectors are PUBLIC — all professors can see them.\n"
            f"- Each professor has PRIVATE preferences over these topics. Your utility score for "
            f"each student is pre-calculated and shown to you in the student table. Higher utility "
            f"means the student better matches your research interests.\n"
            f"- IMPORTANT: Your utility scores are PRIVATE — only you can see them. Other professors "
            f"have different utilities based on their own preferences.\n"
            f"- All professors share a single token budget of {self.token_budget} total tokens "
            f"(only <GROUP> message tokens consume this budget).\n\n"
            f"ACTION RULES (CRITICAL):\n"
            f"Every turn MUST begin with <THINK>...</THINK> reasoning, followed by EXACTLY ONE action tag.\n"
            f"Required format: <THINK>your private reasoning</THINK> followed by one of:\n"
            f"1. <GROUP>your message</GROUP> - Send a message visible to all (uses token budget)\n"
            f"2. <WAIT> - Skip your turn\n"
            f"3. <VOTE>student_index</VOTE> - Cast your FINAL vote for a student (e.g. <VOTE>2</VOTE>)\n\n"
            f"ALL VALID ACTIONS: (no two group, wait, or vote are allowed in one turn)\n"
            f"1. <THINK>your reasoning</THINK><GROUP>your message</GROUP>\n"
            f"2. <THINK>your reasoning</THINK><WAIT>\n"
            f"3. <THINK>your reasoning</THINK><VOTE>student_index</VOTE>\n\n"
            f"TOKEN BUDGET RULES:\n"
            f"- <THINK> tokens do NOT count against the shared token budget.\n"
            f"- <THINK> tokens DO count against your personal ticker time (turn order).\n"
            f"- Only <GROUP> message tokens consume the shared budget.\n\n"
            f"VISIBILITY RULES:\n"
            f"- <THINK> content is PRIVATE — only you can see your own thoughts in your history.\n"
            f"- <GROUP> content is PUBLIC — all professors see these messages.\n"
            f"- Other professors' <THINK> blocks are NEVER shown to you.\n\n"
            f"STRICT FORMATTING RESTRICTION:\n"
            f"- You MUST always output <THINK>...</THINK> before your action.\n"
            f"- CRITICAL: Always close your <THINK> tag with </THINK> before starting your action tag.\n"
            f"- Format: <THINK>reasoning</THINK><ACTION_TAG>content</ACTION_TAG>\n"
            f"- Plain text or responses missing <THINK> will be treated as discuss-type actions.\n"
            f"- Never add text outside the tags.\n\n"
            f"- Keep <GROUP> messages concise - aim for 2 sentences or less.\n"
            f"- Keep <THINK> reasoning concise - aim for 3 sentences or less to organize your thoughts efficiently. You don't want to take up too much space, or your message history will get truncated in the future.\n"
            f"VOTING RULES:\n"
            f"- To cast a vote you MUST use the tag <VOTE>N</VOTE> where N is the student index (0-{self.students_per_batch - 1}).\n"
            f"  Example: <THINK>Student 2 is best.</THINK><VOTE>2</VOTE>\n"
            f"- NEVER write your vote inside a <GROUP> message. <GROUP>Vote for Student 2</GROUP> does NOT count as a vote.\n"
            f"- <VOTE> is a FINAL, IRREVOCABLE commitment. Once you vote, you cannot act again.\n"
            f"- ALWAYS check the CURRENT VOTE TALLY shown in your observation before acting.\n"
            f"- If you see that 2 professors already agree on a student, vote for that student immediately to reach consensus — even if it is not your top choice. Getting something is better than everyone getting 0.\n"
            f"- If you have made up your mind, VOTE immediately using <VOTE>N</VOTE>. Do not send GROUP messages about it.\n"
            f"- Discussing your intention to vote in a GROUP message wastes tokens. Use <VOTE>N</VOTE> directly.\n\n"
            f"STRATEGIC CONSIDERATIONS:\n"
            f"- Your utility scores are shown in the student table. Use them to guide your preferences.\n"
            f"- DO NOT reveal your exact utility numbers to others in <GROUP> messages.\n"
            f"- The student ability vectors are public, so focus your negotiation on which student best serves the group's combined interests.\n"
            f"- Use <THINK> for private reasoning before each action, but be concise.\n"
            f"- Use <WAIT> to skip your turn and save token budget.\n"
            f"- Try to maximize your utility by persuading the other professors strategically within the token budget."
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

        # Compute token-based user-content budget when the tokenizer and prompt_length
        # are both available.  This ensures the assembled observation always fits within
        # prompt_length before it reaches the tokenizer in the agent loop, so the
        # fallback left-truncation (which drops the critical header) never fires.
        max_user_tokens: int | None = None
        if self.tokenizer is not None and self.prompt_length is not None:
            # Cache system-prompt token length on first encounter for this agent.
            if agent_id not in self.system_prompt_token_lengths:
                self.system_prompt_token_lengths[agent_id] = len(
                    self.tokenizer.encode(sys_prompt, add_special_tokens=False)
                )
            system_tokens = self.system_prompt_token_lengths[agent_id]
            # Reserve ~25 tokens for chat-template wrappers (im_start/im_end etc.).
            _CHAT_TEMPLATE_OVERHEAD = 25
            max_user_tokens = self.prompt_length - system_tokens - _CHAT_TEMPLATE_OVERHEAD

        current_votes = self.episode_state.get("votes", {}) if self.episode_state else {}
        user_content, self.last_prompt_truncated = self.history_manager.get_obs(
            agent_id=agent_id,
            current_ticker=current_ticker,
            token_budget_used=tokens_used,
            token_budget=self.token_budget,
            student_batch=self.student_batch,
            professor_interest_vector=professor_interest_vec,
            feature_dim=self.feature_dim,
            max_prompt_words=self.max_prompt_words if max_user_tokens is None else None,
            max_prompt_tokens=max_user_tokens,
            tokenizer=self.tokenizer if max_user_tokens is not None else None,
            current_votes=current_votes,
        )

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[List[Dict[str, str]], Dict[str, Dict]]:
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
            "step_count": 0,
        }

        # Select first agent (lexicographic order when all at 0)
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
            extra_fields["condition"] = "skip"

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
            self.time_manager.set_wait(active_agent, {
                "condition_type": "any_response",
                "wait_issued_at": new_ticker,
            })

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

        # Update total tokens used — think + action both count against budget
        self.episode_state["tokens_used"] += total_ticker_tokens
        self.episode_state["step_count"] += 1

        # Check for consensus
        consensus_reached, consensus_choice = self._check_consensus()
        if consensus_reached:
            self.episode_state["consensus_reached"] = True
            self.episode_state["consensus_choice"] = consensus_choice
            done = True
        else:
            all_voted = self.time_manager.all_voted()
            budget_exceeded = self.episode_state["tokens_used"] >= self.token_budget
            steps_exceeded = (
                self.max_steps is not None
                and self.episode_state["step_count"] >= self.max_steps
            )
            done = all_voted or budget_exceeded or steps_exceeded

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

        # Add episode metrics if episode is done (for VeRLOG/W&B logging)
        if done:
            episode_metrics = self._calculate_episode_metrics(agent_rewards)
            flattened_metrics = self._flatten_metrics_for_logging(episode_metrics)
            base_info["episode_metrics"] = episode_metrics  # Full nested version
            base_info["metrics"] = flattened_metrics  # Flattened for logging

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

    def get_all_agent_observations(self) -> Dict[str, List[Dict[str, str]]]:
        """Return current chat-message observations for every agent (for bootstrapping)."""
        return {aid: self._build_chat_messages(aid) for aid in self.professor_ids}

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
          - "wait"          : <WAIT/>
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
            think_text = think_match.group(1).strip()  # content inside <THINK> tags only
            # Remove the think block from remainder to parse the action
            remainder = action_text[think_match.end():].strip()

        # If nothing remains after the think block, treat as a think-only turn
        if not remainder:
            return {
                "type": "think",
                "think_text": think_text,
                "action_text": "",
            }

        # Parse the action from the remainder (priority: VOTE > WAIT > GROUP)
        _ACTION_TAG_RE = re.compile(r"<VOTE>|<WAIT[\s/]|<GROUP>")

        def _has_extra_action_tag(text: str, consumed_span) -> bool:
            """Return True if there is an action tag outside the consumed match span."""
            before = text[:consumed_span[0]]
            after = text[consumed_span[1]:]
            return bool(_ACTION_TAG_RE.search(before) or _ACTION_TAG_RE.search(after))

        vote_match = re.search(r"<VOTE>\s*(\d+)\s*</VOTE>", remainder)
        if vote_match:
            if _has_extra_action_tag(remainder, vote_match.span()):
                return {"type": "raw", "think_text": think_text, "action_text": remainder}
            return {
                "type": "vote",
                "choice": int(vote_match.group(1)),
                "think_text": think_text,
                "action_text": vote_match.group(0),
            }

        wait_match = re.search(r"<WAIT\s*/>|<WAIT\s*>(?:</WAIT\s*>)?", remainder, re.DOTALL)
        if wait_match:
            if _has_extra_action_tag(remainder, wait_match.span()):
                return {"type": "raw", "think_text": think_text, "action_text": remainder}
            return {
                "type": "wait",
                "think_text": think_text,
                "action_text": wait_match.group(0),
            }

        communication_match = re.search(r"<GROUP>(.*?)</GROUP>", remainder, re.DOTALL)
        if communication_match:
            if _has_extra_action_tag(remainder, communication_match.span()):
                return {"type": "raw", "think_text": think_text, "action_text": remainder}
            group_content = communication_match.group(1).strip()
            # Detect GROUP messages that are actually vote declarations, e.g.:
            #   "Vote for Student 2", "Voting for Student 3",
            #   "Cast my vote for Student 4", "Will vote for Student 0",
            #   "I will cast my vote for Student 2"
            # Anchored at start to avoid matching "I will persuade ... vote for".
            vote_in_group_match = re.match(
                r"(?:i\s+will\s+|i(?:'ll)?\s+|will\s+)?"
                r"(?:cast\s+(?:my\s+|a\s+|our\s+)?)?"
                r"(?:vote|voting)\s+for\s+student\s*(\d+)",
                group_content,
                re.IGNORECASE,
            )
            if vote_in_group_match:
                student_idx = vote_in_group_match.group(1)
                return {
                    "type": "vote",
                    "choice": int(student_idx),
                    "think_text": think_text,
                    "action_text": f"<VOTE>{student_idx}</VOTE>",
                }
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

        A penalty of 10% of the maximum reward (0.4) is applied per invalid
        action (discuss/raw message types) per agent.
        """
        invalid_penalty = 0.1 * (self.feature_dim - 1)  # 10% of max utility

        # Count invalid actions per agent
        invalid_counts: Dict[str, int] = {agent_id: 0 for agent_id in self.professor_ids}
        for msg in self.history_manager._all_messages:
            if msg["message_type"] in ("discuss", "raw") and msg["agent_id"] in invalid_counts:
                invalid_counts[msg["agent_id"]] += 1

        if not self.episode_state["consensus_reached"]:
            return {
                agent_id: -(invalid_counts[agent_id] * invalid_penalty)
                for agent_id in self.professor_ids
            }

        consensus_choice = self.episode_state["consensus_choice"]
        if (
            consensus_choice is None
            or consensus_choice < 0
            or consensus_choice >= len(self.student_batch)
        ):
            return {
                agent_id: -(invalid_counts[agent_id] * invalid_penalty)
                for agent_id in self.professor_ids
            }

        selected_student = self.student_batch[consensus_choice]

        return {
            agent_id: self._calculate_utility_for_student(
                self.professor_interests[agent_id],
                selected_student["profile_vector"],
            ) - (invalid_counts[agent_id] * invalid_penalty)
            for agent_id in self.professor_ids
        }

    def _calculate_episode_metrics(self, agent_rewards: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate comprehensive episode metrics for logging.

        These metrics capture information that cannot be reconstructed
        without the full episode interaction log.
        """
        import numpy as np

        metrics = {}

        # 1. SOCIAL WELFARE METRICS
        # Calculate penalty-free utilities for all students to find optimal
        all_utilities = []
        for student in self.student_batch:
            total_utility = sum(
                self._calculate_utility_for_student(
                    self.professor_interests[agent_id],
                    student["profile_vector"]
                )
                for agent_id in self.professor_ids
            )
            all_utilities.append(total_utility)

        optimal_total_utility = max(all_utilities)
        optimal_student = all_utilities.index(optimal_total_utility)

        # Compute actual utility without penalties for efficiency metric
        consensus_choice = self.episode_state.get("consensus_choice")
        if self.episode_state["consensus_reached"] and consensus_choice is not None and 0 <= consensus_choice < len(self.student_batch):
            selected_profile = self.student_batch[consensus_choice]["profile_vector"]
            actual_total_utility = sum(
                self._calculate_utility_for_student(self.professor_interests[a], selected_profile)
                for a in self.professor_ids
            )
        else:
            actual_total_utility = 0.0

        metrics["social_welfare"] = {
            "actual_total_utility": actual_total_utility,
            "optimal_total_utility": optimal_total_utility,
            "optimal_student": optimal_student,
            "efficiency": actual_total_utility / optimal_total_utility if optimal_total_utility > 0 else 0.0,
            "utilitarian_gap": optimal_total_utility - actual_total_utility,
        }

        # 2. TOKEN ACCOUNTING
        # Get token breakdown from history manager
        think_tokens_by_agent = {}
        action_tokens_by_agent = {}

        for msg in self.history_manager._all_messages:
            agent_id = msg["agent_id"]
            tokens = msg["token_count"]

            if agent_id not in think_tokens_by_agent:
                think_tokens_by_agent[agent_id] = 0
                action_tokens_by_agent[agent_id] = 0

            if msg["message_type"] == "think":
                think_tokens_by_agent[agent_id] += tokens
            else:
                action_tokens_by_agent[agent_id] += tokens

        total_think = sum(think_tokens_by_agent.values())
        total_action = sum(action_tokens_by_agent.values())

        by_agent = {}
        for agent_id in self.professor_ids:
            think = think_tokens_by_agent.get(agent_id, 0)
            action = action_tokens_by_agent.get(agent_id, 0)
            by_agent[agent_id] = {
                "think_tokens": think,
                "action_tokens": action,
                "total_tokens": think + action,
            }

        metrics["token_accounting"] = {
            "total_budget": self.token_budget,
            "tokens_used": self.episode_state["tokens_used"],
            "budget_utilization": self.episode_state["tokens_used"] / self.token_budget if self.token_budget > 0 else 0.0,
            "total_response_tokens": total_think + total_action,
            "token_breakdown": {
                "think_tokens": total_think,
                "action_tokens": total_action,
                "think_ratio": total_think / total_action if total_action > 0 else 0.0,
            },
            "by_agent": by_agent,
        }

        # 3. ACTION VALIDITY
        # Valid turns: single valid action (comm/vote/wait) OR think-only (status message).
        # Invalid turns: raw (malformed/multi-action), discuss (missing tags).
        # Denominator = total steps taken (every turn counts).
        valid_count = 0
        invalid_turns = []
        type_counts = {
            "communication": 0,
            "vote": 0,
            "wait": 0,
            "think_only": 0,
            "raw": 0,
            "discuss": 0,
        }

        for msg in self.history_manager._all_messages:
            if msg["message_type"] == "think":
                continue

            msg_type = msg["message_type"]

            if msg_type == "communication":
                type_counts["communication"] += 1
                valid_count += 1
            elif msg_type == "vote":
                type_counts["vote"] += 1
                valid_count += 1
            elif msg_type == "wait":
                type_counts["wait"] += 1
                valid_count += 1
            elif msg_type == "status":
                # status = think-only turn; agent chose not to act publicly (valid)
                type_counts["think_only"] += 1
                valid_count += 1
            elif msg_type == "raw":
                type_counts["raw"] += 1
                invalid_turns.append({
                    "turn": len([m for m in self.history_manager._all_messages if m["ticker_time"] <= msg["ticker_time"]]),
                    "agent": msg["agent_id"],
                    "type": "raw",
                    "reason": "malformed_output"
                })
            elif msg_type == "discuss":
                type_counts["discuss"] += 1
                invalid_turns.append({
                    "turn": len([m for m in self.history_manager._all_messages if m["ticker_time"] <= msg["ticker_time"]]),
                    "agent": msg["agent_id"],
                    "type": "discuss",
                    "reason": "missing_action_tags"
                })

        total_steps = self.episode_state["step_count"]
        invalid_count = type_counts["raw"] + type_counts["discuss"]

        metrics["action_validity"] = {
            "total_actions": total_steps,
            "valid_actions": valid_count,
            "invalid_actions": invalid_count,
            "validity_rate": valid_count / total_steps if total_steps > 0 else 0.0,
            "by_type": type_counts,
            "invalid_turns": invalid_turns,
        }

        # 4. NEGOTIATION DYNAMICS
        # Find first and last vote turns
        vote_turns = []
        for i, msg in enumerate(self.history_manager._all_messages):
            if msg["message_type"] == "vote":
                vote_turns.append(i)

        first_vote_turn = vote_turns[0] if vote_turns else None
        last_vote_turn = vote_turns[-1] if vote_turns else None

        # Count votes by student
        vote_distribution = {}
        for msg in self.history_manager._all_messages:
            if msg["message_type"] == "vote" and "choice" in msg:
                student_idx = msg["choice"]
                vote_distribution[student_idx] = vote_distribution.get(student_idx, 0) + 1

        non_think_messages = [m for m in self.history_manager._all_messages if m["message_type"] != "think"]
        total_turns = len(non_think_messages)

        # Find the exact turn at which consensus was reached (the deciding vote)
        consensus_turn = None
        if self.episode_state["consensus_reached"]:
            running_votes = {}
            for i, msg in enumerate(non_think_messages):
                if msg["message_type"] == "vote" and "choice" in msg:
                    running_votes[msg["agent_id"]] = msg["choice"]
                    valid = [idx for idx in running_votes.values() if 0 <= idx < len(self.student_batch)]
                    for student_idx, count in Counter(valid).items():
                        if count / len(self.professor_ids) >= self.vote_threshold:
                            consensus_turn = i + 1  # 1-indexed
                            break
                if consensus_turn is not None:
                    break

        metrics["negotiation_dynamics"] = {
            "total_turns": total_turns,
            "turns_to_first_vote": first_vote_turn if first_vote_turn is not None else None,
            "turns_to_consensus": consensus_turn,
            "voting_duration": (last_vote_turn - first_vote_turn) if (first_vote_turn is not None and last_vote_turn is not None) else 0,
            "votes_cast": len(self.episode_state["votes"]),
            "vote_distribution": vote_distribution,
            "consensus_reached": int(self.episode_state["consensus_reached"]),
        }

        # 5. FAIRNESS METRICS — use penalty-free utilities for Gini
        _consensus_choice = self.episode_state.get("consensus_choice")
        if self.episode_state["consensus_reached"] and _consensus_choice is not None and 0 <= _consensus_choice < len(self.student_batch):
            _selected_profile = self.student_batch[_consensus_choice]["profile_vector"]
            utility_values = [
                self._calculate_utility_for_student(self.professor_interests[a], _selected_profile)
                for a in self.professor_ids
            ]
        else:
            utility_values = [0.0] * len(self.professor_ids)

        if utility_values:
            util_array = np.array(utility_values)
            sorted_util = np.sort(util_array)
            n = len(sorted_util)
            total_util = np.sum(sorted_util)
            gini = (2 * np.sum(np.arange(1, n + 1) * sorted_util)) / (n * total_util) - (n + 1) / n if total_util > 0 else 0.0

            metrics["fairness"] = {
                "min_reward": float(np.min(util_array)),
                "max_reward": float(np.max(util_array)),
                "reward_range": float(np.max(util_array) - np.min(util_array)),
                "reward_std": float(np.std(util_array)),
                "gini_coefficient": float(gini),
            }
        else:
            metrics["fairness"] = {
                "min_reward": 0.0,
                "max_reward": 0.0,
                "reward_range": 0.0,
                "reward_std": 0.0,
                "gini_coefficient": 0.0,
            }

        # 6. PREFERENCE ALIGNMENT
        preference_alignment = {}
        for agent_id in self.professor_ids:
            # Calculate utility for all students
            utilities = [
                self._calculate_utility_for_student(
                    self.professor_interests[agent_id],
                    student["profile_vector"]
                )
                for student in self.student_batch
            ]

            # Get ranking (indices sorted by utility, descending)
            utility_ranking = sorted(range(len(utilities)), key=lambda i: utilities[i], reverse=True)
            top_choice = utility_ranking[0]
            max_utility = utilities[top_choice]

            # Check what this agent voted for
            voted_for = self.episode_state["votes"].get(agent_id)

            if voted_for is not None and 0 <= voted_for < len(self.student_batch):
                voted_for_rank = utility_ranking.index(voted_for) + 1  # 1-indexed
                voted_for_utility = utilities[voted_for]
                compromise_ratio = voted_for_utility / max_utility if max_utility > 0 else 0.0
            else:
                voted_for_rank = None
                compromise_ratio = None

            preference_alignment[agent_id] = {
                "utility_ranking": utility_ranking,
                "top_choice": top_choice,
                "voted_for": voted_for,
                "voted_for_rank": voted_for_rank,
                "compromise_ratio": compromise_ratio,
            }

        metrics["preference_alignment"] = preference_alignment

        # 7. COMMUNICATION METRICS
        group_messages_by_agent = {}
        total_message_tokens = 0
        message_count = 0

        for msg in self.history_manager._all_messages:
            if msg["message_type"] == "communication":
                agent_id = msg["agent_id"]
                group_messages_by_agent[agent_id] = group_messages_by_agent.get(agent_id, 0) + 1
                total_message_tokens += msg["token_count"]
                message_count += 1

        # Find agents who never sent GROUP messages
        silent_agents = [agent_id for agent_id in self.professor_ids if group_messages_by_agent.get(agent_id, 0) == 0]

        metrics["communication"] = {
            "total_messages": message_count,
            "total_tokens": total_message_tokens,
            "messages_per_agent": {agent_id: group_messages_by_agent.get(agent_id, 0) for agent_id in self.professor_ids},
            "silent_agents": silent_agents,
            "avg_message_length": total_message_tokens / message_count if message_count > 0 else 0.0,
        }

        # 8. CORE METRICS (for W&B logging)
        total_turns = metrics["negotiation_dynamics"]["total_turns"]
        comm_turns = type_counts["communication"]
        think_turns = type_counts["think_only"]
        reward_values_list = list(agent_rewards.values())

        # Did the first agent to move vote on their first turn?
        first_agent_voted_first_turn = 0
        if non_think_messages:
            first_msg = non_think_messages[0]
            if first_msg["message_type"] == "vote":
                first_agent_voted_first_turn = 1

        metrics["core_metrics"] = {
            "efficiency": metrics["social_welfare"]["efficiency"],
            "gini_coefficient": metrics["fairness"]["gini_coefficient"],
            "communication_turn_rate": comm_turns / total_turns if total_turns > 0 else 0.0,
            "wait_turn_rate": type_counts["wait"] / total_turns if total_turns > 0 else 0.0,
            "total_response_tokens": metrics["token_accounting"]["total_response_tokens"],
            "total_communication_tokens": metrics["communication"]["total_tokens"],
            "mean_reward": float(np.mean(reward_values_list)) if reward_values_list else 0.0,
            "valid_action_rate": metrics["action_validity"]["validity_rate"],
            "consensus_reached": metrics["negotiation_dynamics"]["consensus_reached"],
            "first_agent_vote_first_turn": first_agent_voted_first_turn,
        }

        return metrics

    def _flatten_metrics_for_logging(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten metrics dict for VeRLOG/W&B logging compatibility.

        Converts nested structures to flat key-value pairs with scalar values only.
        Keys use '/' separator for hierarchical grouping in W&B.
        """
        flattened = {}

        # 1. SOCIAL WELFARE - all scalars, flatten directly
        for key, value in metrics["social_welfare"].items():
            flattened[f"social_welfare/{key}"] = value

        # 2. TOKEN ACCOUNTING - flatten nested structure
        flattened["token_accounting/total_budget"] = metrics["token_accounting"]["total_budget"]
        flattened["token_accounting/tokens_used"] = metrics["token_accounting"]["tokens_used"]
        flattened["token_accounting/budget_utilization"] = metrics["token_accounting"]["budget_utilization"]
        flattened["token_accounting/think_tokens"] = metrics["token_accounting"]["token_breakdown"]["think_tokens"]
        flattened["token_accounting/action_tokens"] = metrics["token_accounting"]["token_breakdown"]["action_tokens"]
        flattened["token_accounting/think_ratio"] = metrics["token_accounting"]["token_breakdown"]["think_ratio"]

        # Per-agent token stats
        for agent_id, agent_tokens in metrics["token_accounting"]["by_agent"].items():
            flattened[f"token_accounting/{agent_id}/think"] = agent_tokens["think_tokens"]
            flattened[f"token_accounting/{agent_id}/action"] = agent_tokens["action_tokens"]
            flattened[f"token_accounting/{agent_id}/total"] = agent_tokens["total_tokens"]

        # 3. ACTION VALIDITY - flatten, skip complex types
        flattened["action_validity/total_actions"] = metrics["action_validity"]["total_actions"]
        flattened["action_validity/valid_actions"] = metrics["action_validity"]["valid_actions"]
        flattened["action_validity/invalid_actions"] = metrics["action_validity"]["invalid_actions"]
        flattened["action_validity/validity_rate"] = metrics["action_validity"]["validity_rate"]

        # Action types breakdown
        for action_type, count in metrics["action_validity"]["by_type"].items():
            flattened[f"action_validity/type_{action_type}"] = count

        # 4. NEGOTIATION DYNAMICS
        flattened["negotiation/total_turns"] = metrics["negotiation_dynamics"]["total_turns"]
        flattened["negotiation/turns_to_first_vote"] = metrics["negotiation_dynamics"]["turns_to_first_vote"] if metrics["negotiation_dynamics"]["turns_to_first_vote"] is not None else -1
        flattened["negotiation/turns_to_consensus"] = metrics["negotiation_dynamics"]["turns_to_consensus"] if metrics["negotiation_dynamics"]["turns_to_consensus"] is not None else -1
        flattened["negotiation/consensus_reached"] = metrics["negotiation_dynamics"]["consensus_reached"]
        flattened["negotiation/voting_duration"] = metrics["negotiation_dynamics"]["voting_duration"]
        flattened["negotiation/votes_cast"] = metrics["negotiation_dynamics"]["votes_cast"]

        # Vote distribution
        for student_idx, vote_count in metrics["negotiation_dynamics"]["vote_distribution"].items():
            flattened[f"negotiation/votes_for_student_{student_idx}"] = vote_count

        # 5. FAIRNESS - all scalars
        for key, value in metrics["fairness"].items():
            flattened[f"fairness/{key}"] = value

        # 6. PREFERENCE ALIGNMENT - per agent, skip lists
        for agent_id, alignment in metrics["preference_alignment"].items():
            flattened[f"preference/{agent_id}/top_choice"] = alignment["top_choice"]
            flattened[f"preference/{agent_id}/voted_for"] = alignment["voted_for"] if alignment["voted_for"] is not None else -1
            flattened[f"preference/{agent_id}/voted_for_rank"] = alignment["voted_for_rank"] if alignment["voted_for_rank"] is not None else -1
            flattened[f"preference/{agent_id}/compromise_ratio"] = alignment["compromise_ratio"] if alignment["compromise_ratio"] is not None else -1.0

        # 7. COMMUNICATION
        flattened["communication/total_messages"] = metrics["communication"]["total_messages"]
        flattened["communication/avg_message_length"] = metrics["communication"]["avg_message_length"]
        flattened["communication/num_silent_agents"] = len(metrics["communication"]["silent_agents"])

        # Per-agent message counts
        for agent_id, count in metrics["communication"]["messages_per_agent"].items():
            flattened[f"communication/{agent_id}/messages"] = count
        flattened["communication/total_tokens"] = metrics["communication"]["total_tokens"]

        # 8. CORE METRICS
        for key, value in metrics["core_metrics"].items():
            flattened[f"core/{key}"] = value

        return flattened

class AsyncTickerEnvWrapper(gym.Wrapper):
    """
    Wrapper around AsyncTickerAdmissionsEnv that caches the last observation and infos
    and exposes get_last_obs(agent_id=None) for resuming (e.g. training rollouts).
    reset() and step() return the same format as the inner env.
    """

    def __init__(self, env: AsyncTickerAdmissionsEnv):
        super().__init__(env)
        self._last_observations: Optional[List[Dict[str, str]]] = None
        self._last_infos: Optional[Dict[str, Dict]] = None
        self._episode_done: bool = False

    def reset(self, agent_id: str | None = None) -> Tuple[List[Dict[str, str]], Dict[str, Dict]]:
        observations, infos = self.env.reset()
        info = infos[self.env.episode_state["active_agent"]]
        self._last_observations = observations
        self._last_infos = info
        self._episode_done = False
        return observations, info

    def step(
        self, action: str | Dict[str, Any]
    ) -> Tuple[
        List[Dict[str, str]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
        if self._episode_done:
            observations, info = self.reset()
            # Discard the stale action from the previous episode; return the
            # fresh initial observation so the caller generates a new response.
            return observations, 0.0, False, False, info
        acting_agent = self.env.episode_state["active_agent"]
        observations, rewards, terminations, truncations, infos = self.env.step(action)
        reward = rewards[acting_agent]
        terminated = terminations[acting_agent]
        truncated = truncations[acting_agent]
        info = infos[acting_agent]
        self._last_observations = observations
        self._last_infos = info
        if terminated or truncated:
            self._episode_done = True
        return observations, reward, terminated, truncated, info

    def get_last_obs(
        self, agent_id: str | None = None
    ) -> Tuple[Optional[List[Dict[str, str]]], Optional[Dict[str, Dict]]]:
        """Return the last (observations, infos) from the most recent reset() or step().
        Returns (None, None) if the episode has ended, so the caller triggers reset()."""
        if self._last_observations is None or self._last_infos is None or self._episode_done:
            return None, None
        return self._last_observations, self._last_infos