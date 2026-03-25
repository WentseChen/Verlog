from typing import Any, Dict, List

import numpy as np


class HistoryManager:
    """
    Manage public and private message buffers and construct observations.

    Public buffer:
      - Stores messages visible to all agents (GROUP, vote, wait, status, discuss).
    Private buffers:
      - Per-agent buffers storing that agent's private THINK messages.
    """

    # Message types that are public (shown to all agents in conversation history)
    # NOTE: "wait" and "status" are included so that waiting / no-op turns
    # are explicitly visible in the shared conversation history.
    _PUBLIC_MESSAGE_TYPES = frozenset(
        {"communication", "vote", "discuss", "wait", "status"}
    )

    def __init__(self, professor_ids: List[str]) -> None:
        self.professor_ids = list(professor_ids)

        # Public, global buffer
        self.public_messages: List[Dict[str, Any]] = []

        # Private buffers per professor for THINK messages
        self.private_messages: Dict[str, List[Dict[str, Any]]] = {
            pid: [] for pid in self.professor_ids
        }

        # Combined chronological history for episode_state snapshots
        self._all_messages: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Message recording
    # ------------------------------------------------------------------

    def add_think(
        self,
        agent_id: str,
        text: str,
        ticker_time: int,
        token_count: int,
    ) -> None:
        """Record a private THINK message for *agent_id*."""
        msg = {
            "agent_id": agent_id,
            "text": text,
            "ticker_time": ticker_time,
            "token_count": token_count,
            "message_type": "think",
            "private": True,
        }
        self.private_messages[agent_id].append(msg)
        self._all_messages.append(msg)

    def add_public(
        self,
        agent_id: str,
        text: str,
        ticker_time: int,
        token_count: int,
        message_type: str,
        extra_fields: Dict[str, Any] | None = None,
    ) -> None:
        """
        Record a public message, visible to all agents.

        extra_fields can include keys like "choice" (for votes) or
        "condition" (for waits).
        """
        msg: Dict[str, Any] = {
            "agent_id": agent_id,
            "text": text,
            "ticker_time": ticker_time,
            "token_count": token_count,
            "message_type": message_type,
        }
        if extra_fields:
            msg.update(extra_fields)

        self.public_messages.append(msg)
        self._all_messages.append(msg)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_public_history(self) -> List[Dict[str, Any]]:
        """Return the list of public messages."""
        return list(self.public_messages)

    def get_episode_message_history_snapshot(self) -> List[Dict[str, Any]]:
        """
        Return a snapshot of the full episode message history.

        This matches the original env.py structure used in episode_state.
        """
        return list(self._all_messages)

    def _get_visible_messages_for_agent(
        self,
        agent_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Collect visible messages for *agent_id*:
        - All public messages from anyone.
        - Private THINK messages from this agent only.
        """
        visible_messages: List[Dict[str, Any]] = []
        for m in self._all_messages:
            if m["message_type"] in self._PUBLIC_MESSAGE_TYPES:
                visible_messages.append(m)
            elif m["message_type"] == "think" and m["agent_id"] == agent_id:
                visible_messages.append(m)

        visible_messages.sort(key=lambda m: m["ticker_time"])
        return visible_messages

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_utility_for_student(
        preference_vector: np.ndarray,
        profile_vector: List[float],
    ) -> float:
        """Simple dot-product utility."""
        return float(np.dot(preference_vector, profile_vector))

    def get_obs(
        self,
        agent_id: str,
        current_ticker: int,
        token_budget_used: int,
        token_budget: int,
        student_batch: List[Dict[str, Any]],
        professor_interest_vector: np.ndarray,
        feature_dim: int,
        max_prompt_words: int | None = None,
        max_prompt_tokens: int | None = None,
        tokenizer=None,
        current_votes: Dict[str, int] | None = None,
    ) -> tuple[str, bool]:
        """
        Build the *user* turn content string for the given agent.

        Conversation history shown to agent_id:
        - All public messages from ALL agents.
        - The observing agent's OWN THINK blocks (private to owner).
        """
        visible_messages = self._get_visible_messages_for_agent(agent_id)

        conversation_lines: List[str] = []
        for msg in visible_messages:
            if msg["message_type"] == "think":
                # Show own think blocks with a private marker
                conversation_lines.append(
                    f"[{msg['agent_id']} at t={msg['ticker_time']}] (YOUR PRIVATE THOUGHT): {msg['text']}"
                )
            else:
                conversation_lines.append(
                    f"[{msg['agent_id']} at t={msg['ticker_time']}]: {msg['text']}"
                )

        # Build public student table (ability vectors visible to all)
        topic_names = ["AI/ML", "Systems", "Theory", "HCI", "CompBio"]
        header = "  {:<12} | {:>6} | {}".format(
            "Student",
            "Utility",
            "Ability Vector  ["
            + ", ".join(f"{t:>6}" for t in topic_names[:feature_dim])
            + "]",
        )
        separator = "  " + "-" * (len(header) - 2)
        student_lines = [header, separator]

        for student in student_batch:
            utility = self._calculate_utility_for_student(
                professor_interest_vector,
                student["profile_vector"],
            )
            ability_str = "[" + ", ".join(f"{v:.1f}" for v in student["profile_vector"]) + "]"
            student_lines.append(
                f"  {student['name']:<12} | {utility:>6.1f} | {ability_str} "
            )
        students_text = "\n".join(student_lines)

        # If a word budget is set, drop oldest messages until history fits.
        # This preserves message boundaries and keeps the header/footer intact.
        # Build vote tally block
        if current_votes:
            vote_lines = []
            for prof, student_idx in current_votes.items():
                student_name = (
                    student_batch[student_idx]["name"]
                    if 0 <= student_idx < len(student_batch)
                    else f"#{student_idx}"
                )
                vote_lines.append(f"  {prof} -> {student_name} (index {student_idx})")
            votes_text = "\n".join(vote_lines)
        else:
            votes_text = "  (no votes cast yet)"

        was_truncated = False
        if max_prompt_tokens is not None and tokenizer is not None:
            # Token-based trimming: measure fixed header/footer tokens exactly,
            # then drop oldest history lines until everything fits.
            fixed_text = (
                f"=== YOUR TURN (t={current_ticker}) ===\n\n"
                f"STUDENTS — PUBLIC ABILITY VECTORS (sum = 1.0 per student):\n"
                f"{students_text}\n\n"
                f"NOTE: The ability vectors above are visible to ALL professors. "
                f"Your utility column is computed privately from your preference vector "
                f"and is NOT visible to others.\n\n"
                f"Token budget used: {token_budget_used}/{token_budget} "
                f"(only <GROUP> messages consume budget; <THINK> is free)\n\n"
                f"CURRENT VOTE TALLY:\n{votes_text}\n\n"
                f"CONVERSATION HISTORY (chronological by ticker time):\n"
                f"(No messages yet)\n\n"
                f"Your turn (remember: start with <THINK>your reasoning</THINK>, then your action):"
            )
            fixed_tokens = len(tokenizer.encode(fixed_text, add_special_tokens=False))
            history_budget = max_prompt_tokens - fixed_tokens
            # Pre-tokenize each line once, then greedily drop oldest until within budget.
            line_token_counts = [
                len(tokenizer.encode(line, add_special_tokens=False))
                for line in conversation_lines
            ]
            total_tokens = sum(line_token_counts)
            # Add 1 token per line for the "\n" separator between lines.
            total_tokens += len(line_token_counts)
            while conversation_lines and total_tokens > history_budget:
                total_tokens -= line_token_counts[0] + 1
                line_token_counts.pop(0)
                conversation_lines.pop(0)
                was_truncated = True
        elif max_prompt_words is not None:
            # Legacy word-based trimming (fallback when no tokenizer is available).
            fixed_text = (
                f"=== YOUR TURN (t={current_ticker}) ===\n\n"
                f"STUDENTS — PUBLIC ABILITY VECTORS (sum = 1.0 per student):\n"
                f"{students_text}\n\n"
                f"NOTE: The ability vectors above are visible to ALL professors. "
                f"Your utility column is computed privately from your preference vector "
                f"and is NOT visible to others.\n\n"
                f"Token budget used: {token_budget_used}/{token_budget} "
                f"(only <GROUP> messages consume budget; <THINK> is free)\n\n"
                f"CURRENT VOTE TALLY:\n{votes_text}\n\n"
                f"CONVERSATION HISTORY (chronological by ticker time):\n"
                f"\n\n"
                f"Your turn:"
            )
            fixed_words = len(fixed_text.split())
            history_budget = max_prompt_words - fixed_words
            while conversation_lines and len(" ".join(conversation_lines).split()) > history_budget:
                conversation_lines.pop(0)
                was_truncated = True

        conversation_text = (
            "\n".join(conversation_lines) if conversation_lines else "(No messages yet)"
        )

        obs = (
            f"=== YOUR TURN (t={current_ticker}) ===\n\n"
            f"STUDENTS — PUBLIC ABILITY VECTORS (sum = 1.0 per student):\n"
            f"{students_text}\n\n"
            f"NOTE: The ability vectors above are visible to ALL professors. "
            f"Your utility column is computed privately from your preference vector "
            f"and is NOT visible to others.\n\n"
            f"Token budget used: {token_budget_used}/{token_budget} "
            f"(only <GROUP> messages consume budget; <THINK> is free)\n\n"
            f"CURRENT VOTE TALLY:\n{votes_text}\n\n"
            f"CONVERSATION HISTORY (chronological by ticker time):\n{conversation_text}\n\n"
            f"Your turn (remember: start with <THINK>your reasoning</THINK>, then your action):"
        )

        return obs, was_truncated
