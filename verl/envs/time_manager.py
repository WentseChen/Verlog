import heapq
from typing import Any, Dict, List, Tuple


class TimeManager:
    """
    Manage agent turn order based on ticker values and wait conditions.

    Responsibilities:
    - Track per-agent ticker values.
    - Track agents that have already voted (cannot act again).
    - Track waiting conditions for agents.
    - Select the next active agent (smallest ticker, with wait logic).
    """

    def __init__(self, professor_ids: List[str]) -> None:
        self.professor_ids = list(professor_ids)
        # Public state snapshots
        self.agent_tickers: Dict[str, int] = {agent_id: 0 for agent_id in self.professor_ids}
        self.waiting_agents: Dict[str, Dict[str, Any]] = {}

        # Internal state
        self._heap: List[Tuple[int, str]] = [
            (0, agent_id) for agent_id in self.professor_ids
        ]
        heapq.heapify(self._heap)
        self._voted_agents = set()

    # ------------------------------------------------------------------
    # Public snapshot accessors
    # ------------------------------------------------------------------

    def get_public_tickers(self) -> Dict[str, int]:
        """Return a copy of the current agent tickers."""
        return dict(self.agent_tickers)

    def get_public_waiting_state(self) -> Dict[str, Dict[str, Any]]:
        """Return a shallow copy of waiting agents state."""
        return dict(self.waiting_agents)

    # ------------------------------------------------------------------
    # Mutation helpers used by env.step
    # ------------------------------------------------------------------

    def record_action_advance(self, agent_id: str, delta_tokens: int) -> int:
        """
        Advance the ticker for *agent_id* by *delta_tokens* and update the heap.
        Returns the new ticker value.
        """
        old_ticker = self.agent_tickers.get(agent_id, 0)
        new_ticker = old_ticker + delta_tokens
        self.agent_tickers[agent_id] = new_ticker
        heapq.heappush(self._heap, (new_ticker, agent_id))
        return new_ticker

    def record_vote(self, agent_id: str) -> None:
        """Mark an agent as having voted; they will no longer be selected."""
        self._voted_agents.add(agent_id)

    def set_wait(self, agent_id: str, wait_info: Dict[str, Any]) -> None:
        """
        Register or update a wait condition for *agent_id*.

        wait_info format:
          - {"condition_type": "any_response", "wait_issued_at": ticker}
        """
        self.waiting_agents[agent_id] = dict(wait_info)

    def clear_wait(self, agent_id: str) -> None:
        """Clear any wait condition for *agent_id*."""
        self.waiting_agents.pop(agent_id, None)

    # ------------------------------------------------------------------
    # Next-agent selection
    # ------------------------------------------------------------------

    def _is_wait_satisfied(
        self,
        wait_info: Dict[str, Any],
        message_history: List[Dict[str, Any]],
    ) -> bool:
        """A wait is satisfied once any other agent acts after the wait was issued."""
        if not message_history:
            return False
        last_message = message_history[-1]
        return last_message["ticker_time"] > wait_info["wait_issued_at"]

    def get_next_agent(self, public_message_history: List[Dict[str, Any]]) -> str:
        """
        Select the next agent to speak based on minimum ticker value.

        Semantics match the original _select_next_agent logic:
        - Agents who have already voted are skipped.
        - Waiting agents either:
            - Become eligible if their wait condition is satisfied.
            - Or have their ticker "jumped" to min(others) + 1 if still waiting.
        - Deadlock guard: if every agent is waiting or has voted, all waits are
          cleared and the minimum-ticker non-voted agent is chosen.
        - Tiebreak: lexicographic order on agent_id.
        """
        # Build candidate list using the same logic as the original env.
        candidates: List[Tuple[int, str]] = []

        for agent_id, ticker in self.agent_tickers.items():
            # Skip agents who have already voted
            if agent_id in self._voted_agents:
                continue

            if agent_id in self.waiting_agents:
                wait_info = self.waiting_agents[agent_id]
                if self._is_wait_satisfied(wait_info, public_message_history):
                    # Wait satisfied -> clear and make eligible.
                    del self.waiting_agents[agent_id]
                    candidates.append((ticker, agent_id))
                else:
                    # Still waiting: jump ticker to min(others) + 1, if possible.
                    other_tickers = [
                        t
                        for aid, t in self.agent_tickers.items()
                        if aid != agent_id and aid not in self._voted_agents
                    ]
                    if other_tickers:
                        target_ticker = min(other_tickers) + 1
                        new_ticker = max(ticker, target_ticker)
                        if new_ticker != ticker:
                            self.agent_tickers[agent_id] = new_ticker
                            heapq.heappush(self._heap, (new_ticker, agent_id))
            else:
                candidates.append((ticker, agent_id))

        # Deadlock guard: if every agent is still waiting or has voted, clear waits.
        if not candidates:
            self.waiting_agents.clear()
            candidates = sorted(
                (ticker, agent_id)
                for agent_id, ticker in self.agent_tickers.items()
                if agent_id not in self._voted_agents
            )

        # If still no candidates (all have voted), return the first agent
        # (shouldn't happen in practice, but keep behavior well-defined).
        if not candidates:
            return min(self.professor_ids)

        candidates.sort()
        return candidates[0][1]
