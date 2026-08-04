"""AgingSystem: increments the agent's age once per year and enforces the
age limit.

Passive system — the agent cannot stop aging.  Removing this system means the
agent never ages and the game runs until another system kills it or the runner
hits ``max_steps``.
"""

from typing import Any, Dict, Optional

from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.types import AgentState, Action
from wealthsandbox.config import END_AGE


class AgingSystem(BaseSystem):
    """Advance the agent's age every 12 months and terminate at ``end_age``.

    Reads ``total_months`` from the macro snapshot to detect birthday months.
    """

    def __init__(self, end_age: int = END_AGE):
        """
        Args:
            end_age: Age at which the episode terminates with ``"age_limit"``.
        """
        self.end_age = end_age

    # ------------------------------------------------------------------
    # BaseSystem protocol
    # ------------------------------------------------------------------

    def tick(self, state: AgentState, macro: Dict[str, Any]) -> None:
        """No pre-action work — age increment is in ``finalize()``."""
        pass

    def handle_action(self, action: Action, state: AgentState) -> bool:
        """This system handles no agent actions."""
        return False

    def finalize(self, state: AgentState, macro: Dict[str, Any]) -> None:
        """Increment age once per 12 months.

        Called AFTER ``macro.step()``, so ``total_months`` reflects the month
        that just completed.
        """
        total_months = macro.get("total_months", 0)
        if total_months > 0 and total_months % 12 == 0:
            state.age += 1
            state.last_month_events.append(f"Turned {state.age}.")

    def check_dead(self, state: AgentState) -> Optional[str]:
        """Age limit: agent has reached or exceeded the maximum age."""
        if state.age >= self.end_age:
            return "age_limit"
        return None
