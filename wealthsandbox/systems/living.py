"""LivingExpenseSystem: deducts monthly living expenses from agent cash.

Passive system — the agent cannot prevent or alter the deduction.
"""

from typing import Any, Dict, Optional

from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.types import AgentState, Action
from wealthsandbox.config import MONTHLY_LIVING_EXPENSE


class LivingExpenseSystem(BaseSystem):
    """Deduct a fixed living expense from cash every month."""

    def __init__(self, monthly_living_expense: float = MONTHLY_LIVING_EXPENSE):
        self.monthly_living_expense = monthly_living_expense

    def tick(self, state: AgentState, macro: Dict[str, Any]) -> None:
        pass

    def handle_action(self, action: Action, state: AgentState) -> bool:
        return False

    def finalize(self, state: AgentState, macro: Dict[str, Any]) -> None:
        state.cash -= self.monthly_living_expense
        if state.cash < 0.0:
            state.last_month_events.append("Cannot afford living expenses!")
        else:
            state.last_month_events.append(
                f"Paid ${self.monthly_living_expense:,.0f} living expenses."
            )

    def check_dead(self, state: AgentState) -> Optional[str]:
        if state.cash <= 0.0:
            return "bankruptcy"
        return None
