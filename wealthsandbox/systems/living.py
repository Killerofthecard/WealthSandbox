"""LivingExpenseSystem: deducts monthly living expenses from agent cash.

Passive system — the agent cannot prevent or alter the deduction.
Shortfall is automatically covered from savings.  Bankruptcy = cash + savings ≤ 0.
"""

from typing import Any, Dict, Optional

from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.types import AgentState, Action
from wealthsandbox.config import MONTHLY_LIVING_EXPENSE


class LivingExpenseSystem(BaseSystem):
    """Deduct a fixed living expense from cash every month.

    If cash can't cover the expense, savings are auto-withdrawn.
    Bankruptcy only when both cash and savings are exhausted.
    """

    def __init__(self, monthly_living_expense: float = MONTHLY_LIVING_EXPENSE):
        self.monthly_living_expense = monthly_living_expense

    def tick(self, state: AgentState, macro: Dict[str, Any]) -> None:
        pass

    def handle_action(self, action: Action, state: AgentState) -> bool:
        return False

    def finalize(self, state: AgentState, macro: Dict[str, Any]) -> None:
        state.cash -= self.monthly_living_expense
        # Auto-cover shortfall from savings
        if state.cash < 0.0 and state.savings > 0:
            shortfall = -state.cash
            pulled = min(shortfall, state.savings)
            state.savings -= pulled
            state.cash += pulled
            state.last_month_events.append(
                f"Auto-withdrew ${pulled:,.0f} from savings to cover living expenses."
            )
        if state.cash >= 0.0:
            state.last_month_events.append(
                f"Paid ${self.monthly_living_expense:,.0f} living expenses."
            )
        else:
            state.last_month_events.append("Cannot afford living expenses!")

    def check_dead(self, state: AgentState) -> Optional[str]:
        """Bankruptcy: cash + savings at or below zero."""
        if state.cash + state.savings <= 0.0:
            return "bankruptcy"
        return None
