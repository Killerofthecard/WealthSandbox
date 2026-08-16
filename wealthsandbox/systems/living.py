"""LivingExpenseSystem: deducts monthly living expenses from agent cash.

Passive system — the agent cannot prevent or alter the deduction.
Shortfall is covered from savings first, then forced stock liquidation.
Bankruptcy is determined by AssetSystem.check_dead (net worth).
"""

from typing import Any, Dict, Optional

from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.types import AgentState, Action
from wealthsandbox.config import MONTHLY_LIVING_EXPENSE


class LivingExpenseSystem(BaseSystem):
    """Deduct a fixed living expense from cash every month.

    If cash can't cover the expense, savings are auto-withdrawn first,
    then stocks are force-liquidated at a discount.  Bankruptcy is handled
    by AssetSystem (net worth ≤ 0).
    """

    def __init__(
        self,
        monthly_living_expense: float = MONTHLY_LIVING_EXPENSE,
        asset_system: Optional[object] = None,
    ):
        self.monthly_living_expense = monthly_living_expense
        self._asset_system = asset_system

    def tick(self, state: AgentState, macro: Dict[str, Any]) -> None:
        pass

    def handle_action(self, action: Action, state: AgentState) -> bool:
        return False

    def finalize(self, state: AgentState, macro: Dict[str, Any]) -> None:
        # Nominal living expense grows with the price level (CPI-driven).
        price_level = macro.get("price_level", 1.0)
        expense = self.monthly_living_expense * price_level
        state.cash -= expense
        state.record_flow("living_expense", -expense)

        # Auto-cover shortfall from savings first
        if state.cash < 0.0 and state.savings > 0:
            shortfall = -state.cash
            pulled = min(shortfall, state.savings)
            state.savings -= pulled
            state.cash += pulled
            state.last_month_events.append(
                f"Auto-withdrew ${pulled:,.0f} from savings to cover living expenses."
            )

        # If still short, force-liquidate stocks
        if state.cash < 0.0 and self._asset_system is not None:
            shortfall = -state.cash
            self._asset_system.force_liquidate(state, shortfall)

        if state.cash >= 0.0:
            state.last_month_events.append(
                f"Paid ${expense:,.0f} living expenses."
            )
        else:
            state.last_month_events.append(
                f"Cannot afford living expenses! Cash: ${state.cash:,.2f}"
            )

    def check_dead(self, state: AgentState) -> Optional[str]:
        """Bankruptcy is now handled by AssetSystem.check_dead (net worth)."""
        return None
