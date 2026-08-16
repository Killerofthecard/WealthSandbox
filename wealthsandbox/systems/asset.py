"""AssetSystem: stock index fund investment with T+1 settlement.

* ``buy_stock(amount)`` — move cash into stocks.  Purchases do NOT earn the
  current month's return.
* ``sell_stock(amount)`` — sell holdings.  Proceeds settle NEXT month.
* ``force_liquidate(shortfall)`` — emergency liquidation at a discount
  (called by LivingExpenseSystem when cash + savings are exhausted).
* Stocks are never auto-liquidated — the agent must sell or the forced-sale
  safety net kicks in during finalize.
"""

from typing import Any, Dict, List, Optional

from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.types import Action, AgentState, CareerMove


class AssetSystem(BaseSystem):
    """Manages stock index fund holdings with monthly mark-to-market.

    * Purchases made this month do NOT earn this month's return.  New money
      is added AFTER the mark-to-market step.
    * Sales settle T+1: proceeds arrive in cash at the start of next month.
    """

    def __init__(self, forced_sale_discount: float = 0.10):
        self.forced_sale_discount = forced_sale_discount

    # ------------------------------------------------------------------
    # BaseSystem protocol
    # ------------------------------------------------------------------

    def tick(self, state: AgentState, macro: Dict[str, Any]) -> None:
        """Month-start processing: settle pending sales from LAST month,
        apply market return to EXISTING holdings, then add this month's
        purchases.

        1. Settle pending sales (T+1 from *last* month's sales) → cash.
        2. Apply market return to pre-existing stock_value only
           (purchases made this month in handle_action don't earn return).
        3. Add this month's purchases at cost.
        4. Record last_month_stock_return.
        """
        # 1. T+1 settlement: sale proceeds from LAST month arrive NOW.
        #    This month's sales (via _this_month_stock_sales) are deferred
        #    until finalize() → they settle NEXT month.
        if state.pending_settlement > 0:
            settled = state.pending_settlement
            state.cash += settled
            state.last_month_events.append(
                f"Stock sale settled: +${settled:,.0f} in cash."
            )
            state.pending_settlement = 0.0

        sp500_tr = macro.get("sp500_tr", 0.0)

        # 2. Separate this month's new purchases from pre-existing holdings.
        new_purchases = getattr(state, "_this_month_stock_purchases", 0.0)
        pre_existing = state.stock_value - new_purchases

        # 3. Apply market return to pre-existing holdings only
        if pre_existing > 0:
            old_value = pre_existing
            pre_existing *= (1.0 + sp500_tr)
            change = pre_existing - old_value
            state.record_flow("stock_pnl", change)
            pct = sp500_tr * 100.0
            direction = "gained" if change >= 0 else "lost"
            state.last_month_events.append(
                f"Your stock holdings {direction} ${abs(change):,.0f} "
                f"({pct:+.1f}%) this month."
            )

        # 4. Recombine: pre-existing (with return) + new purchases (at cost)
        state.stock_value = pre_existing + new_purchases
        state._this_month_stock_purchases = 0.0

        # 5. Record for observation display
        state.last_month_stock_return = sp500_tr

    def handle_action(self, action: Action, state: AgentState) -> bool:
        """Dispatch BUY_STOCK / SELL_STOCK."""
        if action.career_move == CareerMove.BUY_STOCK:
            return self._handle_buy(state, action.amount)
        elif action.career_move == CareerMove.SELL_STOCK:
            return self._handle_sell(state, action.amount)
        return False

    def finalize(self, state: AgentState, macro: Dict[str, Any]) -> None:
        """Move this month's sales to pending_settlement for NEXT month's tick.

        This ensures T+1: sales made in Phase 1 (execute) are NOT settled
        in Phase 2 (tick) of the same month.
        """
        new_sales = getattr(state, "_this_month_stock_sales", 0.0)
        if new_sales > 0:
            state.pending_settlement += new_sales
            state._this_month_stock_sales = 0.0

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_buy(self, state: AgentState, amount: float) -> bool:
        """Move cash into stock index fund.

        Purchases do NOT earn this month's return — the amount is tracked
        separately and added AFTER mark-to-market in tick().
        """
        state.cash -= amount
        state.stock_value += amount
        state.total_invested += amount
        # Track this as a new purchase so tick() doesn't apply
        # this month's return to it.
        prev = getattr(state, "_this_month_stock_purchases", 0.0)
        state._this_month_stock_purchases = prev + amount
        state.last_month_events.append(
            f"Invested ${amount:,.0f} in a stock index fund."
        )
        return True

    def _handle_sell(self, state: AgentState, amount: float) -> bool:
        """Sell stocks.  Proceeds settle NEXT month (T+1).

        The sale amount is stored in a temporary field and moved to
        pending_settlement in finalize(), so tick() (which runs after
        handle_action in the same month) does NOT process it immediately.

        total_invested is reduced proportionally.
        """
        # Clamp to actual stock_value (guard already checks, but safety net)
        amount = min(amount, state.stock_value)

        # Proportionally reduce total_invested
        if state.stock_value > 0 and state.total_invested > 0:
            fraction = amount / state.stock_value
            state.total_invested -= state.total_invested * fraction
        state.total_invested = max(0.0, state.total_invested)

        state.stock_value -= amount
        # Defer to finalize() so same-month tick() doesn't settle it
        prev = getattr(state, "_this_month_stock_sales", 0.0)
        state._this_month_stock_sales = prev + amount
        state.last_month_events.append(
            f"Sold ${amount:,.0f} of stocks. Funds will be available next month."
        )
        return True

    # ------------------------------------------------------------------
    # Force liquidation (called by LivingExpenseSystem)
    # ------------------------------------------------------------------

    def force_liquidate(
        self,
        state: AgentState,
        shortfall: float,
    ) -> float:
        """Emergency stock sale to cover a cash shortfall.

        Args:
            state: Current agent state.
            shortfall: Cash needed (positive number).

        Returns:
            Cash raised (may be less than *shortfall* if stocks don't cover it).
        """
        if state.stock_value <= 0 or shortfall <= 0:
            return 0.0

        discount = self.forced_sale_discount
        # Each $1 of stock liquidates at $(1 - discount)
        # To raise $X we need to sell $X / (1 - discount) worth of stock
        max_raised = state.stock_value * (1.0 - discount)
        actual_raised = min(max_raised, shortfall)

        # How much stock value was consumed
        stock_consumed = actual_raised / (1.0 - discount)

        # Proportionally reduce total_invested
        if state.stock_value > 0 and state.total_invested > 0:
            fraction = stock_consumed / state.stock_value
            state.total_invested -= state.total_invested * fraction
        state.total_invested = max(0.0, state.total_invested)

        state.stock_value -= stock_consumed
        state.cash += actual_raised
        # The discount is a real loss: net worth drops by (stock_consumed - actual_raised).
        state.record_flow("forced_sale_loss", actual_raised - stock_consumed)

        state.last_month_events.append(
            f"EMERGENCY: Forced to sell stocks at a {discount:.0%} loss, "
            f"raising ${actual_raised:,.0f}."
        )
        return actual_raised

    # ------------------------------------------------------------------
    # Bankruptcy / death check
    # ------------------------------------------------------------------

    def check_dead(self, state: AgentState) -> Optional[str]:
        """Net worth ≤ 0 → bankruptcy.

        Net worth = cash + savings + stock_value + pending_settlement - loan_balance.
        """
        net_worth = (
            state.cash
            + state.savings
            + state.stock_value
            + state.pending_settlement
            - state.loan_balance
        )
        if net_worth <= 0.0:
            return "bankruptcy"
        return None
