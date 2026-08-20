"""AssetSystem: stock index fund investment with T+1 settlement.

* ``buy_stock(amount)`` — move cash into stocks.  Purchases do NOT earn the
  current month's return.
* ``sell_stock(amount)`` — sell holdings.  Proceeds settle NEXT month.
* ``force_liquidate(shortfall)`` — emergency liquidation at a discount
  (called by LivingExpenseSystem when cash + savings are exhausted).
* Stocks are never auto-liquidated — the agent must sell or the forced-sale
  safety net kicks in during finalize.

The stock fund is one ``Position`` in ``AgentState.positions``, keyed by
``STOCK_INDEX``.  Adding a new asset class (bonds, gold, ...) means a new
system that owns its own position key — ``net_worth`` (types.py) already sums
the whole portfolio, so no flat fields or net-worth edits are needed.
"""

from typing import Any, Dict, Optional

from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.types import (
    Action, AgentState, CareerMove, Position, STOCK_INDEX, net_worth,
)


def stock_position(state: AgentState) -> Optional[Position]:
    """Return the agent's stock position, or ``None`` if they hold none yet."""
    return state.positions.get(STOCK_INDEX)


def _ensure_stock(state: AgentState) -> Position:
    """Return the stock position, creating an empty one if needed."""
    return state.positions.setdefault(STOCK_INDEX, Position())


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
        2. Apply market return to pre-existing value only
           (purchases made this month in handle_action don't earn return).
        3. Add this month's purchases at cost.
        4. Record the month's return for observation display.
        """
        # 1. T+1 settlement: sale proceeds from LAST month arrive NOW.
        #    This month's sales (via the position's _month_sells) are deferred
        #    until finalize() → they settle NEXT month.
        if state.pending_settlement > 0:
            settled = state.pending_settlement
            state.cash += settled
            state.last_month_events.append(
                f"Stock sale settled: +${settled:,.0f} in cash."
            )
            state.pending_settlement = 0.0

        pos = stock_position(state)
        if pos is None:
            # Never held stocks — nothing to mark to market or record.
            return

        sp500_tr = macro.get("sp500_tr", 0.0)

        # 2. Separate this month's new purchases from pre-existing holdings.
        new_purchases = pos._month_buys
        pre_existing = pos.value - new_purchases

        # 3. Apply market return to pre-existing holdings only.
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

        # 4. Recombine: pre-existing (with return) + new purchases (at cost).
        pos.value = pre_existing + new_purchases
        pos._month_buys = 0.0

        # 5. Record for observation display.
        pos.last_return = sp500_tr

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
        pos = stock_position(state)
        if pos is None:
            return
        new_sales = pos._month_sells
        if new_sales > 0:
            state.pending_settlement += new_sales
            pos._month_sells = 0.0

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_buy(self, state: AgentState, amount: float) -> bool:
        """Move cash into stock index fund.

        Purchases do NOT earn this month's return — the amount is tracked
        separately and added AFTER mark-to-market in tick().
        """
        pos = _ensure_stock(state)
        state.cash -= amount
        pos.value += amount
        pos.cost_basis += amount
        pos._month_buys += amount
        state.last_month_events.append(
            f"Invested ${amount:,.0f} in a stock index fund."
        )
        return True

    def _handle_sell(self, state: AgentState, amount: float) -> bool:
        """Sell stocks.  Proceeds settle NEXT month (T+1).

        The sale amount is stored in the position's _month_sells and moved to
        pending_settlement in finalize(), so tick() (which runs after
        handle_action in the same month) does NOT process it immediately.

        cost_basis is reduced proportionally.
        """
        pos = _ensure_stock(state)

        # Clamp to actual holdings (guard already checks, but safety net).
        amount = min(amount, pos.value)

        # Proportionally reduce cost_basis.
        if pos.value > 0 and pos.cost_basis > 0:
            fraction = amount / pos.value
            pos.cost_basis -= pos.cost_basis * fraction
            # Keep the return base consistent with the proportional cost basis:
            # this month's purchases shrink in the same proportion, so tick()'s
            # `pre_existing = value - _month_buys` never over-subtracts.
            if pos._month_buys > 0:
                pos._month_buys -= pos._month_buys * fraction
        pos.cost_basis = max(0.0, pos.cost_basis)

        pos.value -= amount
        # Defer to finalize() so same-month tick() doesn't settle it.
        pos._month_sells += amount
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
        pos = stock_position(state)
        if pos is None or pos.value <= 0 or shortfall <= 0:
            return 0.0

        discount = self.forced_sale_discount
        # Each $1 of stock liquidates at $(1 - discount)
        # To raise $X we need to sell $X / (1 - discount) worth of stock
        max_raised = pos.value * (1.0 - discount)
        actual_raised = min(max_raised, shortfall)

        # How much stock value was consumed
        stock_consumed = actual_raised / (1.0 - discount)

        # Proportionally reduce cost_basis.
        if pos.value > 0 and pos.cost_basis > 0:
            fraction = stock_consumed / pos.value
            pos.cost_basis -= pos.cost_basis * fraction
        pos.cost_basis = max(0.0, pos.cost_basis)

        pos.value -= stock_consumed
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

        Net worth = cash + savings + portfolio value + pending_settlement
        - loan_balance (see ``types.net_worth``).
        """
        if net_worth(state) <= 0.0:
            return "bankruptcy"
        return None
