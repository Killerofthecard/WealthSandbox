"""BankSystem: savings account + loan facility, both with monthly interest.

* Deposit / withdraw — cash <-> savings, savings earn FEDFUNDS/12 per month.
* Borrow / repay — take a loan, loan charges (FEDFUNDS+2%)/12 per month.
* Loan limit: 12 × monthly after-tax income (must be employed).
* Removing this system means no savings or loan access.
"""

from typing import Any, Dict, Optional

from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.types import Action, AgentState, CareerMove, JobStatus


class BankSystem(BaseSystem):
    """Bank account with savings deposits and personal loans."""

    def __init__(self, min_cash_after_deposit: float = 2_000.0):
        self.min_cash_after_deposit = min_cash_after_deposit

    # ------------------------------------------------------------------
    # BaseSystem protocol
    # ------------------------------------------------------------------

    def tick(self, state: AgentState, macro: Dict[str, Any]) -> None:
        pass

    def handle_action(self, action: Action, state: AgentState) -> bool:
        if action.career_move == CareerMove.DEPOSIT:
            return self._handle_deposit(state, action.amount)
        elif action.career_move == CareerMove.WITHDRAW:
            return self._handle_withdraw(state, action.amount)
        elif action.career_move == CareerMove.BORROW:
            return self._handle_borrow(state, action.amount)
        elif action.career_move == CareerMove.REPAY:
            return self._handle_repay(state, action.amount)
        return False

    def finalize(self, state: AgentState, macro: Dict[str, Any]) -> None:
        fedfunds = macro.get("fedfunds", 3.0)
        # Savings interest (FEDFUNDS / 12)
        if state.savings > 0:
            monthly_save = (fedfunds / 100.0) / 12.0
            interest = state.savings * monthly_save
            if interest > 0.01:
                state.savings += interest
                state.record_flow("savings_interest", interest)
                state.last_month_events.append(
                    f"Savings interest: +${interest:,.2f} "
                    f"(rate {fedfunds:.1f}%/yr). Balance: ${state.savings:,.0f}."
                )
        # Loan interest ((FEDFUNDS + 2%) / 12)
        if state.loan_balance > 0:
            loan_rate = (fedfunds + 2.0) / 100.0 / 12.0
            interest = state.loan_balance * loan_rate
            if interest > 0.01:
                state.loan_balance += interest
                state.record_flow("loan_interest", -interest)
                state.last_month_events.append(
                    f"Loan interest: +${interest:,.2f} "
                    f"(rate {fedfunds+2.0:.1f}%/yr). Balance: ${state.loan_balance:,.0f}."
                )
            # Auto minimum repayment: 2% of balance, at least $50
            min_pay = max(state.loan_balance * 0.02, 50.0)
            actual = min(min_pay, state.loan_balance)
            # Deduct from cash, fallback to savings
            state.cash -= actual
            if state.cash < 0.0 and state.savings > 0:
                shortfall = -state.cash
                pulled = min(shortfall, state.savings)
                state.savings -= pulled
                state.cash += pulled
                state.last_month_events.append(
                    f"Auto-withdrew ${pulled:,.0f} from savings for loan payment."
                )
            state.loan_balance -= actual
            state.last_month_events.append(
                f"Loan auto-payment: ${actual:,.0f}. "
                f"Remaining: ${state.loan_balance:,.0f}."
            )

    # ------------------------------------------------------------------
    # Deposit / Withdraw
    # ------------------------------------------------------------------

    def _handle_deposit(self, state: AgentState, amount: float) -> bool:
        # Amount and buffer checks are done by the validator.
        state.cash -= amount
        state.savings += amount
        state.last_month_events.append(
            f"Deposited ${amount:,.0f} to savings. Balance: ${state.savings:,.0f}."
        )
        return True

    def _handle_withdraw(self, state: AgentState, amount: float) -> bool:
        # Amount and balance checks are done by the validator.
        state.savings -= amount
        state.cash += amount
        state.last_month_events.append(
            f"Withdrew ${amount:,.0f} from savings. Balance: ${state.savings:,.0f}."
        )
        return True

    # ------------------------------------------------------------------
    # Borrow / Repay
    # ------------------------------------------------------------------

    def _handle_borrow(self, state: AgentState, amount: float) -> bool:
        # Amount and limit checks are done by the validator.
        if state.job_status == JobStatus.EMPLOYED and state.monthly_after_tax_income > 0:
            limit = state.monthly_after_tax_income * 12.0
        else:
            limit = 8_000.0
        state.loan_balance += amount
        state.cash += amount
        state.last_month_events.append(
            f"Borrowed ${amount:,.0f} from bank. "
            f"Loan balance: ${state.loan_balance:,.0f} "
            f"(limit ${limit:,.0f})."
        )
        return True

    def _handle_repay(self, state: AgentState, amount: float) -> bool:
        # Amount checks are done by the validator.
        if amount > state.loan_balance:
            amount = state.loan_balance
        state.cash -= amount
        state.loan_balance -= amount
        state.last_month_events.append(
            f"Repaid ${amount:,.0f} of loan. "
            f"Loan balance: ${state.loan_balance:,.0f}."
        )
        return True
