"""ActionValidator: centralized guard layer for agent action legitimacy."""

from dataclasses import dataclass
from typing import Callable, Dict, List

from wealthsandbox.types import Action, AgentState, CareerMove, JobStatus


# ---------------------------------------------------------------------------
# GuardResult
# ---------------------------------------------------------------------------

@dataclass
class GuardResult:
    allowed: bool
    event_key: str = ""
    message: str = ""

    @classmethod
    def ok(cls) -> "GuardResult":
        return cls(allowed=True)

    @classmethod
    def reject(cls, event_key: str, message: str) -> "GuardResult":
        return cls(allowed=False, event_key=event_key, message=message)


# ---------------------------------------------------------------------------
# Guard functions
# ---------------------------------------------------------------------------

def guard_switch_occupation(
    state: AgentState,
    career,
) -> GuardResult:
    if state.training_months_remaining > 0:
        return GuardResult.reject(
            "switch_rejected_already_training",
            f"You are already training for {state.training_target_occupation} "
            f"({state.training_months_remaining} months remaining).",
        )
    return GuardResult.ok()


def guard_switch_occupation_target(
    state: AgentState,
    career,
    target_id: str,
) -> GuardResult:
    if not target_id:
        return GuardResult.reject(
            "switch_occupation_no_target",
            "No target occupation specified.",
        )

    try:
        occ = career.get_occupation(target_id)
    except ValueError:
        return GuardResult.reject(
            "switch_occupation_invalid",
            f"'{target_id}' is not a valid occupation.",
        )

    # General skill gate
    if state.general_skill < occ.min_general_skill:
        return GuardResult.reject(
            "switch_rejected_skill_too_low",
            f"Your general skill ({state.general_skill}) is too low for "
            f"{occ.display_name} (requires {occ.min_general_skill}).",
        )

    # Health gate
    if state.health < occ.min_health:
        return GuardResult.reject(
            "switch_rejected_health_too_low",
            f"Your health ({state.health:.3f}) is too low for "
            f"{occ.display_name} (requires {occ.min_health:.1f}). "
            f"Consider a less physically demanding occupation.",
        )

    # Cash gate — manufacturing_worker is the safety-net occupation:
    # always free to join, even with $0 cash (no switch cost, no buffer).
    if target_id != "manufacturing_worker":
        total_cost = career.switch_base_cost + occ.entry_cost
        required = total_cost + career.living_expense
        if state.cash < required:
            return GuardResult.reject(
                "switch_rejected_insufficient_cash",
                f"Insufficient cash to switch to {occ.display_name}: "
                f"need ${total_cost:,} + ${career.living_expense:,.0f} living = "
                f"${required:,}, have ${state.cash:,.0f}.",
            )

    return GuardResult.ok()


def guard_upskill(state: AgentState, career) -> GuardResult:
    if state.general_skill >= career.max_general_skill:
        return GuardResult.reject(
            "upskill_rejected_at_max_skill",
            f"Already at maximum general skill ({career.max_general_skill}).",
        )
    if state.upskill_months_remaining > 0:
        return GuardResult.reject(
            "upskill_rejected_already_in_progress",
            f"Already upskilling ({state.upskill_months_remaining} months remaining).",
        )
    required = career.upskill_cost + career.living_expense
    if state.cash < required:
        return GuardResult.reject(
            "upskill_rejected_insufficient_cash",
            f"Insufficient cash to upskill: need ${career.upskill_cost:,} + "
            f"${career.living_expense:,.0f} living = ${required:,}, "
            f"have ${state.cash:,.0f}.",
        )
    return GuardResult.ok()


def guard_intensive_work(state: AgentState, career) -> GuardResult:
    """Check whether the agent can start intensive work."""
    if state.job_status != JobStatus.EMPLOYED:
        return GuardResult.reject(
            "intensive_work_rejected_not_employed",
            "You must be employed to do intensive work.",
        )
    occ_skill = state.occupation_skills.get(state.occupation_id, 1)
    if occ_skill >= career.max_occ_skill:
        return GuardResult.reject(
            "intensive_work_rejected_at_max_occ_skill",
            f"Already at maximum occupation skill ({career.max_occ_skill}).",
        )
    if state.intensive_work_months_remaining > 0:
        return GuardResult.reject(
            "intensive_work_rejected_already_in_progress",
            f"Already doing intensive work "
            f"({state.intensive_work_months_remaining} months remaining).",
        )
    return GuardResult.ok()


def guard_quit_job(state: AgentState, career) -> GuardResult:
    if state.job_status != JobStatus.EMPLOYED:
        return GuardResult.reject(
            "quit_rejected_not_employed",
            "You are not currently employed.",
        )
    return GuardResult.ok()


def guard_deposit(state: AgentState, career) -> GuardResult:
    if state.cash < 2_000.0:
        return GuardResult.reject(
            "deposit_rejected_low_cash",
            f"Not enough cash to deposit while keeping $2,000 for expenses.",
        )
    return GuardResult.ok()


def guard_withdraw(state: AgentState, career) -> GuardResult:
    if state.savings <= 0:
        return GuardResult.reject(
            "withdraw_rejected_no_savings",
            "No savings to withdraw.",
        )
    return GuardResult.ok()


def guard_borrow(state: AgentState, career) -> GuardResult:
    # Limit: if employed, 12× income. If unemployed, flat $8,000.
    if state.job_status == JobStatus.EMPLOYED and state.monthly_after_tax_income > 0:
        limit = state.monthly_after_tax_income * 12.0
    else:
        limit = 8_000.0
    if state.loan_balance >= limit:
        return GuardResult.reject(
            "borrow_rejected_at_limit",
            f"Already at loan limit (${limit:,.0f}).",
        )
    return GuardResult.ok()


def guard_repay(state: AgentState, career) -> GuardResult:
    if state.loan_balance <= 0:
        return GuardResult.reject(
            "repay_rejected_no_loan",
            "No outstanding loan to repay.",
        )
    if state.cash <= 0:
        return GuardResult.reject(
            "repay_rejected_no_cash",
            "No cash to repay with.",
        )
    return GuardResult.ok()


# ---------------------------------------------------------------------------
# Amount-specific bank guards (parameter validation, used on actual tool calls)
# ---------------------------------------------------------------------------

def guard_deposit_amount(
    state: AgentState, career, amount: float
) -> GuardResult:
    """Validate the specific deposit *amount* (generic guard already checked cash ≥ buffer)."""
    if amount <= 0:
        return GuardResult.reject(
            "deposit_rejected_amount_zero",
            "Deposit amount must be greater than zero.",
        )
    if amount > state.cash:
        return GuardResult.reject(
            "deposit_rejected_amount_exceeds_cash",
            f"Cannot deposit ${amount:,.0f} — only have ${state.cash:,.0f} cash.",
        )
    buffer = getattr(career, "living_expense", 2_000.0)
    if state.cash - amount < buffer:
        return GuardResult.reject(
            "deposit_rejected_buffer",
            f"Cannot deposit ${amount:,.0f} — need at least "
            f"${buffer:,.0f} remaining for living expenses.",
        )
    return GuardResult.ok()


def guard_withdraw_amount(
    state: AgentState, career, amount: float
) -> GuardResult:
    """Validate the specific withdraw *amount*."""
    if amount <= 0:
        return GuardResult.reject(
            "withdraw_rejected_amount_zero",
            "Withdrawal amount must be greater than zero.",
        )
    if amount > state.savings:
        return GuardResult.reject(
            "withdraw_rejected_amount_exceeds_savings",
            f"Cannot withdraw ${amount:,.0f} — savings only has ${state.savings:,.0f}.",
        )
    return GuardResult.ok()


def guard_borrow_amount(
    state: AgentState, career, amount: float
) -> GuardResult:
    """Validate the specific borrow *amount*."""
    if amount <= 0:
        return GuardResult.reject(
            "borrow_rejected_amount_zero",
            "Loan amount must be greater than zero.",
        )
    from wealthsandbox.types import JobStatus
    if state.job_status == JobStatus.EMPLOYED and state.monthly_after_tax_income > 0:
        limit = state.monthly_after_tax_income * 12.0
    else:
        limit = 8_000.0
    if state.loan_balance + amount > limit:
        return GuardResult.reject(
            "borrow_rejected_exceeds_limit",
            f"Cannot borrow ${amount:,.0f} — loan limit is "
            f"${limit:,.0f}. Current loan: ${state.loan_balance:,.0f}.",
        )
    return GuardResult.ok()


def guard_repay_amount(
    state: AgentState, career, amount: float
) -> GuardResult:
    """Validate the specific repay *amount*."""
    if amount <= 0:
        return GuardResult.reject(
            "repay_rejected_amount_zero",
            "Repayment amount must be greater than zero.",
        )
    if amount > state.cash:
        return GuardResult.reject(
            "repay_rejected_amount_exceeds_cash",
            f"Cannot repay ${amount:,.0f} — only have ${state.cash:,.0f} cash.",
        )
    if amount > state.loan_balance:
        return GuardResult.reject(
            "repay_rejected_amount_exceeds_loan",
            f"Cannot repay ${amount:,.0f} — loan balance is only ${state.loan_balance:,.0f}.",
        )
    return GuardResult.ok()


# ---------------------------------------------------------------------------
# Stock guards
# ---------------------------------------------------------------------------

def guard_buy_stock(state: AgentState, career) -> GuardResult:
    """Can the agent buy stocks?  Must have enough cash to keep a buffer."""
    buffer = getattr(career, "living_expense", 2_000.0)
    if state.cash <= buffer:
        return GuardResult.reject(
            "buy_stock_rejected_no_cash",
            f"Not enough cash to invest while keeping ${buffer:,.0f} for expenses.",
        )
    return GuardResult.ok()


def guard_buy_stock_amount(
    state: AgentState, career, amount: float
) -> GuardResult:
    """Validate the specific buy *amount*."""
    buffer = getattr(career, "living_expense", 2_000.0)
    if amount <= 0:
        return GuardResult.reject(
            "buy_stock_rejected_amount_zero",
            "Investment amount must be greater than zero.",
        )
    if amount > state.cash:
        return GuardResult.reject(
            "buy_stock_rejected_amount_exceeds_cash",
            f"Cannot invest ${amount:,.0f} — only have ${state.cash:,.0f} cash.",
        )
    if state.cash - amount < buffer:
        return GuardResult.reject(
            "buy_stock_rejected_buffer",
            f"Cannot invest ${amount:,.0f} — need at least "
            f"${buffer:,.0f} remaining for living expenses.",
        )
    return GuardResult.ok()


def guard_sell_stock(state: AgentState, career) -> GuardResult:
    """Can the agent sell stocks?  Must have stock_value > 0."""
    if state.stock_value <= 0:
        return GuardResult.reject(
            "sell_stock_rejected_no_stocks",
            "No stocks to sell — stock_value is $0.",
        )
    return GuardResult.ok()


def guard_sell_stock_amount(
    state: AgentState, career, amount: float
) -> GuardResult:
    """Validate the specific sell *amount*."""
    if amount <= 0:
        return GuardResult.reject(
            "sell_stock_rejected_amount_zero",
            "Sell amount must be greater than zero.",
        )
    if amount > state.stock_value:
        return GuardResult.reject(
            "sell_stock_rejected_amount_exceeds_holdings",
            f"Cannot sell ${amount:,.0f} — stock holdings only worth "
            f"${state.stock_value:,.0f}.",
        )
    return GuardResult.ok()


# ---------------------------------------------------------------------------
# ActionValidator
# ---------------------------------------------------------------------------

class ActionValidator:
    """Central registry of all action guard functions."""

    def __init__(self, career, energy_threshold: float = 0.4):
        self._career = career
        self._guards: Dict[CareerMove, List[Callable]] = {}

        self.register(CareerMove.SWITCH_OCCUPATION, guard_switch_occupation)
        self.register(CareerMove.UPSKILL, guard_upskill)
        self.register(CareerMove.INTENSIVE_WORK, guard_intensive_work)
        self.register(CareerMove.QUIT_JOB, guard_quit_job)
        self.register(CareerMove.DEPOSIT, guard_deposit)
        self.register(CareerMove.WITHDRAW, guard_withdraw)
        self.register(CareerMove.BORROW, guard_borrow)
        self.register(CareerMove.REPAY, guard_repay)
        self.register(CareerMove.BUY_STOCK, guard_buy_stock)
        self.register(CareerMove.SELL_STOCK, guard_sell_stock)

        # Energy gate for upskill and intensive_work
        threshold = energy_threshold

        def guard_energy(state, career):
            if state.energy < threshold:
                return GuardResult.reject(
                    "rejected_energy_too_low",
                    f"Not enough energy: need ≥{threshold:.0%}, "
                    f"have {state.energy:.0%}.  Rest (do nothing) to recover.",
                )
            return GuardResult.ok()

        self.register(CareerMove.UPSKILL, guard_energy)
        self.register(CareerMove.INTENSIVE_WORK, guard_energy)

    def register(self, action: CareerMove, guard_fn: Callable) -> None:
        self._guards.setdefault(action, []).append(guard_fn)

    def validate(self, action: Action, state: AgentState) -> GuardResult:
        guards = self._guards.get(action.career_move, [])
        for fn in guards:
            result = fn(state, self._career)
            if not result.allowed:
                return result
        return GuardResult.ok()

    def validate_switch_target(
        self, action: Action, state: AgentState
    ) -> GuardResult:
        result = self.validate(action, state)
        if not result.allowed:
            return result
        return guard_switch_occupation_target(
            state, self._career, action.target_occupation_id
        )

    def validate_deposit(self, action: Action, state: AgentState) -> GuardResult:
        result = self.validate(action, state)
        if not result.allowed:
            return result
        return guard_deposit_amount(state, self._career, action.amount)

    def validate_withdraw(self, action: Action, state: AgentState) -> GuardResult:
        result = self.validate(action, state)
        if not result.allowed:
            return result
        return guard_withdraw_amount(state, self._career, action.amount)

    def validate_borrow(self, action: Action, state: AgentState) -> GuardResult:
        result = self.validate(action, state)
        if not result.allowed:
            return result
        return guard_borrow_amount(state, self._career, action.amount)

    def validate_repay(self, action: Action, state: AgentState) -> GuardResult:
        result = self.validate(action, state)
        if not result.allowed:
            return result
        return guard_repay_amount(state, self._career, action.amount)

    def validate_buy_stock(self, action: Action, state: AgentState) -> GuardResult:
        result = self.validate(action, state)
        if not result.allowed:
            return result
        return guard_buy_stock_amount(state, self._career, action.amount)

    def validate_sell_stock(self, action: Action, state: AgentState) -> GuardResult:
        result = self.validate(action, state)
        if not result.allowed:
            return result
        return guard_sell_stock_amount(state, self._career, action.amount)

    def available_actions(self, state: AgentState) -> Dict[str, dict]:
        result: Dict[str, dict] = {}
        for move in self._guards:
            action = Action(career_move=move)
            r = self.validate(action, state)
            result[move.value] = {
                "allowed": r.allowed,
                "reason": "" if r.allowed else r.message,
            }
        return result
