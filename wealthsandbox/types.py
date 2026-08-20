"""Core type definitions for the WealthSandBox environment.

This module defines the data structures that form the contract between the Agent
and the Environment: Action, Observation, and AgentState.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CareerMove(str, Enum):
    """Career actions available to the Agent each month."""
    NONE = "none"                               # no action — auto-work continues
    SWITCH_OCCUPATION = "switch_occupation"     # switch to a different occupation
    UPSKILL = "upskill"                         # invest money to improve general skill
    QUIT_JOB = "quit_job"                       # voluntarily resign (become unemployed)
    INTENSIVE_WORK = "intensive_work"           # invest energy to improve occupation skill
    DEPOSIT = "deposit"                         # move cash to savings
    WITHDRAW = "withdraw"                       # move savings to cash
    BORROW = "borrow"                           # take a bank loan
    REPAY = "repay"                             # repay bank loan
    BUY_STOCK = "buy_stock"                     # move cash into stock index fund
    SELL_STOCK = "sell_stock"                   # sell stocks (T+1 settlement)
    REST = "rest"                               # rest to recover health + energy (costs income)
    MEDICAL_CARE = "medical_care"               # pay cash to recover health


class JobStatus(str, Enum):
    """Employment status of the Agent."""
    EMPLOYED = "employed"
    UNEMPLOYED = "unemployed"


# ---------------------------------------------------------------------------
# Tier — a career ladder rung inside an occupation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tier:
    """One rung on an occupation's career ladder.

    Promotion is automatic when the agent meets BOTH ``min_occ_skill`` and
    ``min_tenure_months`` during the monthly tick.

    Attributes:
        name: Human-readable label (e.g. "Junior", "Mid-level").
        min_occ_skill: Minimum occupation-specific skill required.
        min_tenure_months: Minimum months spent in this occupation.
        salary_multiplier: Multiplier applied to the occupation's base salary.
    """
    name: str
    min_occ_skill: int
    min_tenure_months: int
    salary_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Marketable assets
# ---------------------------------------------------------------------------

# asset_id for the S&P 500 index fund — the first (and currently only) asset
# class in the portfolio.  Adding a new asset class means a new constant + a
# new system that owns its own position key, NOT a new set of flat fields.
STOCK_INDEX = "stock_index"


@dataclass
class Position:
    """One marketable-asset holding (stock index, bonds, gold, ...).

    ``value`` is the current mark-to-market value; ``cost_basis`` is the
    cumulative net capital deployed (drives the reported profit/loss).  The
    two private fields are per-month bookkeeping for the T+1 / return rules:
    purchases this month do not earn this month's return, and sales this month
    settle next month.
    """
    value: float = 0.0
    cost_basis: float = 0.0
    last_return: float = 0.0      # this asset's return over the most recent month
    _month_buys: float = 0.0      # internal: purchases this month (excluded from return)
    _month_sells: float = 0.0     # internal: sales this month (settle next month)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Mutable per-agent state mutated in-place during a step.

    Skill is now two-dimensional:
    * ``general_skill`` — transferrable capability (upskill), governs which
      occupations the agent can enter.
    * ``occupation_skills`` — per-occupation experience (intensive_work or passive
      tenure), governs tier progression within the current job.  Not carried
      across occupation switches.
    """
    age: int = 20
    health: float = 1.0
    energy: float = 1.0
    cash: float = 0.0
    savings: float = 0.0              # bank savings balance
    loan_balance: float = 0.0         # bank loan balance (owed)

    # Two-dimensional skill
    general_skill: int = 1
    occupation_skills: Dict[str, int] = field(default_factory=dict)

    # Current occupation
    occupation_id: str = ""
    prev_occupation_id: str = ""     # last occupation before layoff / forced resign
    job_status: JobStatus = JobStatus.UNEMPLOYED
    tenure_months: int = 0           # months spent in current occupation
    unemployed_months: int = 0       # consecutive months unemployed
    current_tier: int = 0            # index into the occupation's tier list

    # Income
    monthly_after_tax_income: float = 0.0

    # Training / upskilling timers
    upskill_months_remaining: int = 0
    intensive_work_months_remaining: int = 0
    training_months_remaining: int = 0
    training_target_occupation: str = ""

    # Wellbeing (health recovery)
    resting_this_month: bool = False        # set by `rest`; reduces this month's income
    medical_care_uses_this_year: int = 0    # medical_care uses in the current year

    # Marketable assets — a portfolio of positions keyed by asset_id (e.g.
    # STOCK_INDEX).  Adding a new asset class (bonds, gold, real estate) is a
    # new dict entry + a system, not a new set of flat fields on this state.
    positions: Dict[str, Position] = field(default_factory=dict)
    pending_settlement: float = 0.0            # T+1 sale proceeds, available next month

    # History / events
    career_history: List[Dict[str, Any]] = field(default_factory=list)
    last_month_events: List[str] = field(default_factory=list)

    # Cash-flow ledger (reward decomposition)
    monthly_flow: Dict[str, float] = field(default_factory=dict)     # this month's component flows
    cumulative_flow: Dict[str, float] = field(default_factory=dict)  # running totals across months

    def record_flow(self, key: str, amount: float) -> None:
        """Add *amount* to this month's flow ledger under *key*.

        ``amount`` is a net-worth delta: positive for inflows (income,
        interest, stock gains), negative for outflows (expenses, costs,
        interest paid, losses).  Balance-neutral transfers (deposit/withdraw,
        borrow/repay, buy/sell) are deliberately NOT recorded.
        """
        self.monthly_flow[key] = self.monthly_flow.get(key, 0.0) + amount


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

def get_position(state: AgentState, asset_id: str) -> Optional[Position]:
    """Return the agent's position in *asset_id*, or ``None`` if they hold none."""
    return state.positions.get(asset_id)


def net_worth(state: AgentState) -> float:
    """Total wealth = cash + savings + marketable assets + pending settlement − debt.

    This is the single source of truth for bankruptcy checks, scoring, and any
    reported net worth.  New asset classes are automatically included because
    it sums the whole ``positions`` portfolio.
    """
    assets = sum(p.value for p in state.positions.values())
    return (
        state.cash
        + state.savings
        + assets
        + state.pending_settlement
        - state.loan_balance
    )


@dataclass
class Action:
    """Standardised action submitted by the Agent each month.

    Attributes:
        career_move: The career action to take this month.
        target_occupation_id: Required when career_move is SWITCH_OCCUPATION.
    """
    career_move: CareerMove = CareerMove.NONE
    target_occupation_id: str = ""
    amount: float = 0.0


@dataclass
class Observation:
    """Standardised observation returned to the Agent each month.

    Attributes:
        individual: Snapshot of the AgentState as a plain dict.
        macro: Snapshot of visible macro variables as a plain dict.
        narrative: Natural-language summary of the month (useful for LLM Agents).
        month: Elapsed months since the start of the episode (1-based).
        done: Whether the episode has terminated.
        info: Diagnostic metadata.
    """
    individual: Dict[str, Any] = field(default_factory=dict)
    macro: Dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    month: int = 0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
