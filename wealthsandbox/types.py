"""Core type definitions for the WealthSandBox environment.

This module defines the data structures that form the contract between the Agent
and the Environment: Action, Observation, and AgentState.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
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

    # History / events
    career_history: List[Dict[str, Any]] = field(default_factory=list)
    last_month_events: List[str] = field(default_factory=list)


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
        year: Current calendar year.
        month: Current month (1-12).
        done: Whether the episode has terminated.
        info: Diagnostic metadata.
    """
    individual: Dict[str, Any] = field(default_factory=dict)
    macro: Dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    year: int = 2024
    month: int = 1
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
