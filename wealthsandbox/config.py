"""Configuration constants and the EnvConfig dataclass for WealthSandBox.

Agent-specific initial attributes have been extracted to ``AgentProfile``
(see ``wealthsandbox.profile``).  ``EnvConfig`` now holds only **environment
rules** — costs, tax rates, limits — that are the same regardless of which
agent is playing.
"""

from dataclasses import dataclass, field
from typing import Optional

from wealthsandbox.profile import AgentProfile


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
END_AGE = 60                     # default age at which the episode terminates
START_YEAR = 2024
START_MONTH = 1

# ---------------------------------------------------------------------------
# Career / skill
# ---------------------------------------------------------------------------
UPSKILL_COST = 5_000.0               # dollar cost of initiating an upskill
UPSKILL_MONTHS = 6                   # months before the skill boost takes effect
UPSKILL_SKILL_BOOST = 1              # skill levels gained per completed upskill
MAX_SKILL_LEVEL = 10                 # hard ceiling on skill
SWITCH_OCCUPATION_COST = 2_000.0     # one-time cost to switch occupations

# ---------------------------------------------------------------------------
# Skill transfer on occupation switch
# ---------------------------------------------------------------------------
BASE_SKILL_RETENTION = 0.8                  # same-industry retention (upper bound)
MIN_SKILL_RETENTION = 0.2                   # cross-industry retention (floor)

# ---------------------------------------------------------------------------
# Living expenses & health
# ---------------------------------------------------------------------------
MONTHLY_LIVING_EXPENSE = 2_000.0     # baseline monthly living cost

# ---------------------------------------------------------------------------
# Tax (flat rate for simplicity)
# ---------------------------------------------------------------------------
TAX_RATE = 0.15                      # 15 % flat income tax

# ---------------------------------------------------------------------------
# EnvConfig — single source of truth for environment rules
# ---------------------------------------------------------------------------
@dataclass
class EnvConfig:
    """Environment-wide parameters that apply regardless of the agent.

    Agent-specific initial conditions live in ``profile`` (an ``AgentProfile``).
    """
    # ---- agent profile ----
    profile: AgentProfile = field(default_factory=AgentProfile)

    # ---- time ----
    end_age: int = END_AGE
    start_year: int = START_YEAR
    start_month: int = START_MONTH

    # ---- career costs (no hardcoded defaults in logic — all driven from here) ----
    monthly_living_expense: float = MONTHLY_LIVING_EXPENSE
    upskill_cost: float = UPSKILL_COST
    upskill_months: int = UPSKILL_MONTHS
    upskill_skill_boost: int = UPSKILL_SKILL_BOOST
    max_skill_level: int = MAX_SKILL_LEVEL
    switch_occupation_base_cost: float = SWITCH_OCCUPATION_COST

    # ---- health (age-accelerated decline — environment physics, not agent-controlled) ----
    health_decline_20_29: float = 0.0003  # per month, ages 20–29
    health_decline_30_39: float = 0.001   # per month, ages 30–39
    health_decline_40_49: float = 0.003   # per month, ages 40–49
    health_decline_50_plus: float = 0.006 # per month, ages 50+

    # ---- energy (short-term stamina — gates training pacing) ----
    energy_cost_per_upskill: float = 0.4          # one-time deduction when starting upskill
    energy_cost_per_intensive_work: float = 0.5   # one-time deduction when starting intensive work
    energy_decline_per_training_month: float = 0.15  # drain per month while training
    energy_recovery_per_month: float = 0.10        # recovery per month when not training
    energy_threshold_for_upskill: float = 0.4      # minimum energy to start upskill / intensive work

    # ---- occupation skill (within-career progression) ----
    intensive_work_months: int = 3          # months to gain +1 occ_skill via intensive work
    occ_skill_passive_months: int = 12      # months of auto-work to gain +1 occ_skill passively

    # ---- macro ----
    macro_data_dir: str = "raw_data"       # path to cycle CSV directories
    macro_cycle: str = ""                  # "boom" / "normal" / "recession" / "" (random)
    macro_cycle_file: str = ""             # specific CSV, e.g. "2008_2009.csv" (overrides macro_cycle)
    layoff_base_rate: float = 0.05         # base monthly layoff probability (literature ~0.028)

    # ---- misc ----
    seed: Optional[int] = None

    # ---- stock market ----
    stock_data_file: str = "stock_monthly.csv"
    forced_sale_discount: float = 0.10
    require_stock_data: bool = True
    min_cash_buffer: float = 2_000.0           # minimum cash to keep after stock purchase
