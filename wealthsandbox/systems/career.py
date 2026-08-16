"""CareerSystem: manages occupations, tiers, skills, and monthly income.

Skill is two-dimensional:
* ``general_skill`` — transferable capability (upskill, $, 6 mo).  Gates
  occupation entry and carries across switches via the proximity rule.
* ``occupation_skills[occ_id]`` — per-job experience (intensive_work, energy,
  3 mo; or passive, 12 mo).  Gates tier promotion within a job.  Not carried
  across occupation switches.

This module owns:
* The registry of occupations with tier ladders.
* The skill-based monthly income calculation (pre-tax and after-tax).
* Career actions: switch occupation, upskill, intensive work, quit job.
* Industry proximity for dynamic skill transfer on occupation switch.
* Training period logic with energy consumption.
* Health-based forced resignation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.types import (
    Action, AgentState, CareerMove, JobStatus, Tier,
)
from wealthsandbox.config import TAX_RATE, BASE_SKILL_RETENTION, MIN_SKILL_RETENTION


# ---------------------------------------------------------------------------
# Occupation definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Occupation:
    """A single occupation with a tier-based career ladder.

    Attributes:
        occupation_id: Unique machine-readable identifier.
        industry: Broad sector (e.g. "tech", "finance").
        display_name: Human-readable name.
        base_monthly_salary: Base pay at Tier 0 ×1.0 multiplier.
        skill_sensitivity: Percent change in base pay per general_skill level
            above/below the reference level (3).
        min_general_skill: Minimum general_skill needed to enter.
        min_health: Minimum health score needed to enter AND remain in this
            occupation.  If health drops below this threshold the agent is
            forced to resign at the start of the next month.
        entry_cost: One-time training / certification fee (in addition to
            the base switch cost).
        training_months: Months of training required before working.
        tiers: Career ladder rungs.  Tier 0 is the starting rung.
    """
    occupation_id: str
    industry: str
    display_name: str
    base_monthly_salary: float
    skill_sensitivity: float = 0.05
    min_general_skill: int = 1
    min_health: float = 0.0
    entry_cost: float = 0.0
    training_months: int = 0
    tiers: Tuple[Tier, ...] = (
        Tier("Standard", min_occ_skill=1, min_tenure_months=0, salary_multiplier=1.0),
    )


# ---------------------------------------------------------------------------
# Default occupation registry
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Calibration anchor: BLS OEWS May 2025 National
#   base_monthly = P25 annual ÷ 12  (entry / tier-0 salary)
#   tier multiplier(k) = P(k) ÷ P25  where k ∈ {P25, P50, P75, P90}
#   (except manufacturing_worker and civil_servant: median-based rough
#    estimates — P25 data unavailable for these SOC aggregates)
# ---------------------------------------------------------------------------
DEFAULT_OCCUPATIONS: Dict[str, Occupation] = {
    occ.occupation_id: occ
    for occ in [
        # SOC 15-1252 — P25~$105,600  P50~$142,900  P75~$184,200  P90~$225,300
        Occupation(
            occupation_id="software_engineer",
            industry="tech",
            display_name="Software Engineer",
            base_monthly_salary=8_800.0,
            skill_sensitivity=0.06,
            min_general_skill=4,
            min_health=0.3,
            entry_cost=10_000.0,
            training_months=4,
            tiers=(
                Tier("Junior",      min_occ_skill=1, min_tenure_months=0,  salary_multiplier=1.0),
                Tier("Mid-level",   min_occ_skill=3, min_tenure_months=18, salary_multiplier=1.35),
                Tier("Senior",      min_occ_skill=6, min_tenure_months=48, salary_multiplier=1.74),
                Tier("Principal",   min_occ_skill=9, min_tenure_months=96, salary_multiplier=2.13),
            ),
        ),
        # SOC 15-2051 — P25 $85,660  P50 $120,230  P75 $158,880  P90 $199,130
        Occupation(
            occupation_id="data_scientist",
            industry="tech",
            display_name="Data Scientist",
            base_monthly_salary=7_100.0,
            skill_sensitivity=0.06,
            min_general_skill=4,
            min_health=0.3,
            entry_cost=10_000.0,
            training_months=4,
            tiers=(
                Tier("Junior",      min_occ_skill=1, min_tenure_months=0,  salary_multiplier=1.0),
                Tier("Mid-level",   min_occ_skill=3, min_tenure_months=18, salary_multiplier=1.40),
                Tier("Senior",      min_occ_skill=6, min_tenure_months=48, salary_multiplier=1.85),
                Tier("Principal",   min_occ_skill=9, min_tenure_months=96, salary_multiplier=2.33),
            ),
        ),
        # SOC 41-3031 — P25~$58,000  P50 $78,660  P75~$115,000  P90 $212,880
        # Mean $109,150 used to anchor mid-career; steep ladder from right-skew
        Occupation(
            occupation_id="investment_banker",
            industry="finance",
            display_name="Investment Banker",
            base_monthly_salary=6_500.0,
            skill_sensitivity=0.05,
            min_general_skill=5,
            min_health=0.4,
            entry_cost=12_000.0,
            training_months=6,
            tiers=(
                Tier("Analyst",     min_occ_skill=1, min_tenure_months=0,  salary_multiplier=1.0),
                Tier("Associate",   min_occ_skill=4, min_tenure_months=24, salary_multiplier=1.40),
                Tier("VP",          min_occ_skill=7, min_tenure_months=60, salary_multiplier=2.10),
                Tier("MD",          min_occ_skill=9, min_tenure_months=120,salary_multiplier=2.73),
            ),
        ),
        # SOC 13-2051 — P25~$78,500  P50 $102,740  P75~$133,100  P90~$174,300
        Occupation(
            occupation_id="financial_analyst",
            industry="finance",
            display_name="Financial Analyst",
            base_monthly_salary=6_500.0,
            skill_sensitivity=0.05,
            min_general_skill=3,
            min_health=0.3,
            entry_cost=5_000.0,
            training_months=3,
            tiers=(
                Tier("Junior",      min_occ_skill=1, min_tenure_months=0,  salary_multiplier=1.0),
                Tier("Mid-level",   min_occ_skill=3, min_tenure_months=12, salary_multiplier=1.31),
                Tier("Senior",      min_occ_skill=6, min_tenure_months=36, salary_multiplier=1.70),
                Tier("Lead",        min_occ_skill=8, min_tenure_months=72, salary_multiplier=2.22),
            ),
        ),
        # SOC 51-0000 — Median ~$46,000  (P25 estimate ~$35,000; low confidence)
        # Using median-based anchor — safety-net job, kept stable
        Occupation(
            occupation_id="manufacturing_worker",
            industry="manufacturing",
            display_name="Manufacturing Worker",
            base_monthly_salary=3_800.0,
            skill_sensitivity=0.03,
            min_general_skill=1,
            min_health=0.6,
            entry_cost=0.0,
            training_months=0,
            tiers=(
                Tier("Apprentice",  min_occ_skill=1, min_tenure_months=0,  salary_multiplier=1.0),
                Tier("Skilled",     min_occ_skill=3, min_tenure_months=12, salary_multiplier=1.30),
                Tier("Supervisor",  min_occ_skill=5, min_tenure_months=36, salary_multiplier=1.70),
                Tier("Manager",     min_occ_skill=7, min_tenure_months=72, salary_multiplier=2.20),
            ),
        ),
        # SOC 29-1141 — P25~$83,000  P50~$100,000  P90~$150,000  (3-tier, compressed)
        Occupation(
            occupation_id="nurse",
            industry="healthcare",
            display_name="Nurse",
            base_monthly_salary=6_900.0,
            skill_sensitivity=0.04,
            min_general_skill=2,
            min_health=0.5,
            entry_cost=4_000.0,
            training_months=3,
            tiers=(
                Tier("Staff Nurse", min_occ_skill=1, min_tenure_months=0,  salary_multiplier=1.0),
                Tier("Senior Nurse",min_occ_skill=4, min_tenure_months=24, salary_multiplier=1.20),
                Tier("Head Nurse",  min_occ_skill=7, min_tenure_months=60, salary_multiplier=1.81),
            ),
        ),
        # SOC 43-0000 proxy — Median ~$47,000  (P25 estimate ~$38,000; low confidence)
        Occupation(
            occupation_id="civil_servant",
            industry="gov",
            display_name="Civil Servant",
            base_monthly_salary=3_800.0,
            skill_sensitivity=0.03,
            min_general_skill=2,
            min_health=0.3,
            entry_cost=1_000.0,
            training_months=2,
            tiers=(
                Tier("Junior Officer",  min_occ_skill=1, min_tenure_months=0,  salary_multiplier=1.0),
                Tier("Senior Officer",  min_occ_skill=4, min_tenure_months=36, salary_multiplier=1.24),
                Tier("Director",        min_occ_skill=7, min_tenure_months=96, salary_multiplier=1.58),
            ),
        ),
    ]
}


# ---------------------------------------------------------------------------
# Industry proximity matrix (for skill transfer on occupation switch)
# ---------------------------------------------------------------------------
INDUSTRY_PROXIMITY: Dict[str, Dict[str, float]] = {
    "tech":          {"tech": 1.0, "finance": 0.5, "healthcare": 0.2, "manufacturing": 0.1, "gov": 0.1},
    "finance":       {"tech": 0.5, "finance": 1.0, "healthcare": 0.2, "manufacturing": 0.1, "gov": 0.2},
    "healthcare":    {"tech": 0.2, "finance": 0.2, "healthcare": 1.0, "manufacturing": 0.2, "gov": 0.3},
    "manufacturing": {"tech": 0.1, "finance": 0.1, "healthcare": 0.2, "manufacturing": 1.0, "gov": 0.2},
    "gov":           {"tech": 0.1, "finance": 0.2, "healthcare": 0.3, "manufacturing": 0.2, "gov": 1.0},
}

_REFERENCE_SKILL_LEVEL: int = 3


# ---------------------------------------------------------------------------
# CareerSystem
# ---------------------------------------------------------------------------
class CareerSystem(BaseSystem):
    """Manages occupations, tiers, and two-dimensional skill progression.

    * ``general_skill`` — improves via *upskill* ($, 6 mo).  Transferrable,
      gates occupation entry.
    * ``occupation_skills`` — improves via *intensive_work* (energy, 3 mo) or
      passive tenure (every 12 mo).  Per-job, not transferrable.  Gates tier
      promotion.

    Implements the BaseSystem protocol.
    """

    def __init__(
        self,
        occupations: Optional[Dict[str, Occupation]] = None,
        *,
        upskill_cost: float = 5_000.0,
        upskill_months: int = 6,
        upskill_skill_boost: int = 1,
        max_general_skill: int = 10,
        max_occ_skill: int = 10,
        switch_base_cost: float = 2_000.0,
        living_expense: float = 2_000.0,
        intensive_work_months: int = 3,
        occ_skill_passive_months: int = 12,
        layoff_base_rate: float = 0.02,
        rest_income_penalty: float = 0.20,
        seed: Optional[int] = None,
    ):
        self.occupations = occupations or dict(DEFAULT_OCCUPATIONS)
        self.upskill_cost = upskill_cost
        self.upskill_months = upskill_months
        self.upskill_skill_boost = upskill_skill_boost
        self.max_general_skill = max_general_skill
        self.max_occ_skill = max_occ_skill
        self.switch_base_cost = switch_base_cost
        self.living_expense = living_expense
        self.intensive_work_months = intensive_work_months
        self.occ_skill_passive_months = occ_skill_passive_months
        self.layoff_base_rate = layoff_base_rate
        self.rest_income_penalty = rest_income_penalty
        import random
        self._rng = random.Random(seed)
        # Macro-driven state
        self._income_mult: float = 1.0
        self._price_level: float = 1.0

    # ------------------------------------------------------------------
    # Industry sensitivity for layoff and income
    # ------------------------------------------------------------------

    # Layoff sensitivity by industry (multiplier on base layoff rate)
    LAYOFF_INDUSTRY_MULT: Dict[str, float] = {
        "tech": 1.3, "finance": 1.2, "manufacturing": 1.0,
        "healthcare": 0.7, "gov": 0.5,
    }

    # Income demand multiplier during recession (USRECM=1).  Normal=1.0.
    RECESSION_INCOME_MULT: Dict[str, float] = {
        "tech": 0.75, "finance": 0.70, "manufacturing": 0.80,
        "healthcare": 0.90, "gov": 0.95,
    }

    # ------------------------------------------------------------------
    # Registry access
    # ------------------------------------------------------------------

    def get_occupation(self, occupation_id: str) -> Occupation:
        if occupation_id not in self.occupations:
            raise ValueError(f"Unknown occupation_id: {occupation_id}")
        return self.occupations[occupation_id]

    def list_occupations(self) -> List[str]:
        return list(self.occupations.keys())

    def has_occupation(self, state: AgentState) -> bool:
        return state.occupation_id in self.occupations

    def _occ_skill(self, state: AgentState) -> int:
        """Current occupation skill, or 1 if none."""
        if not state.occupation_id:
            return 1
        return state.occupation_skills.get(state.occupation_id, 1)

    def _current_tier(self, state: AgentState) -> Tier:
        """Return the agent's current Tier in their occupation."""
        if not self.has_occupation(state):
            return Tier("Unemployed", 0, 0, 1.0)
        occ = self.get_occupation(state.occupation_id)
        idx = state.current_tier
        if idx < len(occ.tiers):
            return occ.tiers[idx]
        return occ.tiers[-1]

    # ------------------------------------------------------------------
    # BaseSystem protocol
    # ------------------------------------------------------------------

    def tick(self, state: AgentState, macro: Dict[str, Any]) -> None:
        """Monthly automatic processing.

        0. Absorb macro (UNRATE, USRECM).
        1. Health-based forced resignation.
        2. Layoff check (employed agents).
        3. Rehire check (unemployed agents).
        4. Auto income × recession demand multiplier.
        5. Tenure, passive occ_skill, promotion.
        6. Timer advancement.
        """
        unrate = macro.get("unrate", 0.05)
        usrecm = macro.get("usrecm", 0)
        self._price_level = macro.get("price_level", 1.0)

        # ---- 0. Set income multiplier from recession flag ----
        industry = self.get_industry(state)
        if usrecm == 1:
            self._income_mult = self.RECESSION_INCOME_MULT.get(industry, 0.85)
        else:
            self._income_mult = 1.0

        # ---- 1. Health forced resignation ----
        if self.has_occupation(state) and state.job_status == JobStatus.EMPLOYED:
            occ = self.get_occupation(state.occupation_id)
            if state.health < occ.min_health:
                self._force_resign(state, occ)

        # ---- 2. Layoff check (employed) ----
        if self.has_occupation(state) and state.job_status == JobStatus.EMPLOYED:
            layoff_p = self._compute_layoff_prob(state, unrate, usrecm)
            if self._rng.random() < layoff_p:
                self._apply_layoff(state, unrate)

        # ---- 3. Track unemployment ----
        if not self.has_occupation(state) or state.job_status == JobStatus.UNEMPLOYED:
            if state.training_months_remaining <= 0:
                state.unemployed_months += 1

        # ---- 4. Automatic income ----
        if self.has_occupation(state) and state.job_status == JobStatus.EMPLOYED:
            income = self.compute_monthly_after_tax_income(state)
            if state.resting_this_month:
                income = round(income * (1.0 - self.rest_income_penalty), 2)
            state.monthly_after_tax_income = income
            state.cash += income
            state.record_flow("employment_income", income)
            tier = self._current_tier(state)
            occ = self.get_occupation(state.occupation_id)
            if state.resting_this_month:
                state.last_month_events.append(
                    f"Earned ${income:,.0f} as {tier.name} {occ.display_name} "
                    f"(reduced {self.rest_income_penalty:.0%} for resting)."
                )
            else:
                state.last_month_events.append(
                    f"Earned ${income:,.0f} as {tier.name} {occ.display_name}."
                )
        elif self.has_occupation(state) and state.job_status == JobStatus.UNEMPLOYED:
            state.monthly_after_tax_income = 0.0
            state.last_month_events.append("Unemployed — no income this month.")
        else:
            state.monthly_after_tax_income = 0.0
            state.last_month_events.append("No occupation — no income this month.")
        state.resting_this_month = False

        # ---- 3. Tenure & passive occ_skill & promotion ----
        if self.has_occupation(state) and state.job_status == JobStatus.EMPLOYED:
            state.tenure_months += 1
            # Passive occ_skill growth every N months
            if state.tenure_months > 0 and state.tenure_months % self.occ_skill_passive_months == 0:
                self._gain_occ_skill(state, 1)
                state.last_month_events.append(
                    f"Occupation skill increased to {self._occ_skill(state)} (passive tenure)."
                )
            # Check for auto-promotion
            self._check_promotion(state)

        # ---- 4. Timers ----
        self.tick_upskill(state)
        self.tick_intensive_work(state)
        self.tick_training(state)

    def handle_action(self, action: Action, state: AgentState) -> bool:
        """Dispatch an agent-initiated career action."""
        if action.career_move == CareerMove.SWITCH_OCCUPATION:
            target = action.target_occupation_id
            if not target:
                state.last_month_events.append("Tried to switch but no occupation specified.")
            else:
                try:
                    self.process_switch_occupation(state, target)
                except ValueError:
                    state.last_month_events.append(f"'{target}' is not a valid occupation.")
            return True
        elif action.career_move == CareerMove.UPSKILL:
            self.process_upskill(state)
            return True
        elif action.career_move == CareerMove.INTENSIVE_WORK:
            self.process_intensive_work(state)
            return True
        elif action.career_move == CareerMove.QUIT_JOB:
            self.process_quit_job(state)
            return True
        return False

    # ------------------------------------------------------------------
    # Income calculation
    # ------------------------------------------------------------------

    def get_monthly_base_salary(self, state: AgentState) -> float:
        """Pre-tax base salary = occupation.base * sensitivity * tier * industry."""
        if not self.has_occupation(state):
            return 0.0
        occ = self.get_occupation(state.occupation_id)
        skill_delta = state.general_skill - _REFERENCE_SKILL_LEVEL
        salary = occ.base_monthly_salary * (1 + occ.skill_sensitivity * skill_delta)
        salary *= self._current_tier(state).salary_multiplier
        salary *= self._income_mult
        salary *= self._price_level  # cost-of-living adjustment (nominal)
        return max(0.0, salary)

    def compute_monthly_after_tax_income(self, state: AgentState) -> float:
        """Take-home pay (pre-tax minus flat tax)."""
        gross = self.get_monthly_base_salary(state)
        after_tax = gross * (1.0 - TAX_RATE)
        return round(after_tax, 2)

    # ------------------------------------------------------------------
    # Career actions (agent-initiated)
    # ------------------------------------------------------------------

    def process_switch_occupation(self, state: AgentState, target_id: str) -> None:
        """Request a switch to *target_id*."""
        occ = self.get_occupation(target_id)

        ok, reason = self.check_entry_requirement(state, target_id)
        if not ok:
            state.last_month_events.append(reason)
            return

        total_cost = self.switch_base_cost + occ.entry_cost

        if target_id == "manufacturing_worker":
            # Safety-net occupation — always free to join, no cash required.
            self._apply_occupation_switch(state, target_id)
            state.last_month_events.append(
                f"Switched to {occ.display_name} (no cost — safety-net job)."
            )
            return

        if state.cash < total_cost:
            state.last_month_events.append(
                f"Cannot afford switch: need ${total_cost:,.0f}, have ${state.cash:,.0f}."
            )
            return

        state.cash -= total_cost
        state.record_flow("career_cost", -total_cost)

        if occ.training_months > 0:
            state.training_months_remaining = occ.training_months
            state.training_target_occupation = target_id
            state.last_month_events.append(
                f"Started training for {occ.display_name} ({occ.training_months} months, ${total_cost:,.0f})."
            )
        else:
            self._apply_occupation_switch(state, target_id)
            state.last_month_events.append(
                f"Switched to {occ.display_name} (cost ${total_cost:,.0f})."
            )

    def process_upskill(self, state: AgentState) -> None:
        """Start upskilling: general_skill +1 after upskill_months.

        Costs $ and energy (deducted by EnergySystem).
        """
        if state.training_months_remaining > 0:
            state.last_month_events.append(
                f"Cannot upskill while training for {state.training_target_occupation}."
            )
            return
        if state.general_skill >= self.max_general_skill:
            state.last_month_events.append("Already at maximum general skill.")
            return
        if state.upskill_months_remaining > 0:
            state.last_month_events.append("Already upskilling — wait for it to finish.")
            return
        if state.cash < self.upskill_cost:
            state.last_month_events.append("Not enough cash to upskill.")
            return

        state.cash -= self.upskill_cost
        state.record_flow("career_cost", -self.upskill_cost)
        state.upskill_months_remaining = self.upskill_months
        state.last_month_events.append(
            f"Started upskilling (${self.upskill_cost:,.0f}, {self.upskill_months} months)."
        )

    def process_intensive_work(self, state: AgentState) -> None:
        if not self.has_occupation(state) or state.job_status != JobStatus.EMPLOYED:
            state.last_month_events.append("Must be employed to do intensive work.")
            return
        occ_skill = self._occ_skill(state)
        if occ_skill >= self.max_occ_skill:
            state.last_month_events.append("Already at maximum occupation skill.")
            return
        if state.intensive_work_months_remaining > 0:
            state.last_month_events.append("Already doing intensive work — wait for it to finish.")
            return

        state.intensive_work_months_remaining = self.intensive_work_months
        state.last_month_events.append(
            f"Started intensive work ({self.intensive_work_months} months)."
        )

    def process_quit_job(self, state: AgentState) -> None:
        if state.job_status != JobStatus.EMPLOYED:
            state.last_month_events.append("Cannot quit — not currently employed.")
            return
        state.prev_occupation_id = state.occupation_id
        state.job_status = JobStatus.UNEMPLOYED
        state.occupation_id = ""
        state.monthly_after_tax_income = 0.0
        state.tenure_months = 0
        state.current_tier = 0
        state.unemployed_months = 0
        state.last_month_events.append("Quit job — now unemployed.")

    # ------------------------------------------------------------------
    # Entry requirement checks
    # ------------------------------------------------------------------

    def check_entry_requirement(self, state: AgentState, target_id: str) -> Tuple[bool, str]:
        occ = self.get_occupation(target_id)
        if state.general_skill < occ.min_general_skill:
            return False, (
                f"General skill too low: need {occ.min_general_skill}, "
                f"have {state.general_skill}."
            )
        if state.health < occ.min_health:
            return False, (
                f"Health too low for {occ.display_name}: "
                f"need {occ.min_health:.1f}, have {state.health:.3f}."
            )
        if state.training_months_remaining > 0:
            return False, (
                f"Already training for {state.training_target_occupation} "
                f"({state.training_months_remaining} months remaining)."
            )
        return True, ""

    # ------------------------------------------------------------------
    # Skill transfer (general_skill only)
    # ------------------------------------------------------------------

    def get_skill_retention(self, from_occ_id: str, to_occ_id: str) -> float:
        if not from_occ_id:
            return 1.0
        from_ind = self.get_occupation(from_occ_id).industry
        to_ind = self.get_occupation(to_occ_id).industry
        proximity = INDUSTRY_PROXIMITY.get(from_ind, {}).get(to_ind, 0.0)
        return max(MIN_SKILL_RETENTION, proximity * BASE_SKILL_RETENTION)

    # ------------------------------------------------------------------
    # Training tick
    # ------------------------------------------------------------------

    def tick_training(self, state: AgentState) -> None:
        if state.training_months_remaining <= 0:
            return
        state.training_months_remaining -= 1
        if state.training_months_remaining == 0:
            target = state.training_target_occupation
            self._apply_occupation_switch(state, target)
            state.training_target_occupation = ""
            occ = self.get_occupation(target)
            state.last_month_events.append(f"Training completed — now a {occ.display_name}.")

    def _apply_occupation_switch(self, state: AgentState, target_id: str) -> None:
        """Apply occupation switch with skill retention (both general and occupation).

        General skill carries over by industry proximity.
        Occupation skill: if switching back to a previously worked occupation,
        the saved skill is restored (× industry proximity retention).  First
        time in an occupation always starts at tier 0's min_skill.
        """
        from_id = state.occupation_id

        # ---- General skill transfer ----
        old_gen = state.general_skill
        retention = self.get_skill_retention(from_id, target_id)
        state.general_skill = max(1, int(old_gen * retention))

        state.occupation_id = target_id
        state.job_status = JobStatus.EMPLOYED
        state.tenure_months = 0
        state.current_tier = 0

        # ---- Occupation skill: restore previous experience or start fresh ----
        occ = self.get_occupation(target_id)
        floor = occ.tiers[0].min_occ_skill if occ.tiers else 1

        prev_skill = state.occupation_skills.get(target_id)
        if prev_skill is not None and prev_skill > floor:
            # Returning to a previously worked occupation — restore with retention
            retained = max(floor, int(prev_skill * retention))
            state.occupation_skills[target_id] = retained
            state.last_month_events.append(
                f"Occupation skill restored: {prev_skill} → {retained} "
                f"(previously worked here, {retention:.0%} retention)."
            )
        else:
            # Fresh occupation or skill already at floor — start clean
            state.occupation_skills[target_id] = floor

        state.last_month_events.append(
            f"General skill transferred: {old_gen} → {state.general_skill} "
            f"({retention:.0%} retention, cross-industry)."
        )

    # ------------------------------------------------------------------
    # Health forced resignation
    # ------------------------------------------------------------------

    def _force_resign(self, state: AgentState, occ: Occupation) -> None:
        """Force the agent out of their current occupation due to low health."""
        tier_name = self._current_tier(state).name
        state.prev_occupation_id = state.occupation_id
        state.job_status = JobStatus.UNEMPLOYED
        state.occupation_id = ""
        state.monthly_after_tax_income = 0.0
        state.tenure_months = 0
        state.current_tier = 0
        state.unemployed_months = 0
        state.last_month_events.append(
            f"HEALTH — Your health ({state.health:.3f}) fell below the minimum "
            f"({occ.min_health:.1f}) for {occ.display_name} ({tier_name}). "
            f"Forced to resign. Choose a less physically demanding occupation."
        )

    # ------------------------------------------------------------------
    # Macro-driven layoff / rehire
    # ------------------------------------------------------------------

    def _compute_layoff_prob(
        self, state: AgentState, unrate: float, usrecm: int,
    ) -> float:
        """Compute monthly layoff probability for an employed agent.

        layoff_prob = base_rate × macro_mult × industry_mult × age_mult × health_mult
        Capped at 0.15.
        """
        base_rate = self.layoff_base_rate
        macro_mult = 1.0 + max(0.0, unrate - 0.05) * 8.0 + (usrecm * 0.3)
        industry = self.get_industry(state)
        industry_mult = self.LAYOFF_INDUSTRY_MULT.get(industry, 1.0)
        age_mult = 1.3 if state.age > 45 else 1.0
        health_mult = 1.2 if state.health < 0.5 else 1.0
        prob = base_rate * macro_mult * industry_mult * age_mult * health_mult
        return min(0.15, prob)

    def _apply_layoff(self, state: AgentState, unrate: float) -> None:
        """Execute a layoff: become unemployed, preserve occupation data for rehire."""
        occ = self.get_occupation(state.occupation_id)
        tier_name = self._current_tier(state).name
        state.prev_occupation_id = state.occupation_id
        state.job_status = JobStatus.UNEMPLOYED
        state.occupation_id = ""
        state.monthly_after_tax_income = 0.0
        state.tenure_months = 0
        state.current_tier = 0
        state.unemployed_months = 0
        state.last_month_events.append(
            f"LAYOFF — Due to weak economic conditions, you have been laid off "
            f"from your job as {tier_name} {occ.display_name}. No income starting "
            f"next month, but living expenses (${self.living_expense:,.0f}/mo) "
            f"still apply. You can look for a new job or wait to be rehired."
        )

    # ------------------------------------------------------------------
    # Occupation-skill helpers
    # ------------------------------------------------------------------

    def _gain_occ_skill(self, state: AgentState, amount: int) -> None:
        """Add *amount* occupation skill points.

        If the agent is currently employed the skill goes to the current
        occupation.  If the agent was laid off / forced to resign earlier in
        this same tick, the skill goes to the previous occupation instead
        (the intensive work was already in progress — the agent still learns
        from it).
        """
        occ_id = state.occupation_id or state.prev_occupation_id
        if not occ_id:
            return
        cur = state.occupation_skills.get(occ_id, 1)
        state.occupation_skills[occ_id] = min(
            self.max_occ_skill, cur + amount
        )

    def _check_promotion(self, state: AgentState) -> None:
        """Check and apply tier promotion if conditions are met."""
        if not self.has_occupation(state):
            return
        occ = self.get_occupation(state.occupation_id)
        occ_skill = self._occ_skill(state)
        # Check each tier above current
        for idx in range(state.current_tier + 1, len(occ.tiers)):
            tier = occ.tiers[idx]
            if occ_skill >= tier.min_occ_skill and state.tenure_months >= tier.min_tenure_months:
                old_tier = occ.tiers[state.current_tier].name
                state.current_tier = idx
                state.last_month_events.append(
                    f"PROMOTION — Promoted from {old_tier} to {tier.name} "
                    f"in {occ.display_name}. Salary multiplier now ×{tier.salary_multiplier}."
                )
                # Only promote one tier per month
                break

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def tick_upskill(self, state: AgentState) -> None:
        if state.upskill_months_remaining <= 0:
            return
        state.upskill_months_remaining -= 1
        if state.upskill_months_remaining == 0:
            old = state.general_skill
            state.general_skill = min(self.max_general_skill, old + self.upskill_skill_boost)
            state.last_month_events.append(
                f"Upskill completed — general skill {old} → {state.general_skill}."
            )

    def tick_intensive_work(self, state: AgentState) -> None:
        if state.intensive_work_months_remaining <= 0:
            return
        state.intensive_work_months_remaining -= 1
        if state.intensive_work_months_remaining == 0:
            self._gain_occ_skill(state, 1)
            occ_id = state.occupation_id or state.prev_occupation_id
            skill = state.occupation_skills.get(occ_id, 1) if occ_id else 1
            state.last_month_events.append(
                f"Intensive work completed — occupation skill now {skill}."
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_industry(self, state: AgentState) -> str:
        """Return the industry of the agent's current occupation, or empty string."""
        if not self.has_occupation(state):
            return ""
        return self.get_occupation(state.occupation_id).industry

    def get_industry_incomes(self) -> Dict[str, float]:
        by_industry: Dict[str, List[float]] = {}
        for occ in self.occupations.values():
            by_industry.setdefault(occ.industry, []).append(occ.base_monthly_salary)
        return {
            ind: round(sum(salaries) / len(salaries), 2)
            for ind, salaries in by_industry.items()
        }

    def get_occupation_details(self) -> Dict[str, Dict[str, Any]]:
        """Return occupation metadata for agent observation."""
        return {
            occ_id: {
                "industry": occ.industry,
                "display_name": occ.display_name,
                "base_monthly_salary": occ.base_monthly_salary,
                "skill_sensitivity": occ.skill_sensitivity,
                "min_general_skill": occ.min_general_skill,
                "min_health": occ.min_health,
                "entry_cost": occ.entry_cost,
                "training_months": occ.training_months,
                "tiers": [
                    {
                        "name": t.name,
                        "min_occ_skill": t.min_occ_skill,
                        "min_tenure_months": t.min_tenure_months,
                        "salary_multiplier": t.salary_multiplier,
                    }
                    for t in occ.tiers
                ],
            }
            for occ_id, occ in self.occupations.items()
        }
