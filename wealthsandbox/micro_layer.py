"""Micro-layer: thin wrapper around AgentState."""

from typing import Any, Dict

from wealthsandbox.types import AgentState, JobStatus
from wealthsandbox.profile import AgentProfile


class MicroLayer:
    """Manages the mutable AgentState."""

    def __init__(self, profile: AgentProfile):
        self.profile = profile
        self.state = self._make_initial_state()

    def _make_initial_state(self) -> AgentState:
        return AgentState(
            age=self.profile.age,
            health=self.profile.initial_health,
            energy=self.profile.initial_energy,
            cash=self.profile.initial_cash,
            general_skill=self.profile.initial_general_skill,
            occupation_skills={},
            occupation_id="",
            job_status=JobStatus.UNEMPLOYED,
            tenure_months=0,
            current_tier=0,
            monthly_after_tax_income=0.0,
            career_history=[],
            upskill_months_remaining=0,
            intensive_work_months_remaining=0,
            training_months_remaining=0,
            training_target_occupation="",
            last_month_events=[],
        )

    def reset(self) -> AgentState:
        self.state = self._make_initial_state()
        return self.state

    def snapshot(self) -> Dict[str, Any]:
        s = self.state
        occ_skill = s.occupation_skills.get(s.occupation_id, 1) if s.occupation_id else 0
        return {
            "age": s.age,
            "health": round(s.health, 3),
            "energy": round(s.energy, 3),
            "cash": round(s.cash, 2),
            "savings": round(s.savings, 2),
            "loan_balance": round(s.loan_balance, 2),
            "occupation_id": s.occupation_id,
            "general_skill": s.general_skill,
            "occ_skill": occ_skill,
            "tenure_months": s.tenure_months,
            "current_tier": s.current_tier,
            "monthly_after_tax_income": round(s.monthly_after_tax_income, 2),
            "job_status": s.job_status.value,
            "upskill_months_remaining": s.upskill_months_remaining,
            "intensive_work_months_remaining": s.intensive_work_months_remaining,
            "training_months_remaining": s.training_months_remaining,
            "training_target_occupation": s.training_target_occupation,
            "last_month_events": list(s.last_month_events),
        }
