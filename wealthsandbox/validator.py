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

    # Cash gate
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
