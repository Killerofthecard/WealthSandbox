"""HealthSystem: applies age-accelerated health decline and declares death.

Passive system — the agent observes its health ticking down but cannot directly
influence it.  The decline rate increases with age:

========  =============  ============================================
Ages      Monthly loss   Effect
========  =============  ============================================
20–29     0.0003         Barely noticeable (~0.4% per year)
30–39     0.002          Slow decline (~2.4% per year)
40–49     0.006          Noticeable (~7.2% per year)
50+       0.012          Steep (~14.4% per year)
========  =============  ============================================

Removing this system from ``systems`` means the agent never ages out of
physically demanding occupations.
"""

from typing import Any, Dict, Optional

from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.types import AgentState, Action, CareerMove


class HealthSystem(BaseSystem):
    """Reduce the agent's health at an age-dependent rate every month.

    When health reaches zero, ``check_dead()`` returns ``"death"`` and the
    episode terminates.  Occupations with a ``min_health`` higher than the
    agent's current health cannot be entered (enforced in the validator).
    """

    def __init__(
        self,
        decline_20_29: float = 0.0003,
        decline_30_39: float = 0.002,
        decline_40_49: float = 0.006,
        decline_50_plus: float = 0.012,
        rest_health_gain: float = 0.02,
        medical_care_cost: float = 3_000.0,
        medical_care_health_gain: float = 0.05,
        health_max: float = 1.0,
    ):
        """
        Args:
            decline_20_29: Monthly health loss for ages 20–29.
            decline_30_39: Monthly health loss for ages 30–39.
            decline_40_49: Monthly health loss for ages 40–49.
            decline_50_50: Monthly health loss for ages 50+.
            rest_health_gain: Health recovered by ``rest``.
            medical_care_cost: Cash cost of ``medical_care``.
            medical_care_health_gain: Health recovered by ``medical_care``.
            health_max: Ceiling health cannot exceed.
        """
        self.decline_20_29 = decline_20_29
        self.decline_30_39 = decline_30_39
        self.decline_40_49 = decline_40_49
        self.decline_50_plus = decline_50_plus
        self.rest_health_gain = rest_health_gain
        self.medical_care_cost = medical_care_cost
        self.medical_care_health_gain = medical_care_health_gain
        self.health_max = health_max

    # ------------------------------------------------------------------
    # BaseSystem protocol
    # ------------------------------------------------------------------

    def tick(self, state: AgentState, macro: Dict[str, Any]) -> None:
        """No pre-action work — health decline is in ``finalize()``."""
        pass

    def handle_action(self, action: Action, state: AgentState) -> bool:
        """Handle ``rest`` and ``medical_care`` (recover health)."""
        if action.career_move == CareerMove.REST:
            state.health = min(self.health_max, state.health + self.rest_health_gain)
            state.resting_this_month = True
            state.last_month_events.append(
                f"Rested — health +{self.rest_health_gain:.2f} "
                f"(now {state.health:.3f})."
            )
            return True
        elif action.career_move == CareerMove.MEDICAL_CARE:
            state.cash -= self.medical_care_cost
            state.record_flow("medical_cost", -self.medical_care_cost)
            state.health = min(
                self.health_max, state.health + self.medical_care_health_gain
            )
            state.medical_care_uses_this_year += 1
            state.last_month_events.append(
                f"Medical care — paid ${self.medical_care_cost:,.0f}, "
                f"health +{self.medical_care_health_gain:.2f} "
                f"(now {state.health:.3f})."
            )
            return True
        return False

    def finalize(self, state: AgentState, macro: Dict[str, Any]) -> None:
        """Apply the monthly health decline based on the agent's age bracket."""
        rate = self._decline_rate(state.age)
        state.health = max(0.0, state.health - rate)

    def check_dead(self, state: AgentState) -> Optional[str]:
        """Death: health has reached zero."""
        if state.health <= 0.0:
            return "death"
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decline_rate(self, age: int) -> float:
        """Return the monthly health decline rate for *age*."""
        if age < 30:
            return self.decline_20_29
        elif age < 40:
            return self.decline_30_39
        elif age < 50:
            return self.decline_40_49
        else:
            return self.decline_50_plus
