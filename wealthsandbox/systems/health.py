"""HealthSystem: applies age-accelerated health decline and declares death.

Passive system — the agent observes its health ticking down but cannot directly
influence it.  The decline rate increases with age:

========  =============  ============================================
Ages      Monthly loss   Effect
========  =============  ============================================
20–29     0.0003         Barely noticeable (~0.4% per year)
30–39     0.001          Slow decline (~1.2% per year)
40–49     0.003          Noticeable (~3.6% per year)
50+       0.006          Steep (~7.2% per year)
========  =============  ============================================

Removing this system from ``systems`` means the agent never ages out of
physically demanding occupations.
"""

from typing import Any, Dict, Optional

from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.types import AgentState, Action


class HealthSystem(BaseSystem):
    """Reduce the agent's health at an age-dependent rate every month.

    When health reaches zero, ``check_dead()`` returns ``"death"`` and the
    episode terminates.  Occupations with a ``min_health`` higher than the
    agent's current health cannot be entered (enforced in the validator).
    """

    def __init__(
        self,
        decline_20_29: float = 0.0003,
        decline_30_39: float = 0.001,
        decline_40_49: float = 0.003,
        decline_50_plus: float = 0.006,
    ):
        """
        Args:
            decline_20_29: Monthly health loss for ages 20–29.
            decline_30_39: Monthly health loss for ages 30–39.
            decline_40_49: Monthly health loss for ages 40–49.
            decline_50_plus: Monthly health loss for ages 50+.
        """
        self.decline_20_29 = decline_20_29
        self.decline_30_39 = decline_30_39
        self.decline_40_49 = decline_40_49
        self.decline_50_plus = decline_50_plus

    # ------------------------------------------------------------------
    # BaseSystem protocol
    # ------------------------------------------------------------------

    def tick(self, state: AgentState, macro: Dict[str, Any]) -> None:
        """No pre-action work — health decline is in ``finalize()``."""
        pass

    def handle_action(self, action: Action, state: AgentState) -> bool:
        """This system handles no agent actions."""
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
