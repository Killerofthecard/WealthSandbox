"""EnergySystem: manages the agent's short-term stamina / vitality level.

Energy is consumed in two ways:
* **One-time deduction** when starting an upskill (e.g. -0.4).
* **Continuous drain** during occupation training (e.g. -0.15/month).

Energy recovers when the agent is not training.  Low energy gates upskill
(the validator rejects upskill attempts when energy is below the threshold).

Removing this system from ``systems`` means the agent can upskill and train
without any stamina constraint.
"""

from typing import Any, Dict

from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.types import Action, AgentState, CareerMove


class EnergySystem(BaseSystem):
    """Consume energy on upskill/training, recover during rest.

    Energy gates the agent's ability to chain skill improvements or
    occupation switches back-to-back — they must rest between them.
    """

    def __init__(
        self,
        cost_per_upskill: float = 0.4,
        cost_per_intensive_work: float = 0.5,
        decline_per_training_month: float = 0.15,
        recovery_per_month: float = 0.10,
    ):
        """
        Args:
            cost_per_upskill: One-time energy deduction when starting an upskill.
            cost_per_intensive_work: One-time energy deduction when starting
                intensive work.
            decline_per_training_month: Energy lost per month while training
                for a new occupation.
            recovery_per_month: Energy regained per month when NOT training.
        """
        self.cost_per_upskill = cost_per_upskill
        self.cost_per_intensive_work = cost_per_intensive_work
        self.decline_per_training_month = decline_per_training_month
        self.recovery_per_month = recovery_per_month

    # ------------------------------------------------------------------
    # BaseSystem protocol
    # ------------------------------------------------------------------

    def tick(self, state: AgentState, macro: Dict[str, Any]) -> None:
        """Continuous drain during training, recovery otherwise."""
        if state.training_months_remaining > 0:
            state.energy = max(
                0.0, state.energy - self.decline_per_training_month
            )
        else:
            state.energy = min(
                1.0, state.energy + self.recovery_per_month
            )

    def handle_action(self, action: Action, state: AgentState) -> bool:
        """Deduct one-time energy cost on upskill or intensive_work.

        Returns False so CareerSystem also sees the action (we only react,
        we don't consume it).
        """
        if action.career_move == CareerMove.UPSKILL:
            state.energy = max(0.0, state.energy - self.cost_per_upskill)
        elif action.career_move == CareerMove.INTENSIVE_WORK:
            state.energy = max(0.0, state.energy - self.cost_per_intensive_work)
        return False
