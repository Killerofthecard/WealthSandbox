"""EnergySystem: models energy as a fixed capacity that is *occupied* by work,
training, upskilling, and intensive work.

Energy is no longer a consumable that drains and refills.  It is a capacity
(``energy_capacity``, default 1.0) that activities **occupy** for as long as
they last:

* Being employed occupies the current occupation's ``energy_footprint``
  (different jobs demand different amounts of energy).
* Occupation training, upskilling, and intensive work each occupy an
  additional fixed fraction while they are in progress.
* Occupied energy is released automatically when the activity ends (the
  relevant timer reaches zero, or the agent leaves the job).

``rest`` offsets the **non-work** occupancy for the month (training/upskill/
intensive work), but can never reduce the job's own structural occupancy —
only leaving the job frees that up.

Removing this system from ``systems`` means the agent can upskill and train
without any energy constraint.
"""

from typing import Any, Dict

from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.types import Action, AgentState, JobStatus


class EnergySystem(BaseSystem):
    """Recompute available energy each tick from the current occupancy.

    ``state.energy`` is the *available* energy = capacity − occupancy.  It is
    a derived quantity, not an accumulated one — so activities release their
    occupancy automatically the moment they finish.
    """

    def __init__(
        self,
        career,
        capacity: float = 1.0,
        training_footprint: float = 0.15,
        upskill_footprint: float = 0.15,
        intensive_footprint: float = 0.20,
    ):
        """
        Args:
            career: CareerSystem instance (for occupation ``energy_footprint``).
            capacity: Total energy capacity (energy never exceeds this).
            training_footprint: Capacity occupied while training for a new
                occupation.
            upskill_footprint: Capacity occupied while upskilling.
            intensive_footprint: Capacity occupied while doing intensive work.
        """
        self._career = career
        self.capacity = capacity
        self.training_footprint = training_footprint
        self.upskill_footprint = upskill_footprint
        self.intensive_footprint = intensive_footprint

    # ------------------------------------------------------------------
    # BaseSystem protocol
    # ------------------------------------------------------------------

    def tick(self, state: AgentState, macro: Dict[str, Any]) -> None:
        """Recompute available energy from the current occupancy."""
        available = self.capacity - self._occupancy(state)
        state.energy = max(0.0, min(self.capacity, available))

    def handle_action(self, action: Action, state: AgentState) -> bool:
        """No per-action energy mutation — occupancy is recomputed in tick().

        Returns False so other systems (Career/Health) also see the action.
        """
        return False

    # ------------------------------------------------------------------
    # Occupancy
    # ------------------------------------------------------------------

    def _occupancy(self, state: AgentState) -> float:
        """Total energy occupied by work + active training/upskill/intensive.

        ``rest`` (via ``resting_this_month``) offsets the non-work occupancy
        but never the job's structural occupancy.
        """
        occ = self._work_occupancy(state)
        if state.resting_this_month:
            return occ
        if state.training_months_remaining > 0:
            occ += self.training_footprint
        if state.upskill_months_remaining > 0:
            occ += self.upskill_footprint
        if state.intensive_work_months_remaining > 0:
            occ += self.intensive_footprint
        return occ

    def _work_occupancy(self, state: AgentState) -> float:
        """The current job's structural occupancy (0 when unemployed)."""
        if not state.occupation_id or state.job_status != JobStatus.EMPLOYED:
            return 0.0
        occ = self._career.get_occupation(state.occupation_id)
        return getattr(occ, "energy_footprint", 0.5)
