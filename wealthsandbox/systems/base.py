"""System abstract base class for WealthSandBox subsystems.

Each subsystem (career, health, living expenses, etc.) implements this protocol
so the environment can drive them uniformly:

* ``tick()`` — automatic monthly processing (always called).
* ``handle_action()`` — agent-initiated intervention (only when agent calls a tool).
* ``check_dead()`` — declare a terminal condition (called after tick).

This design enables clean ablation experiments: removing a system from the
environment's ``systems`` list completely eliminates that pressure from the
simulation with no other code changes required.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from wealthsandbox.types import AgentState, Action


class BaseSystem(ABC):
    """A pluggable sandbox subsystem.

    Subclasses implement four methods (two required, two optional):

    * ``tick(state, macro)`` — called **every month in Phase 1** (pre-action).
      This is where income arrives, timers advance, and pre-action resource
      changes happen.
    * ``handle_action(action, state)`` — called **only when the agent chooses
      to intervene** this month (Phase 2).  Returns ``True`` if the subsystem
      consumed the action.
    * ``finalize(state, macro)`` — called **every month in Phase 3** (post-action,
      post calendar advance).  This is where expenses are deducted, health
      declines, and age increments — things that should happen AFTER the
      agent's action but BEFORE the termination check.
    * ``check_dead(state)`` — called **after finalize** each month.  Returns a
      termination reason string or ``None``.  The first system to return
      non-None terminates the episode.  Default: never dies.
    """

    @abstractmethod
    def tick(self, state: AgentState, macro: Dict[str, Any]) -> None:
        """Monthly automatic processing — Phase 1 (pre-action)."""
        ...

    @abstractmethod
    def handle_action(self, action: Action, state: AgentState) -> bool:
        """Process an agent-initiated action — Phase 2.  Return True if consumed."""
        ...

    def finalize(self, state: AgentState, macro: Dict[str, Any]) -> None:
        """Post-action processing — Phase 3 (after macro.step()).

        Override for deductions, health decline, aging, etc. — things that
        should happen after the agent's action but before termination check.
        Default: no-op.
        """
        pass

    def check_dead(self, state: AgentState) -> Optional[str]:
        """Return a termination reason if the episode should end, else None.

        Called after ``finalize()``.  The first system to return non-None wins.
        Override in systems that can kill the agent (bankruptcy, death, age).
        """
        return None
