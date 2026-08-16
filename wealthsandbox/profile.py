"""AgentProfile: initial attributes of the agent (individual, not environment rules).

Separated from ``EnvConfig`` so that changing the agent's starting conditions
(age, cash, health, skill) does not require touching the environment's cost
parameters, and vice versa.  This also makes it easy to compare different
agent profiles under the same economic regime in ablation studies.
"""

from dataclasses import dataclass, field


@dataclass
class AgentProfile:
    """Starting attributes for an agent entering the simulation.

    All of these are **initial values** — the environment (via systems) will
    mutate the actual ``AgentState`` over time.  This profile is only used
    during ``env.reset()`` to seed the initial conditions.

    Attributes:
        age: Starting age in years.
        initial_cash: Starting liquid cash balance.
        initial_health: Starting health (1.0 = full).
        initial_energy: Starting energy/vitality (1.0 = fully rested).
        initial_skill: Starting skill level (1 = entry-level).
    """

    age: int = 20
    initial_cash: float = 10_000.0
    initial_health: float = 1.0
    initial_energy: float = 1.0
    initial_general_skill: int = 1
