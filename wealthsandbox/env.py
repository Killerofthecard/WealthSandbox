"""WealthSandBoxEnv: the main Gym-like environment orchestrator.

Monthly step cycle (three-phase):
    Phase 1 — TICK:   all systems run their automatic monthly logic (income,
                       timers, energy, health decline, living expenses, aging).
                       Runs once per month; retries skip this phase.
    Phase 2 — ACTION: validate the agent's chosen action, reject with feedback
                       or execute via system.handle_action().
    Phase 3 — FINALISE: advance calendar, check termination across all systems,
                        floor cash, archive snapshot, build observation.

All game logic lives in pluggable System objects.  env.step() is a pure
scheduler — it contains no hardcoded resource changes or termination rules.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from wealthsandbox.config import EnvConfig
from wealthsandbox.macro_layer import MacroLayer
from wealthsandbox.micro_layer import MicroLayer
from wealthsandbox.types import Action, Observation, AgentState, CareerMove
from wealthsandbox.agents.tools import ToolCall, SWITCH_OCCUPATION, UPSKILL, INTENSIVE_WORK, QUIT_JOB, DEPOSIT, WITHDRAW, BORROW, REPAY, BUY_STOCK, SELL_STOCK
from wealthsandbox.systems.base import BaseSystem
from wealthsandbox.systems.career import CareerSystem
from wealthsandbox.systems.living import LivingExpenseSystem
from wealthsandbox.systems.bank import BankSystem
from wealthsandbox.systems.health import HealthSystem
from wealthsandbox.systems.aging import AgingSystem
from wealthsandbox.systems.energy import EnergySystem
from wealthsandbox.systems.asset import AssetSystem
from wealthsandbox.validator import ActionValidator


# ---------------------------------------------------------------------------
# Rejection-message extraction (shared helper)
# ---------------------------------------------------------------------------

def _extract_rejection(events: List[str]) -> str:
    """Scan a list of events for a rejection message and return a human-readable reason.

    Two sources of rejection events:
    1. Validator-generated: ``"rejected:<human-readable message>"``  (primary path)
    2. Direct process_* rejections: ``"switch_rejected_..."`` etc. (safety net)
    """
    # Validator messages take precedence (they are more detailed)
    for ev in events:
        if ev.startswith("rejected:"):
            return ev[len("rejected:"):]

    # Fallback: legacy event-key mapping
    for keyword, msg in [
        ("switch_rejected_insufficient_cash", "Insufficient cash to switch occupation"),
        ("switch_rejected_skill_too_low", "Your general skill is too low for that occupation"),
        ("switch_rejected_health_too_low", "Your health is too low for that occupation"),
        ("switch_rejected_already_training", "You are already training for another occupation"),
        ("switch_occupation_no_target", "No target occupation specified"),
        ("switch_occupation_invalid", "That occupation does not exist"),
        ("upskill_rejected_insufficient_cash", "Insufficient cash to upskill"),
        ("upskill_rejected_at_max_skill", "Already at maximum general skill"),
        ("upskill_rejected_already_in_progress", "An upskill is already in progress"),
        ("rejected_energy_too_low", "Not enough energy — rest first"),
        ("intensive_work_rejected_not_employed", "Must be employed to do intensive work"),
        ("intensive_work_rejected_at_max_occ_skill", "Already at maximum occupation skill"),
        ("intensive_work_rejected_already_in_progress", "Intensive work already in progress"),
        ("quit_rejected_not_employed", "You are not currently employed"),
        ("forced_out_health", "Health too low — forced to resign"),
    ]:
        for ev in events:
            if keyword in ev:
                return msg
    return ""


# ---------------------------------------------------------------------------
# Default system factory — builds the standard set of five subsystems.
# ---------------------------------------------------------------------------

def _build_default_systems(config: EnvConfig) -> List[BaseSystem]:
    """Create the standard system list from *config*.

    Systems are ordered so that income arrives before expenses are deducted:
      1. CareerSystem   — auto-work income, upskill/training timers
      2. AssetSystem    — stock settlement + mark-to-market
      3. EnergySystem   — energy drain during training, recovery, upskill cost
      4. LivingExpenseSystem — deduct monthly living expense (with stock fallback)
      5. BankSystem     — savings/loan interest
      6. HealthSystem   — apply age-accelerated health decline
      7. AgingSystem    — increment age on birthdays

    Order within tick matters: income must land in cash before living-expense
    checks for bankruptcy.  AssetSystem runs after Career so settlements arrive
    in cash before expenses.  LivingExpenseSystem gets a reference to
    AssetSystem for forced liquidation as a last resort.
    """
    asset = AssetSystem(
        forced_sale_discount=config.forced_sale_discount,
    )
    return [
        CareerSystem(
            occupations=None,  # use DEFAULT_OCCUPATIONS
            upskill_cost=config.upskill_cost,
            upskill_months=config.upskill_months,
            upskill_skill_boost=config.upskill_skill_boost,
            max_general_skill=config.max_skill_level,
            max_occ_skill=config.max_skill_level,
            switch_base_cost=config.switch_occupation_base_cost,
            living_expense=config.monthly_living_expense,
            intensive_work_months=config.intensive_work_months,
            occ_skill_passive_months=config.occ_skill_passive_months,
            layoff_base_rate=config.layoff_base_rate,
            seed=config.seed,
        ),
        asset,
        EnergySystem(
            cost_per_upskill=config.energy_cost_per_upskill,
            decline_per_training_month=config.energy_decline_per_training_month,
            recovery_per_month=config.energy_recovery_per_month,
        ),
        LivingExpenseSystem(
            monthly_living_expense=config.monthly_living_expense,
            asset_system=asset,
        ),
        BankSystem(),
        HealthSystem(
            decline_20_29=config.health_decline_20_29,
            decline_30_39=config.health_decline_30_39,
            decline_40_49=config.health_decline_40_49,
            decline_50_plus=config.health_decline_50_plus,
        ),
        AgingSystem(end_age=config.end_age),
    ]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class WealthSandBoxEnv:
    """Career & life decision sandbox — pluggable systems, pure scheduler.

    Exposes a minimal Gym-like API:
        reset(seed) -> Observation
        step(action, tool_calls) -> (Observation, reward, done, info)

    All game logic is delegated to ``self.systems`` (a list of ``BaseSystem``
    instances).  The environment itself only does:
      * Calendar advancement (MacroLayer)
      * Action parsing & validation dispatch
      * Termination polling across systems
      * Observation assembly
      * History archiving
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        systems: Optional[List[BaseSystem]] = None,
    ):
        """Create the environment.

        Args:
            config: Simulation parameters.  Defaults to ``EnvConfig()``.
            systems: Custom list of subsystems.  If ``None``, the standard
                five-system layout (career, energy, living, health, aging) is
                built from *config*.  Pass an explicit list for ablation
                experiments — e.g. ``[CareerSystem(...)]`` for a pure career
                sandbox with no survival pressure.
        """
        if config is None:
            config = EnvConfig()
        self.config = config

        self.macro = MacroLayer(config)
        self.micro = MicroLayer(config.profile)

        # Build or accept system list
        if systems is not None:
            self.systems = systems
        else:
            self.systems = _build_default_systems(config)

        # Find the CareerSystem for observation building and validation.
        # Validator only guards career actions, so it only needs CareerSystem.
        self.career = _find_system(self.systems, CareerSystem)
        if self.career is None:
            raise ValueError(
                "Systems list must contain a CareerSystem for action "
                "validation and observation assembly."
            )
        self.validator = ActionValidator(
            self.career,
            energy_threshold=config.energy_threshold_for_upskill,
        )

        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the environment to the initial state.

        Args:
            seed: Optional RNG seed.

        Returns:
            The initial Observation.
        """
        if seed is not None:
            self.config.seed = seed
        self.micro.reset()
        self.macro = MacroLayer(self.config)
        self.history.clear()
        return self._make_observation()

    def step(
        self,
        action: Optional[Action] = None,
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Advance one month.

        Three-phase cycle (reordered so agent sees post-tick state):

        Phase 1 — EXECUTE: validate actions on simulated state, then execute
                 on real state.  Rejection returns immediately with no side
                 effects (no tick, no expense deduction, no month advance).
        Phase 2 — TICK:    all systems run automatic monthly logic (income,
                            layoff, timers, energy, health check).
        Phase 3 — FINALISE: deduct expenses, bank interest, health decline,
                            advance calendar, check termination.

        Observation is built AFTER tick+finalise, so the agent always sees
        the freshest state (including layoffs that just happened).
        """
        import copy

        if tool_calls is not None:
            actions: List[Action] = self.apply_tool_calls(tool_calls)
        elif action is not None:
            actions = [action]
        else:
            actions = [Action()]

        state = self.micro.state
        state.last_month_events.clear()

        # ---- Phase 1: validate + execute actions -------------------------
        # Validate on a deepcopy so rejections leave real state untouched.
        sim_state = copy.deepcopy(state)
        for a in actions:
            if a.career_move == CareerMove.NONE:
                continue
            result = self._validate_action(a, sim_state)
            if not result.allowed:
                state.last_month_events.append("rejected:" + result.message)
                # Build observation from (unchanged) real state — no tick,
                # no expenses, no month advance.
                obs = self._make_observation()
                return obs, 0.0, False, {
                    "action_rejected": True,
                    "rejection_message": result.message,
                }
            for sys in self.systems:
                sys.handle_action(a, sim_state)

        # Execute on real state (all validations passed).
        for a in actions:
            if a.career_move == CareerMove.NONE:
                continue
            for sys in self.systems:
                sys.handle_action(a, state)

        # ---- Phase 2: tick (income, layoff, timers, energy) --------------
        # Runs AFTER actions so the agent's next observation reflects any
        # layoff / income / timer events that just occurred.
        macro_snapshot = self.macro.snapshot()
        for sys in self.systems:
            sys.tick(state, macro_snapshot)

        # ---- Phase 3: finalise (expenses, health, calendar, death) --------
        self.macro.step()
        macro_post = self.macro.snapshot()
        for sys in self.systems:
            sys.finalize(state, macro_post)

        done = False
        reason = ""
        for sys in self.systems:
            dead_reason = sys.check_dead(state)
            if dead_reason is not None:
                done = True
                reason = dead_reason
                break

        state.cash = max(0.0, state.cash)

        # ---- Phase 4: archive + observation --------------------------------
        self.history.append({
            "month": self.macro.total_months,
            "year": self.macro.year,
            "cal_month": self.macro.month,
            "age": state.age,
            "cash": round(state.cash, 2),
            "savings": round(state.savings, 2),
            "loan_balance": round(state.loan_balance, 2),
            "stock_value": round(state.stock_value, 2),
            "pending_settlement": round(state.pending_settlement, 2),
            "total_invested": round(state.total_invested, 2),
            "last_month_stock_return": round(state.last_month_stock_return, 6),
            "energy": round(state.energy, 3),
            "occupation_id": state.occupation_id,
            "general_skill": state.general_skill,
            "occ_skill": state.occupation_skills.get(state.occupation_id, 0),
            "tenure_months": state.tenure_months,
            "current_tier": state.current_tier,
            "monthly_after_tax_income": round(state.monthly_after_tax_income, 2),
            "job_status": state.job_status.value,
            "health": round(state.health, 3),
            "upskill_months_remaining": state.upskill_months_remaining,
            "intensive_work_months_remaining": state.intensive_work_months_remaining,
            "training_months_remaining": state.training_months_remaining,
            "events": list(state.last_month_events),
            "macro": {
                "unrate": round(self.macro.unrate, 4),
                "usrecm": self.macro.usrecm,
                "fedfunds": round(self.macro.fedfunds, 2),
                "cycle_label": getattr(self.macro, "current_cycle_label", ""),
                "cycle_file": getattr(self.macro, "current_cycle_file", ""),
                "economy_status": _economy_status(self.macro.unrate, self.macro.usrecm),
            },
        })

        obs = self._make_observation()
        reward = state.monthly_after_tax_income
        info: Dict[str, Any] = {"termination_reason": reason}
        return obs, reward, done, info

    def _validate_action(self, action: Action, state: AgentState):
        """Dispatch to the appropriate action-specific validator."""
        move = action.career_move
        if move == CareerMove.SWITCH_OCCUPATION:
            return self.validator.validate_switch_target(action, state)
        elif move == CareerMove.DEPOSIT:
            return self.validator.validate_deposit(action, state)
        elif move == CareerMove.WITHDRAW:
            return self.validator.validate_withdraw(action, state)
        elif move == CareerMove.BORROW:
            return self.validator.validate_borrow(action, state)
        elif move == CareerMove.REPAY:
            return self.validator.validate_repay(action, state)
        elif move == CareerMove.BUY_STOCK:
            return self.validator.validate_buy_stock(action, state)
        elif move == CareerMove.SELL_STOCK:
            return self.validator.validate_sell_stock(action, state)
        return self.validator.validate(action, state)

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Return the most recent archived snapshot."""
        if not self.history:
            return self.micro.snapshot()
        return self.history[-1]

    def render(self, mode: str = "text") -> str:
        """Text rendering of the current state."""
        if mode != "text":
            raise NotImplementedError(f"Render mode {mode} not supported.")
        s = self.micro.state
        occ_name = ""
        tier_name = ""
        if self.career.has_occupation(s):
            occ = self.career.get_occupation(s.occupation_id)
            occ_name = f" ({occ.display_name})"
            idx = s.current_tier
            if idx < len(occ.tiers):
                tier_name = occ.tiers[idx].name + " "
        occ_skill = s.occupation_skills.get(s.occupation_id, 0)
        lines = [
            f"Year {self.macro.year} Month {self.macro.month:02d} | Age {s.age}",
            f"  Job: {s.job_status.value} | Occupation: {tier_name}{s.occupation_id or 'none'}{occ_name}",
            f"  General skill: {s.general_skill} | Occ skill: {occ_skill} | Tenure: {s.tenure_months}mo",
            f"  Monthly income: ${s.monthly_after_tax_income:,.2f} | Cash: ${s.cash:,.2f}",
            f"  Health: {s.health:.3f} | Energy: {s.energy:.3f}",
            f"  Upskill left: {s.upskill_months_remaining} | Intensive: {s.intensive_work_months_remaining} | Training: {s.training_months_remaining}",
            f"  Events: {s.last_month_events}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Trajectory export
    # ------------------------------------------------------------------

    def save_trajectory(self, filepath: str) -> None:
        """Save the full step-by-step history as a JSON file.

        Each entry records the agent's state and events for one month.
        Includes initial state (computed from config) and final summary.
        """
        initial_state = {
            "age": self.config.profile.age,
            "health": round(self.config.profile.initial_health, 3),
            "energy": round(self.config.profile.initial_energy, 3),
            "cash": round(self.config.profile.initial_cash, 2),
            "savings": 0.0,
            "loan_balance": 0.0,
            "stock_value": 0.0,
            "pending_settlement": 0.0,
            "total_invested": 0.0,
            "net_worth": round(self.config.profile.initial_cash, 2),
            "occupation_id": "",
            "general_skill": self.config.profile.initial_general_skill,
            "occupation_skills": {},
            "tenure_months": 0,
            "current_tier": 0,
            "monthly_after_tax_income": 0.0,
            "job_status": "unemployed",
            "upskill_months_remaining": 0,
            "intensive_work_months_remaining": 0,
            "training_months_remaining": 0,
            "training_target_occupation": "",
        }
        final_summary = None
        if self.history:
            last = self.history[-1]
            final_summary = {
                "months_played": len(self.history),
                "final_cash": last["cash"],
                "final_occupation_id": last["occupation_id"],
                "final_general_skill": last.get("general_skill", 0),
                "final_occ_skill": last.get("occ_skill", 0),
                "final_health": last["health"],
                "final_job_status": last["job_status"],
                "age": last["age"],
            }
        # Enumerate which systems were active
        active_systems = [type(s).__name__ for s in self.systems]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "end_age": self.config.end_age,
                        "seed": self.config.seed,
                        "monthly_living_expense": self.config.monthly_living_expense,
                        "upskill_cost": self.config.upskill_cost,
                        "upskill_months": self.config.upskill_months,
                        "upskill_skill_boost": self.config.upskill_skill_boost,
                        "max_skill_level": self.config.max_skill_level,
                        "switch_occupation_base_cost": self.config.switch_occupation_base_cost,
                        # health (age-accelerated decline)
                        "health_decline_20_29": self.config.health_decline_20_29,
                        "health_decline_30_39": self.config.health_decline_30_39,
                        "health_decline_40_49": self.config.health_decline_40_49,
                        "health_decline_50_plus": self.config.health_decline_50_plus,
                        # energy (stamina / pacing)
                        "energy_cost_per_upskill": self.config.energy_cost_per_upskill,
                        "energy_decline_per_training_month": self.config.energy_decline_per_training_month,
                        "energy_recovery_per_month": self.config.energy_recovery_per_month,
                        "energy_threshold_for_upskill": self.config.energy_threshold_for_upskill,
                        "macro_cycle": self.config.macro_cycle or "random",
                    },
                    "profile": {
                        "age": self.config.profile.age,
                        "initial_cash": self.config.profile.initial_cash,
                        "initial_health": self.config.profile.initial_health,
                        "initial_energy": self.config.profile.initial_energy,
                        "initial_general_skill": self.config.profile.initial_general_skill,
                    },
                    "active_systems": active_systems,
                    "initial_state": initial_state,
                    "total_steps": len(self.history),
                    "trajectory": self.history,
                    "final_summary": final_summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        return filepath

    # ------------------------------------------------------------------
    # Tool-call parsing
    # ------------------------------------------------------------------

    @staticmethod
    def apply_tool_calls(tool_calls: List[ToolCall]) -> List[Action]:
        """Convert LLM tool calls into a list of environment Actions.

        Each tool call becomes one Action.  Multiple tools can be called in
        one month — e.g. ``deposit(3000)`` + ``upskill()``.
        """
        actions: List[Action] = []
        seen: set = set()
        for tc in tool_calls:
            if tc.tool_name in seen:
                continue
            seen.add(tc.tool_name)
            params = tc.parameters or {}

            if tc.tool_name == SWITCH_OCCUPATION:
                actions.append(Action(
                    career_move=CareerMove.SWITCH_OCCUPATION,
                    target_occupation_id=params.get("occupation_id", ""),
                ))
            elif tc.tool_name == UPSKILL:
                actions.append(Action(career_move=CareerMove.UPSKILL))
            elif tc.tool_name == INTENSIVE_WORK:
                actions.append(Action(career_move=CareerMove.INTENSIVE_WORK))
            elif tc.tool_name == QUIT_JOB:
                actions.append(Action(career_move=CareerMove.QUIT_JOB))
            elif tc.tool_name == DEPOSIT:
                actions.append(Action(
                    career_move=CareerMove.DEPOSIT,
                    amount=float(params.get("amount", 0)),
                ))
            elif tc.tool_name == WITHDRAW:
                actions.append(Action(
                    career_move=CareerMove.WITHDRAW,
                    amount=float(params.get("amount", 0)),
                ))
            elif tc.tool_name == BORROW:
                actions.append(Action(
                    career_move=CareerMove.BORROW,
                    amount=float(params.get("amount", 0)),
                ))
            elif tc.tool_name == REPAY:
                actions.append(Action(
                    career_move=CareerMove.REPAY,
                    amount=float(params.get("amount", 0)),
                ))
            elif tc.tool_name == BUY_STOCK:
                actions.append(Action(
                    career_move=CareerMove.BUY_STOCK,
                    amount=float(params.get("amount", 0)),
                ))
            elif tc.tool_name == SELL_STOCK:
                actions.append(Action(
                    career_move=CareerMove.SELL_STOCK,
                    amount=float(params.get("amount", 0)),
                ))

        if not actions:
            actions.append(Action(career_move=CareerMove.NONE))
        return actions

    # ------------------------------------------------------------------
    # Observation assembly
    # ------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        """Build the Observation from current state and macro."""
        macro_snapshot = self.macro.snapshot()
        # Inject industry income data
        macro_snapshot["industry_incomes"] = self.career.get_industry_incomes()
        # Show occupation details only when agent is choosing a career (unemployed)
        if self.micro.state.job_status.value == "unemployed":
            macro_snapshot["available_occupations"] = self.career.get_occupation_details()
        macro_snapshot["economy_status"] = _economy_status(
            macro_snapshot.get("unrate", 0.05),
            macro_snapshot.get("usrecm", 0),
        )
        macro_snapshot["switch_base_cost"] = self.career.switch_base_cost
        macro_snapshot["living_expense"] = self.career.living_expense
        # Cash already includes this month's income (tick ran before observation).
        # No projection needed — use actual state for available_actions.
        macro_snapshot["available_actions"] = self.validator.available_actions(
            self.micro.state
        )

        occ_name = ""
        tier_name = ""
        if self.career.has_occupation(self.micro.state):
            occ = self.career.get_occupation(self.micro.state.occupation_id)
            occ_name = occ.display_name
            idx = self.micro.state.current_tier
            if idx < len(occ.tiers):
                tier_name = occ.tiers[idx].name

        occ_skill = self.micro.state.occupation_skills.get(
            self.micro.state.occupation_id, 1
        ) if self.micro.state.occupation_id else 0

        narrative = (
            f"Month {self.macro.month}/{self.macro.year}. "
            f"Age {self.micro.state.age}, "
            f"{self.micro.state.job_status.value}"
        )
        if occ_name:
            narrative += (
                f" as {tier_name} {occ_name} "
                f"(gen_skill {self.micro.state.general_skill}, "
                f"occ_skill {occ_skill}, "
                f"tenure {self.micro.state.tenure_months}mo). "
                f"Monthly take-home: ${self.micro.state.monthly_after_tax_income:,.0f}. "
            )
        else:
            narrative += (
                f" with no occupation "
                f"(gen_skill {self.micro.state.general_skill}). "
            )
        narrative += f"Cash: ${self.micro.state.cash:,.0f}. "
        if self.micro.state.stock_value > 0:
            ret_pct = self.micro.state.last_month_stock_return * 100
            pnl = self.micro.state.stock_value - self.micro.state.total_invested
            narrative += (
                f"Stocks: ${self.micro.state.stock_value:,.0f} "
                f"({ret_pct:+.1f}% last month; "
                f"invested ${self.micro.state.total_invested:,.0f}, "
                f"P&L: ${pnl:+,.0f}). "
            )
        if self.micro.state.pending_settlement > 0:
            narrative += (
                f"Pending settlement: ${self.micro.state.pending_settlement:,.0f} "
                f"(available next month). "
            )
        net_worth = (
            self.micro.state.cash + self.micro.state.savings
            + self.micro.state.stock_value + self.micro.state.pending_settlement
            - self.micro.state.loan_balance
        )
        narrative += (
            f"Net worth: ${net_worth:,.0f}. "
        )
        narrative += f"Energy: {self.micro.state.energy:.0%}. "
        narrative += f"Health: {self.micro.state.health:.3f}. "
        if self.micro.state.upskill_months_remaining > 0:
            narrative += (
                f"Upskilling (general): "
                f"{self.micro.state.upskill_months_remaining} months remaining. "
            )
        if self.micro.state.intensive_work_months_remaining > 0:
            narrative += (
                f"Intensive work (occ_skill): "
                f"{self.micro.state.intensive_work_months_remaining} months remaining. "
            )
        if self.micro.state.training_months_remaining > 0:
            narrative += (
                f"Training for {self.micro.state.training_target_occupation}: "
                f"{self.micro.state.training_months_remaining} months remaining. "
            )

        # Highlight rejected actions prominently
        rejection = _extract_rejection(self.micro.state.last_month_events)
        if rejection:
            narrative = f"⚠️ LAST ACTION REJECTED: {rejection}. " + narrative

        return Observation(
            individual=self.micro.snapshot(),
            macro=macro_snapshot,
            narrative=narrative,
            year=self.macro.year,
            month=self.macro.month,
            done=False,
            info={"total_months": self.macro.total_months},
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _economy_status(unrate: float, usrecm: int) -> str:
    """Qualitative economic description — no raw numbers exposed to agent."""
    if usrecm == 1:
        return (
            "The economy is in RECESSION. Incomes are reduced across most "
            "industries. Layoffs are common and finding a new job is difficult. "
            "Conserve cash and avoid unnecessary spending."
        )
    elif unrate > 0.08:
        return (
            "The economy is WEAK — unemployment is high. Jobs are hard to "
            "find and incomes are under pressure. Build an emergency fund."
        )
    elif unrate > 0.05:
        return (
            "The economy is SLUGGISH — unemployment is above normal. "
            "Proceed with moderate caution."
        )
    else:
        return (
            "The economy is HEALTHY — unemployment is low and jobs are "
            "plentiful. A good time to invest in your career."
        )


def _find_system(
    systems: List[BaseSystem],
    system_type: type,
) -> Optional[BaseSystem]:
    """Return the first system in *systems* that is an instance of *system_type*."""
    for sys in systems:
        if isinstance(sys, system_type):
            return sys
    return None
