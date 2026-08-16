#!/usr/bin/env python3
"""WealthSandBox CLI entry point (simplified — career only).

Run an LLM agent (or random/mock agent) through the WealthSandBox career
simulation with monthly decisions.

Usage:
    # Mock agent (deterministic, no API calls) — 12 months
    python run_wealthsandbox.py --agent mock --months 12

    # Random baseline agent — 24 months
    python run_wealthsandbox.py --agent random --months 24 --seed 42

    # LLM agent — 6 months
    python run_wealthsandbox.py --agent llm --months 6

Environment variables (loaded from .env):
    OPENAI_API_KEY, OPENAI_BASE_URL, DEFAULT_MODEL
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# 项目根目录（脚本所在目录），用于锚定默认轨迹输出路径，避免受运行时 cwd 影响
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from wealthsandbox import WealthSandBoxEnv, EnvConfig
from wealthsandbox.agents import LLMAgent, Decision, ToolCall
from wealthsandbox.types import Action, CareerMove


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def section(title: str) -> None:
    print(f"\n{C.BOLD}{C.YELLOW}▶ {title}{C.RESET}")
    print(f"{C.DIM}{'─' * 60}{C.RESET}")


def money(val: float) -> str:
    s = f"${val:>10,.0f}"
    return f"{C.RED}{s}{C.RESET}" if val < 0 else f"{C.GREEN}{s}{C.RESET}"


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class RandomAgent:
    """Stochastic baseline agent — randomly selects from all available tools each month.

    Uses the ``available_actions`` dict from the observation to only attempt
    actions that are currently legal, avoiding wasted rejection months.
    """

    # Actions that take no arguments (besides None)
    _NO_ARG_ACTIONS = ["upskill", "intensive_work", "quit_job"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.occupations = [
            "software_engineer", "data_scientist", "investment_banker",
            "financial_analyst", "manufacturing_worker", "nurse", "civil_servant",
        ]
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def decide(self, obs) -> Decision:
        self._step += 1
        ind = obs.individual
        # Available actions as reported by the validator — only pick from legal ones
        available = obs.macro.get("available_actions", {})
        legal = [
            name for name, info in available.items()
            if isinstance(info, dict) and info.get("allowed")
        ]
        # Filter out "none" — we build that separately
        legal = [n for n in legal if n != "none"]

        tool_calls: list = []
        # ~40% of the time, do nothing (auto-work).  Otherwise pick 1–2 random
        # actions from the legal set.
        if legal and self.rng.random() > 0.40:
            # Pick 1–2 distinct actions
            num = self.rng.choices([1, 2], weights=[0.7, 0.3])[0]
            picks = self.rng.sample(legal, min(num, len(legal)))
            for name in picks:
                if name == "switch_occupation":
                    occ = self.rng.choice(self.occupations)
                    tool_calls.append(ToolCall(
                        tool_name="switch_occupation",
                        parameters={"occupation_id": occ},
                    ))
                elif name == "deposit":
                    cash = ind.get("cash", 0)
                    # Deposit a random fraction of cash above $2,000 buffer
                    max_deposit = max(0, cash - 2_000)
                    if max_deposit > 0:
                        amount = round(self.rng.uniform(100, max_deposit), 2)
                        tool_calls.append(ToolCall(
                            tool_name="deposit",
                            parameters={"amount": amount},
                        ))
                elif name == "withdraw":
                    savings = ind.get("savings", 0)
                    if savings > 0:
                        amount = round(self.rng.uniform(10, savings), 2)
                        tool_calls.append(ToolCall(
                            tool_name="withdraw",
                            parameters={"amount": amount},
                        ))
                elif name == "borrow":
                    income = ind.get("monthly_after_tax_income", 0)
                    limit = max(8_000, income * 12)
                    amount = round(self.rng.uniform(500, max(1_000, limit * 0.5)), 2)
                    tool_calls.append(ToolCall(
                        tool_name="borrow",
                        parameters={"amount": amount},
                    ))
                elif name == "repay":
                    loan = ind.get("loan_balance", 0)
                    cash = ind.get("cash", 0)
                    if loan > 0 and cash > 0:
                        amount = round(self.rng.uniform(10, min(cash, loan)), 2)
                        tool_calls.append(ToolCall(
                            tool_name="repay",
                            parameters={"amount": amount},
                        ))
                elif name in self._NO_ARG_ACTIONS:
                    tool_calls.append(ToolCall(tool_name=name, parameters={}))

        reason = ", ".join(tc.tool_name for tc in tool_calls) if tool_calls else "none"
        return Decision(
            reasoning=f"[step {self._step}] random choices: {reason}",
            tool_calls=tool_calls,
        )


class MockAgent:
    """Deterministic agent for dry-runs (no API calls)."""

    def __init__(self):
        self.step_count = 0

    def reset(self) -> None:
        self.step_count = 0

    def decide(self, obs) -> Decision:
        self.step_count += 1
        ind = obs.individual
        occ = ind.get("occupation_id", "")

        # Step 1: pick manufacturing_worker (no skill barrier, no training)
        if not occ:
            return Decision(
                reasoning="I am unemployed. Let me take a low-barrier job first.",
                tool_calls=[
                    ToolCall(
                        tool_name="switch_occupation",
                        parameters={"occupation_id": "manufacturing_worker"},
                    )
                ],
            )

        # Step 7: upskill once enough savings are built (need cash ≥ $7,000 to
        # cover the $5,000 cost + $2,000 living expense safely)
        if self.step_count >= 7 and ind.get("upskill_months_remaining", 0) == 0 \
           and ind.get("cash", 0) >= 7_000:
            return Decision(
                reasoning="I have some cash saved. Let me invest in my skills.",
                tool_calls=[ToolCall(tool_name="upskill", parameters={})],
            )

        # Default: auto-work — no tool call needed
        return Decision(
            reasoning="Continue working and earning automatically.",
            tool_calls=[],
        )


# ---------------------------------------------------------------------------
# Printing routines
# ---------------------------------------------------------------------------
def print_step_header(step: int, age: int) -> None:
    print(f"\n{C.DIM}{'─' * 80}{C.RESET}")
    print(
        f"{C.BOLD}🔄 Step {step:>3}  |  🎂 Age {age}{C.RESET}"
    )
    print(f"{C.DIM}{'─' * 80}{C.RESET}")


def print_environment_state(obs) -> None:
    """Print the environment observation in a tidy panel."""
    ind = obs.individual
    macro = obs.macro

    print(f"\n  {C.BOLD}{C.BLUE}🌐 ENVIRONMENT OBSERVATION{C.RESET}")
    print(f"  {C.DIM}{'─' * 50}{C.RESET}")

    # Personal
    health = ind.get("health", 1.0)
    health_bar = "█" * int(health * 10) + "░" * (10 - int(health * 10))
    energy = ind.get("energy", 1.0)
    energy_bar = "█" * int(energy * 10) + "░" * (10 - int(energy * 10))
    print(f"  {C.BOLD}👤 Personal{C.RESET}")
    print(f"    Health:      {health:.3f} {health_bar}")
    print(f"    Energy:      {energy:.3f} {energy_bar}")
    print(f"    Occupation:  {ind.get('occupation_id', 'none') or 'unemployed'}")
    print(f"    Gen Skill:   {ind.get('general_skill', 1)}")
    print(f"    Occ Skill:   {ind.get('occ_skill', 0)}")
    print(f"    Tenure:      {ind.get('tenure_months', 0)} months")
    upskill = ind.get("upskill_months_remaining", 0)
    if upskill > 0:
        print(f"    Upskill:     {upskill} months remaining")
    intensive = ind.get("intensive_work_months_remaining", 0)
    if intensive > 0:
        print(f"    Intensive:   {intensive} months remaining")
    training = ind.get("training_months_remaining", 0)
    if training > 0:
        target = ind.get("training_target_occupation", "?")
        print(f"    Training:    {target} ({training} months remaining)")

    # Financial
    print(f"\n  {C.BOLD}💰 Financial{C.RESET}")
    print(f"    Monthly Income: {money(ind.get('monthly_after_tax_income', 0))}")
    print(f"    Cash:           {money(ind.get('cash', 0))}")

    # Macro
    print(f"\n  {C.BOLD}🌍 Macro{C.RESET}")
    print(f"    Industry average monthly salaries:")
    for industry, salary in macro.get("industry_incomes", {}).items():
        print(f"      {industry}: ${salary:,.0f}")

    # Available occupations (only shown when unemployed)
    avail = macro.get("available_occupations", {})
    if avail:
        print(f"\n  {C.BOLD}💼 Available Occupations{C.RESET}")
        for occ_id, detail in avail.items():
            req = detail.get("min_general_skill", 1)
            min_h = detail.get("min_health", 0.0)
            cost = detail.get("entry_cost", 0)
            train = detail.get("training_months", 0)
            total = 2000 + cost
            tiers = detail.get("tiers", [])
            tier_str = " → ".join(
                f"{t['name']}(×{t['salary_multiplier']:.1f})" for t in tiers
            ) if tiers else ""
            print(f"    • {occ_id}: ${detail['base_monthly_salary']:,.0f}/mo "
                  f"({detail['industry']}, sens={detail['skill_sensitivity']}, "
                  f"gen≥{req}, health≥{min_h:.1f}, ${total:,.0f}+{train}mo)")
            if tier_str:
                print(f"      {tier_str}")

    # Legal actions
    actions = macro.get("available_actions", {})
    if actions:
        print(f"\n  {C.BOLD}✅ Legal Actions{C.RESET}")
        for name, info in actions.items():
            allowed = info.get("allowed", info) if isinstance(info, dict) else info
            reason = info.get("reason", "") if isinstance(info, dict) else ""
            if allowed:
                print(f"    {C.GREEN}✓{C.RESET} {name}")
            else:
                suffix = f"{C.DIM} — {reason}{C.RESET}" if reason else ""
                print(f"    {C.RED}✗{C.RESET} {name}{suffix}")

    # Events
    events = ind.get("last_month_events", [])
    if events:
        print(f"\n  {C.BOLD}📌 Events{C.RESET}")
        for e in events:
            print(f"    {C.DIM}• {e}{C.RESET}")
    print(f"  {C.DIM}{'─' * 50}{C.RESET}")


def print_agent_decision(decision: Decision) -> None:
    """Print the agent's reasoning and chosen tool calls."""
    print(f"\n  {C.BOLD}{C.MAGENTA}🤖 AGENT DECISION{C.RESET}")
    print(f"  {C.DIM}{'─' * 50}{C.RESET}")

    if decision.reasoning:
        print(f"  {C.BOLD}💭 Reasoning{C.RESET}")
        for line in decision.reasoning.split("\n"):
            print(f"    {C.DIM}{line}{C.RESET}")

    if decision.tool_calls:
        print(f"\n  {C.BOLD}🔧 Tool Calls{C.RESET}")
        for tc in decision.tool_calls:
            params = ", ".join(f"{k}={v}" for k, v in tc.parameters.items())
            print(f"    • {C.CYAN}{tc.tool_name}{C.RESET}({params})" if params else f"    • {C.CYAN}{tc.tool_name}{C.RESET}()")
    else:
        print(f"\n  {C.BOLD}🔧 Tool Calls{C.RESET}")
        print(f"    {C.DIM}• (none — auto-work){C.RESET}")
    print(f"  {C.DIM}{'─' * 50}{C.RESET}")


def print_final_summary(env, total_reward: float, months_played: int) -> None:
    """Print the final experiment summary."""
    s = env.micro.state
    print(f"\n{C.GREEN}{C.BOLD}{'═' * 80}{C.RESET}")
    print(f"{C.GREEN}{C.BOLD}║{'🏁 FINAL SUMMARY':^78}║{C.RESET}")
    print(f"{C.GREEN}{C.BOLD}{'═' * 80}{C.RESET}")

    print(f"  {'Final age':24} {s.age}")
    print(f"  {'Months played':24} {months_played}")
    print(f"  {'Total reward (income)':24} {money(total_reward)}")

    print(f"\n  {C.BOLD}💰 Final State{C.RESET}")
    print(f"  {'Cash':24} {money(s.cash)}")
    occ = s.occupation_id or "unemployed"
    print(f"  {'Occupation':24} {occ}")
    print(f"  {'General Skill':24} {s.general_skill}")
    occ_skill = s.occupation_skills.get(s.occupation_id, 0)
    print(f"  {'Occ Skill':24} {occ_skill}")
    print(f"  {'Health':24} {s.health:.3f}")
    print(f"  {'Job Status':24} {s.job_status.value}")
    net_worth = s.cash + s.savings + s.stock_value + s.pending_settlement - s.loan_balance
    price_level = env.macro.price_level or 1.0
    print(f"  {'Net worth (real)':24} {money(net_worth / price_level)}")
    print(f"{C.GREEN}{C.BOLD}{'═' * 80}{C.RESET}")


# ---------------------------------------------------------------------------
# Incremental trajectory writer (JSONL — crash-safe, one line per step)
# ---------------------------------------------------------------------------

class TrajectoryWriter:
    """Writes trajectory data incrementally to a JSONL file, then converts to JSON.

    During the run, each step is flushed to a ``.jsonl`` file (crash-safe).
    On completion, ``finalize_json()`` reads the JSONL and writes a single
    indented ``.json`` file with run metadata at the top.
    """

    def __init__(self, filepath: str, run_meta: Optional[Dict] = None):
        self._filepath = filepath
        self._f = open(filepath, "w", encoding="utf-8")
        self._step_count = 0
        # Run metadata stored but only written in final JSON (keeps JSONL clean)
        self._run_meta = run_meta or {}

    # ---- header ----

    def write_header(
        self,
        env,
        initial_state: Dict,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Write the run metadata as the first line."""
        header: Dict[str, Any] = {
            "type": "header",
            "config": {
                "end_age": env.config.end_age,
                "seed": env.config.seed,
                "monthly_living_expense": env.config.monthly_living_expense,
                "upskill_cost": env.config.upskill_cost,
                "upskill_months": env.config.upskill_months,
                "upskill_skill_boost": env.config.upskill_skill_boost,
                "max_skill_level": env.config.max_skill_level,
                "switch_occupation_base_cost": env.config.switch_occupation_base_cost,
                "health_decline_20_29": env.config.health_decline_20_29,
                "health_decline_30_39": env.config.health_decline_30_39,
                "health_decline_40_49": env.config.health_decline_40_49,
                "health_decline_50_plus": env.config.health_decline_50_plus,
                "energy_cost_per_upskill": env.config.energy_cost_per_upskill,
                "energy_decline_per_training_month": env.config.energy_decline_per_training_month,
                "energy_recovery_per_month": env.config.energy_recovery_per_month,
                "energy_threshold_for_upskill": env.config.energy_threshold_for_upskill,
                "intensive_work_months": env.config.intensive_work_months,
                "occ_skill_passive_months": env.config.occ_skill_passive_months,
                "macro_cycle": env.config.macro_cycle or "random",
                "macro_continuous_file": env.config.macro_continuous_file or "",
            },
            "profile": {
                "age": env.config.profile.age,
                "initial_cash": env.config.profile.initial_cash,
                "initial_health": env.config.profile.initial_health,
                "initial_energy": env.config.profile.initial_energy,
                "initial_general_skill": env.config.profile.initial_general_skill,
            },
            "active_systems": [type(s).__name__ for s in env.systems],
            "initial_state": initial_state,
        }
        if system_prompt is not None:
            header["system_prompt"] = system_prompt
        self._write_line(header)

    # ---- step ----

    def append_step(self, decision: Decision, state_after: Dict) -> None:
        """Append one completed month to the trajectory."""
        self._step_count += 1
        entry: Dict[str, Any] = {
            "type": "step",
            "step": self._step_count,
            "decision": {
                "reasoning": decision.reasoning,
                "tool_calls": [
                    {"tool_name": tc.tool_name, "parameters": tc.parameters}
                    for tc in decision.tool_calls
                ],
            },
            "state_after": state_after,
        }
        self._write_line(entry)

    # ---- summary ----

    def write_summary(self, summary: Dict) -> None:
        """Write the final summary as the last line and close the file."""
        summary["type"] = "summary"
        summary["total_steps"] = self._step_count
        self._write_line(summary)
        self._f.close()

    # ---- finalize to JSON ----

    def finalize_json(self) -> str:
        """Read the JSONL, build a single JSON, write it, and return its path.

        The JSON has ``run`` (timestamp, model, temperature, etc.) at the top,
        then ``config``, ``profile``, ``active_systems``, ``initial_state``,
        ``trajectory`` (array of steps), and ``final_summary``.
        """
        lines = []
        with open(self._filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))

        header = lines[0] if lines else {}
        summary = lines[-1] if len(lines) > 1 and lines[-1].get("type") == "summary" else {}
        steps = [s for s in lines if s.get("type") == "step"]

        result: Dict[str, Any] = {
            "run": self._run_meta,
            "config": header.get("config", {}),
            "profile": header.get("profile", {}),
            "active_systems": header.get("active_systems", []),
            "initial_state": header.get("initial_state", {}),
            "system_prompt": header.get("system_prompt", ""),
            "total_steps": len(steps),
            "trajectory": [
                {
                    "step": s["step"],
                    "decision": s["decision"],
                    "state_after": s["state_after"],
                }
                for s in steps
            ],
            "final_summary": {
                k: v for k, v in summary.items()
                if k not in ("type", "total_steps")
            },
        }

        json_path = self._filepath.rsplit(".", 1)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        return json_path

    # ---- internal ----

    def _write_line(self, obj: Dict) -> None:
        line = json.dumps(obj, ensure_ascii=False, default=str)
        self._f.write(line + "\n")
        self._f.flush()

    @property
    def step_count(self) -> int:
        return self._step_count


# ---------------------------------------------------------------------------
# Persona profile loader
# ---------------------------------------------------------------------------
PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")


def _load_persona(profile_name: Optional[str]) -> str:
    """Load a persona markdown file from the profiles/ directory.

    Args:
        profile_name: Short name without extension (e.g. ``"ambitious"``).
            If ``None`` or empty, returns an empty string (no persona).

    Returns:
        The file contents as a string, or ``""`` if the profile is not found
        or not specified.
    """
    if not profile_name:
        return ""
    # Support both "ambitious" and "ambitious.md"
    fname = profile_name if profile_name.endswith(".md") else f"{profile_name}.md"
    path = os.path.join(PROFILES_DIR, fname)
    if not os.path.isfile(path):
        print(f"Warning: profile '{profile_name}' not found at {path}", file=sys.stderr)
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()



# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def build_agent(args, config=None) -> object:
    if args.agent == "llm":
        from wealthsandbox.agents.llm_agent import _build_system_prompt
        from wealthsandbox.systems.career import DEFAULT_OCCUPATIONS
        profile = config.profile if config else None

        # Load persona profile from file (if specified)
        persona = _load_persona(getattr(args, "profile", None))

        # Build occupation details dict for dynamic tool descriptions
        occupation_details = {
            occ_id: {
                "industry": occ.industry,
                "base_monthly_salary": occ.base_monthly_salary,
                "entry_cost": occ.entry_cost,
                "training_months": occ.training_months,
                "min_general_skill": occ.min_general_skill,
                "min_health": occ.min_health,
                "tiers": [
                    {"name": t.name, "salary_multiplier": t.salary_multiplier}
                    for t in occ.tiers
                ],
            }
            for occ_id, occ in DEFAULT_OCCUPATIONS.items()
        }
        prompt = _build_system_prompt(
            profile=profile,
            end_age=config.end_age if config else 60,
            living_expense=config.monthly_living_expense if config else 2_000.0,
            upskill_cost=config.upskill_cost if config else 5_000.0,
            upskill_months=config.upskill_months if config else 6,
            max_general_skill=config.max_skill_level if config else 10,
            max_occ_skill=config.max_skill_level if config else 10,
            switch_base_cost=config.switch_occupation_base_cost if config else 2_000.0,
            energy_threshold=config.energy_threshold_for_upskill if config else 0.4,
            energy_cost_per_upskill=config.energy_cost_per_upskill if config else 0.4,
            energy_cost_per_intensive_work=config.energy_cost_per_intensive_work if config else 0.5,
            energy_decline_per_training_month=config.energy_decline_per_training_month if config else 0.15,
            intensive_work_months=config.intensive_work_months if config else 3,
            occ_skill_passive_months=config.occ_skill_passive_months if config else 12,
            energy_recovery_per_month=config.energy_recovery_per_month if config else 0.10,
            occupations=occupation_details,
            persona=persona,
            forced_sale_discount=config.forced_sale_discount if config else 0.10,
            min_cash_buffer=config.min_cash_buffer if config else 2_000.0,
            rest_health_gain=config.rest_health_gain if config else 0.02,
            rest_energy_gain=config.rest_energy_gain if config else 0.3,
            rest_income_penalty=config.rest_income_penalty if config else 0.20,
            medical_care_cost=config.medical_care_cost if config else 3_000.0,
            medical_care_health_gain=config.medical_care_health_gain if config else 0.05,
            medical_care_max_per_year=config.medical_care_max_per_year if config else 2,
            health_max=config.health_max if config else 1.0,
        )
        return LLMAgent(
            model=args.model or os.getenv("DEFAULT_MODEL", "gpt-4.1-mini"),
            temperature=args.temperature,
            max_tokens=None,
            system_prompt=prompt,
            living_expense=config.monthly_living_expense if config else 2_000.0,
            upskill_cost=config.upskill_cost if config else 5_000.0,
            upskill_months=config.upskill_months if config else 6,
            max_skill_level=config.max_skill_level if config else 10,
            switch_base_cost=config.switch_occupation_base_cost if config else 2_000.0,
            energy_threshold_for_upskill=config.energy_threshold_for_upskill if config else 0.4,
            occupations=occupation_details,
            persona=persona,
            forced_sale_discount=config.forced_sale_discount if config else 0.10,
            min_cash_buffer=config.min_cash_buffer if config else 2_000.0,
            rest_health_gain=config.rest_health_gain if config else 0.02,
            rest_energy_gain=config.rest_energy_gain if config else 0.3,
            rest_income_penalty=config.rest_income_penalty if config else 0.20,
            medical_care_cost=config.medical_care_cost if config else 3_000.0,
            medical_care_health_gain=config.medical_care_health_gain if config else 0.05,
            medical_care_max_per_year=config.medical_care_max_per_year if config else 2,
            health_max=config.health_max if config else 1.0,
        )
    elif args.agent == "random":
        return RandomAgent(seed=args.seed)
    elif args.agent == "mock":
        return MockAgent()
    raise ValueError(f"Unknown agent type: {args.agent}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an agent through the WealthSandBox career simulation."
    )
    parser.add_argument(
        "--agent",
        choices=["llm", "random", "mock"],
        default="mock",
        help="Agent type to run (default: mock for dry-run).",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Maximum number of months to simulate (default: 12).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--start-age",
        type=int,
        default=20,
        help="Agent starting age (default: 20).",
    )
    parser.add_argument(
        "--end-age",
        type=int,
        default=60,
        help="Agent ending age / termination (default: 60).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (default: $DEFAULT_MODEL or gpt-4.1-mini).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--macro-cycle",
        type=str,
        default="",
        choices=["", "boom", "normal", "recession"],
        help="Lock macroeconomic cycle type (default: random).",
    )
    parser.add_argument(
        "--macro-file",
        type=str,
        default="",
        help="Specific cycle CSV, e.g. '2008_2009.csv'. Overrides --macro-cycle.",
    )
    parser.add_argument(
        "--macro-continuous-file",
        type=str,
        default="",
        help="Single long CSV at raw_data/ root (e.g. '1986_2025.csv') — reads the whole horizon and freezes on exhaustion.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Agent persona profile to load from profiles/ directory (e.g. 'ambitious', 'cautious').",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save trajectory to a JSON file (default: auto-generate in trajectories/).",
    )
    args = parser.parse_args()

    # Build config
    from wealthsandbox.profile import AgentProfile
    env_config = EnvConfig(
        end_age=args.end_age,
        seed=args.seed,
        profile=AgentProfile(age=args.start_age),
        macro_cycle=args.macro_cycle,
        macro_cycle_file=args.macro_file,
        macro_continuous_file=args.macro_continuous_file,
    )

    # Setup
    section("🚀 STARTING EXPERIMENT")
    env = WealthSandBoxEnv(env_config)
    obs = env.reset(seed=args.seed)
    agent = build_agent(args, config=env_config)
    agent.reset()

    # Capture initial state (before any agent decision)
    s = env.micro.state
    initial_state = {
        "age": s.age,
        "health": round(s.health, 3),
        "energy": round(s.energy, 3),
        "cash": round(s.cash, 2),
        "occupation_id": s.occupation_id,
        "general_skill": s.general_skill,
        "occupation_skills": s.occupation_skills,
        "tenure_months": s.tenure_months,
        "current_tier": s.current_tier,
        "monthly_after_tax_income": round(s.monthly_after_tax_income, 2),
        "job_status": s.job_status.value,
        "upskill_months_remaining": s.upskill_months_remaining,
        "intensive_work_months_remaining": s.intensive_work_months_remaining,
        "training_months_remaining": s.training_months_remaining,
        "training_target_occupation": s.training_target_occupation,
        "month": env.macro.total_months,
    }
    # Capture system prompt (LLM agent only)
    system_prompt: Optional[str] = None
    if hasattr(agent, "system_prompt"):
        system_prompt = agent.system_prompt

    print(f"  Agent: {C.BOLD}{args.agent}{C.RESET}")
    print(f"  Seed: {args.seed}")
    print(f"  Horizon: {args.months} months")
    print(f"  Occupations available: software_engineer, data_scientist, "
          f"investment_banker, financial_analyst, manufacturing_worker, "
          f"nurse, civil_servant")

    # ---- Resolve model name ----
    model_name = args.model or os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")
    if args.agent != "llm":
        model_name = args.agent  # "mock" or "random"
    model_slug = model_name.replace("/", "_").replace(" ", "_")

    # ---- Open trajectory file (JSONL crash-safe, finalised to JSON) ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save:
        # 显式指定路径；相对路径也锚定到脚本目录，避免受运行时 cwd 影响
        save_path = args.save if os.path.isabs(args.save) else os.path.join(BASE_DIR, args.save)
        jsonl_path = save_path if save_path.endswith(".jsonl") else save_path + ".jsonl"
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    else:
        traj_dir = os.path.join(BASE_DIR, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        jsonl_path = os.path.join(traj_dir, f"{model_slug}_{ts}.jsonl")

    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "agent_type": args.agent,
        "model": model_name,
        "temperature": getattr(agent, "temperature", None) if args.agent == "llm" else None,
        "max_tokens": getattr(agent, "max_tokens", None) if args.agent == "llm" else None,
        "seed": args.seed,
        "macro_cycle": args.macro_cycle or "random",
        "macro_cycle_file": getattr(env.macro, "current_cycle_file", ""),
        "macro_continuous_file": args.macro_continuous_file or "",
        "profile": getattr(args, "profile", None),
    }
    writer = TrajectoryWriter(jsonl_path, run_meta=run_meta)
    writer.write_header(env, initial_state, system_prompt=system_prompt)

    # Run loop
    total_reward = 0.0
    done = False
    months_completed = 0
    MAX_RETRIES = 5
    termination_reason = ""

    while not done and months_completed < args.months:
        print_step_header(months_completed + 1, obs.individual.get("age", 20))
        print_environment_state(obs)

        # ---- Single decision + retry on rejection -------------------------
        retries = 0
        month_decision = None
        while retries < MAX_RETRIES:
            decision = agent.decide(obs)
            print_agent_decision(decision)

            obs, reward, done, info = env.step(tool_calls=decision.tool_calls)

            if done:
                month_decision = decision
                break

            if info.get("action_rejected"):
                retries += 1
                rejection_msg = info.get("rejection_message", "unknown reason")
                if retries < MAX_RETRIES:
                    print(f"\n  {C.YELLOW}⚠️  Rejected — retrying ({retries}/{MAX_RETRIES}): {rejection_msg}{C.RESET}")
                else:
                    print(f"\n  {C.RED}⚠️  Max retries ({MAX_RETRIES}) exceeded — forcing NONE this month{C.RESET}")
                    # Force NONE to avoid wasting the month
                    obs, reward, done, info = env.step(action=Action(career_move=CareerMove.NONE))
                    decision = Decision(reasoning="(forced NONE after max retries)", tool_calls=[])
                    break
            else:
                month_decision = decision
                break  # success
        else:
            # Should not reach here, but guard
            month_decision = Decision(reasoning="(no decision)", tool_calls=[])

        total_reward += reward
        months_completed += 1
        writer.append_step(month_decision, env.history[-1])

        if done:
            termination_reason = info.get("termination_reason", "")
            print(f"\n  {C.RED}{C.BOLD}⚠ Episode ended: {termination_reason}{C.RESET}")

    # Build final summary
    s = env.micro.state
    net_worth = s.cash + s.savings + s.stock_value + s.pending_settlement - s.loan_balance
    price_level = env.macro.price_level or 1.0
    final_summary = {
        "months_played": months_completed,
        "total_reward": round(total_reward, 2),
        "final_cash": round(s.cash, 2),
        "final_occupation_id": s.occupation_id,
        "final_general_skill": s.general_skill,
        "final_occ_skill": s.occupation_skills.get(s.occupation_id, 0),
        "final_health": round(s.health, 3),
        "final_energy": round(s.energy, 3),
        "final_job_status": s.job_status.value,
        "final_net_worth": round(net_worth, 2),
        "final_real_net_worth": round(net_worth / price_level, 2),
        "termination_reason": termination_reason or "max_steps_reached",
        "age": s.age,
    }
    writer.write_summary(final_summary)
    json_path = writer.finalize_json()
    print(f"\n  {C.DIM}📁 Trajectory saved to: {json_path}{C.RESET}")

    # Summary
    print_final_summary(env, total_reward, months_completed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
