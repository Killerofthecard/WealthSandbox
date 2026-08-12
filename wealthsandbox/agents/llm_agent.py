"""LLM Agent for WealthSandBox using OpenAI-compatible API.

The agent receives natural-language observations from the environment,
reasons about the situation, and then issues tool calls that are parsed
into the environment's ``Action``.

Usage:
    from wealthsandbox.agents.llm_agent import LLMAgent
    agent = LLMAgent(model="gpt-4.1-mini")
    decision = agent.decide(observation)
    # decision.reasoning -> natural-language explanation
    # decision.tool_calls -> list of ToolCall objects
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from wealthsandbox.types import Observation
from wealthsandbox.agents.tools import ToolCall, Decision


# ---------------------------------------------------------------------------
# System prompt builder — all dollar amounts come from parameters, never
# hardcoded so costs stay in sync with EnvConfig automatically.
# ---------------------------------------------------------------------------
def _build_system_prompt(
    profile=None,  # AgentProfile — use keyword for cleaner call sites
    end_age: int = 60,
    living_expense: float = 2_000.0,
    upskill_cost: float = 5_000.0,
    upskill_months: int = 6,
    upskill_skill_boost: int = 1,
    max_general_skill: int = 10,
    max_occ_skill: int = 10,
    switch_base_cost: float = 2_000.0,
    tax_rate: float = 0.15,
    energy_threshold: float = 0.4,
    energy_cost_per_upskill: float = 0.4,
    energy_recovery_per_month: float = 0.10,
    intensive_work_months: int = 3,
    occ_skill_passive_months: int = 12,
    occupations: Optional[Dict[str, Any]] = None,
    persona: str = "",
) -> str:
    """Build the system prompt from actual configured costs and profile (no magic numbers).

    If *occupations* is not provided, falls back to DEFAULT_OCCUPATIONS.
    If *persona* is provided, it is injected as a ``# Your Persona`` section.
    """
    if profile is None:
        from wealthsandbox.profile import AgentProfile
        profile = AgentProfile()
    start_age = profile.age
    initial_cash = profile.initial_cash
    initial_health = profile.initial_health
    initial_energy = profile.initial_energy
    initial_gen = profile.initial_general_skill
    total_months = (end_age - start_age) * 12

    # ---- Build occupation table dynamically ----
    if occupations is None:
        from wealthsandbox.systems.career import DEFAULT_OCCUPATIONS
        occupations = {
            occ_id: {
                "industry": occ.industry,
                "base_monthly_salary": occ.base_monthly_salary,
                "min_general_skill": occ.min_general_skill,
                "min_health": occ.min_health,
                "entry_cost": occ.entry_cost,
                "training_months": occ.training_months,
                "tiers": [
                    {"name": t.name, "salary_multiplier": t.salary_multiplier}
                    for t in occ.tiers
                ],
            }
            for occ_id, occ in DEFAULT_OCCUPATIONS.items()
        }

    occ_rows: list[str] = []
    for occ_id in sorted(occupations.keys()):
        d = occupations[occ_id]
        industry = d.get("industry", "?")
        base = d["base_monthly_salary"]
        gen_req = d.get("min_general_skill", 1)
        health_req = d.get("min_health", 0.0)
        entry = d.get("entry_cost", 0)
        train = d.get("training_months", 0)
        tiers = d.get("tiers", [])
        tier_str = " → ".join(
            f"{t['name']}×{t['salary_multiplier']}" for t in tiers
        ) if tiers else "—"
        occ_rows.append(
            f"| {occ_id} | {industry} | ${base:,.0f} | {gen_req} | {health_req:.1f} "
            f"| ${entry:,.0f} | {train}mo | {tier_str} |"
        )

    occ_table: str = "\n".join(occ_rows) if occ_rows else "| — | — | — | — | — | — | — | — |"

    # ---- Persona section (only included when a profile is loaded) ----
    persona_block: str = ""
    if persona.strip():
        persona_block = f"""\
# Your Persona

{persona.strip()}

"""

    return f"""\
{persona_block}\
# Game Rules

You are an agent in a career & life simulation.  Your goal is to maximise
your **net worth** (cash + savings − loan) by age {end_age}.

You start at age {start_age} — you have {total_months} months.  Each turn is
**one month** (12 turns = 1 year).

## How the game ends
- Age {end_age} → your final net worth is scored.
- Cash + savings ≤ $0 → **bankruptcy — you lose immediately.**
- Health ≤ 0 → **death — you lose immediately.**

## Work is automatic
If you are employed, you earn your salary every month **without calling any
tool**.  Call tools only when you want to **change** something.

## One shot per month
Each turn you make exactly ONE response.  In that single response you can
call **multiple tools at once** — they execute in the order you list them.
After your tools finish, the month ends (living expenses, health decline,
etc.) and you won't get another chance until next month.

**Think through ALL the tools you need, then call them together.**

## Economy
Each turn you see an Economy Status: HEALTHY, SLUGGISH, WEAK, or RECESSION.
This affects your income, layoff risk, and loan interest rates.  In
recession, conserve cash and expect possible layoffs.

# Monthly Settlement — what happens automatically every month, in order

1. **Income** — if employed, salary is added to cash.
2. **Layoff check** — small chance of being laid off (higher in recession).
3. **Your actions** — you call ALL the tools you need this month in ONE
   response.  After they execute, the month ends.  Plan ahead.
4. **Living expenses** — ${living_expense:,.0f} deducted from cash (auto-withdrawn from
   savings if cash runs short).
5. **Bank interest** — savings earn interest; loan accrues interest, then
   2% of the loan balance is auto-repaid (min $50).
6. **Health decline** — health decreases (rate depends on age).
7. **Bankruptcy / death check** — game ends if cash + savings ≤ $0
   or health ≤ 0.

You can act between steps 3 and 4.  Plan your borrow / deposit / repay
decisions knowing that steps 4–7 always follow.

# Your State — what each number means

**Cash** (${initial_cash:,.0f} at start)
  Spending money.  Living expenses (${living_expense:,.0f}/mo) are deducted
  from cash automatically.  If cash runs out, savings cover the shortfall.

**Savings** ($0 at start)
  Bank deposit earning monthly interest (federal funds rate / 12).
  Not auto-spent — a safety net.

**Loan** ($0 at start)
  Money you owe the bank.  Interest = (federal funds rate + 2%) / 12 per
  month.  2% of the balance is auto-repaid each month.  Borrowing limit:
  12× monthly income if employed, otherwise $8,000.

**Health** ({initial_health:.1f} at start, range 0.0–1.0)
  Declines every month, faster with age.  Purely passive — you cannot
  restore it.  If health drops below an occupation's minimum threshold
  while employed, you are **forced to resign immediately**.

**Energy** ({initial_energy:.1f} at start, range 0.0–1.0)
  Spent on upskill ({energy_cost_per_upskill:.0%}) and intensive_work
  ({energy_cost_per_upskill:.0%}).  Drains {15}%/mo during occupation
  training.  Recovers {energy_recovery_per_month*100:.0f}%/mo when not
  training.  Need ≥{energy_threshold:.0%} to start upskill or intensive_work.

**General Skill** ({initial_gen} at start, range 1–{max_general_skill})
  Transferable capability.  Determines which occupations you can enter.
  Carries across job switches (partially for cross-industry).  Improved
  via `upskill`.

**Occupation Skill** (0 at start, range 1–{max_occ_skill} per job)
  Job-specific experience in your current occupation.  Determines your
  tier (Junior → Senior → …) and salary multiplier.  Grows passively
  every {occ_skill_passive_months} months, faster via `intensive_work`.
  **Resets when switching occupations.**

**Tenure** (0 at start)
  Months spent in your current occupation.  Together with occupation
  skill, gates tier promotions (automatic).

# Tools — what you can do each month

Call several tools in ONE response.  They run in the order you list them.
The month ends after they execute — there is no second chance.

| Tool | Effect | Requirements & Cost |
|---|---|---|
| `switch_occupation(id)` | Change career.  General skill carries over (same industry ~80%, unrelated ~20%).  Occupation skill resets. | gen_skill ≥ occupation minimum, health ≥ occupation minimum.  **manufacturing_worker is always free** (no cash, no training).  All other jobs: cash ≥ ${switch_base_cost:,} + entry_cost + ${living_expense:,.0f} buffer.  Training 0–6 months (keep working). |
| `upskill` | General skill +{upskill_skill_boost}.  Unlocks new occupations, boosts salary. | ${upskill_cost:,} cash + {energy_cost_per_upskill:.0%} energy.  {upskill_months} months.  Need ≥${upskill_cost + living_expense:,.0f} cash + ≥{energy_threshold:.0%} energy. |
| `intensive_work` | Occupation skill +1 in current job.  Accelerates tier promotion. | {energy_cost_per_upskill:.0%} energy (NO cash).  {intensive_work_months} months.  Must be employed, ≥{energy_threshold:.0%} energy. |
| `quit_job` | Resign immediately.  Income → $0.  Living expenses continue. | Must be employed. |
| `deposit(amount)` | Cash → savings.  Earns monthly interest. | Keep ≥${living_expense:,.0f} cash.  amount ≤ cash − ${living_expense:,.0f}. |
| `withdraw(amount)` | Savings → cash.  Instant. | amount ≤ savings balance. |
| `borrow(amount)` | Take a bank loan.  2% auto-repaid monthly. | Employed limit = 12× income.  Unemployed limit = $8,000. |
| `repay(amount)` | Repay loan early.  Reduces interest. | amount ≤ cash, amount ≤ loan balance. |

# Available Occupations

| ID | Industry | Base/mo | Gen≥ | Health≥ | Entry | Train | Tier ladder (×multiplier) |
|---|---|---|---|---|---|---|---|
{occ_table}

"""


class LLMAgent:
    """An LLM-based agent that interacts with WealthSandBox via OpenAI tool calling.

    The agent maintains a conversation history and calls an OpenAI-compatible
    chat completion API to generate reasoning + tool calls each turn.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        # ---- Tool-building parameters (flow from EnvConfig) ----
        living_expense: float = 2_000.0,
        upskill_cost: float = 5_000.0,
        upskill_months: int = 6,
        upskill_skill_boost: int = 1,
        max_skill_level: int = 10,
        switch_base_cost: float = 2_000.0,
        energy_threshold_for_upskill: float = 0.4,
        intensive_work_months: int = 3,
        occ_skill_passive_months: int = 12,
        occupations: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        persona: str = "",
    ):
        self.model = model or os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or _build_system_prompt(
            living_expense=living_expense,
            upskill_cost=upskill_cost,
            upskill_months=upskill_months,
            max_general_skill=max_skill_level,
            max_occ_skill=max_skill_level,
            switch_base_cost=switch_base_cost,
            energy_threshold=energy_threshold_for_upskill,
            intensive_work_months=intensive_work_months,
            occ_skill_passive_months=occ_skill_passive_months,
            occupations=occupations,
            persona=persona,
        )

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key."
            )

        # Build tools dynamically so tool descriptions match config costs.
        # Pre-built tools can be passed directly (e.g. from tests).
        if tools is not None:
            self.tools = tools
        else:
            from wealthsandbox.agents.tools import build_tools
            self.tools = build_tools(
                living_expense=living_expense,
                upskill_cost=upskill_cost,
                upskill_months=upskill_months,
                upskill_skill_boost=upskill_skill_boost,
                max_general_skill=max_skill_level,
                switch_base_cost=switch_base_cost,
                energy_threshold=energy_threshold_for_upskill,
                intensive_work_months=intensive_work_months,
                max_occ_skill=max_skill_level,
                occ_skill_passive_months=occ_skill_passive_months,
                occupations=occupations,
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.decision_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear conversation history (called at env reset)."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.decision_count = 0

    def decide(self, observation: Observation) -> Decision:
        """Generate a Decision (reasoning + tool calls) from the current Observation.

        Only tools that are currently legal (per the validator) are exposed to
        the model — it literally cannot call an unavailable action.  The
        observation text only lists blocked actions and why.
        """
        user_content = self._format_observation(observation)
        self.messages.append({"role": "user", "content": user_content})

        # Only expose actions that are currently legal.
        available = observation.macro.get("available_actions", {})
        legal_tools = self._filter_legal_tools(available)

        reasoning, tool_calls = self._call_llm(tools=legal_tools)

        # If the model called a tool but left content empty, synthesise a
        # placeholder so the conversation history isn't corrupted by an
        # empty assistant message (which confuses the model next turn).
        if not reasoning.strip() and tool_calls:
            tc = tool_calls[0]
            params = ", ".join(f"{k}={v}" for k, v in tc.parameters.items())
            reasoning = f"[Called {tc.tool_name}({params})]"
        elif not reasoning.strip():
            reasoning = "[No action — continuing with auto-work.]"

        # Only store the reasoning in conversation history.
        # Tool calls are NOT stored because the environment executes tools
        # itself (not the OpenAI API), and storing them without tool results
        # violates the API protocol.
        self.messages.append({"role": "assistant", "content": reasoning})

        self.decision_count += 1
        return Decision(reasoning=reasoning, tool_calls=tool_calls)

    def get_history(self) -> List[Dict[str, str]]:
        """Return the full conversation history (for debugging / logging)."""
        return list(self.messages)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(
        self, tools: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[str, List[ToolCall]]:
        """Single API call: model generates reasoning and optional tool call together.

        This avoids the inconsistency caused by two-stage calling where the
        model could say "no tool" in Stage 1 but call a tool in Stage 2.
        With a single call the reasoning and tool choice share the same
        generation, so they are inherently consistent.

        Args:
            tools: Optional filtered tool list.  When None, uses all tools.
                Callers pass a subset to limit what the agent can invoke this
                turn (only legal actions).
        """
        kwargs = dict(
            model=self.model,
            messages=self.messages,
            tools=tools if tools is not None else self.tools,
            tool_choice="auto",
            temperature=self.temperature,
        )
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        reasoning = message.content or ""

        # Parse all tool calls — the environment handles ordered bundles atomically.
        tool_calls: List[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    params = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    params = {}
                tool_calls.append(ToolCall(tool_name=tc.function.name, parameters=params))

        return reasoning, tool_calls

    def _filter_legal_tools(
        self, available_actions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return only the tool definitions for actions that are currently legal.

        When ``available_actions`` is empty (e.g. observation built without
        validator data), fall back to all tools.
        """
        if not available_actions:
            return self.tools
        legal_names = {
            name
            for name, info in available_actions.items()
            if isinstance(info, dict) and info.get("allowed")
        }
        return [t for t in self.tools if t["function"]["name"] in legal_names]

    def _format_observation(self, obs: Observation) -> str:
        """Convert an Observation into a natural-language prompt for the LLM."""
        ind = obs.individual
        macro = obs.macro

        # Detect retry: narrative starts with rejection warning
        is_retry = obs.narrative.startswith("⚠️ LAST ACTION REJECTED:")

        header = f"=== {obs.year}-{obs.month:02d} (month {obs.info.get('total_months', '?')})"
        if is_retry:
            header += " — ⚠️ RETRY: your last action was rejected, try something else ==="

        lines = [
            header,
            "",
            f"# Your Status",
            f"Age: {ind.get('age', '?')}",
            f"Health: {ind.get('health', 0):.3f}",
            f"Energy: {ind.get('energy', 1.0):.1%}",
            f"Occupation: {ind.get('occupation_id', 'none') or 'none'}",
            f"Job Status: {ind.get('job_status', '?')}",
            f"General Skill: {ind.get('general_skill', 1)}",
            f"Occ Skill: {ind.get('occ_skill', 0)}",
            f"Tenure: {ind.get('tenure_months', 0)} months",
            f"Monthly After-Tax Income: ${ind.get('monthly_after_tax_income', 0):,.2f}",
            f"Cash: ${ind.get('cash', 0):,.2f}",
            f"Savings: ${ind.get('savings', 0):,.2f}",
            f"Loan: ${ind.get('loan_balance', 0):,.2f}",
        ]

        upskill = ind.get("upskill_months_remaining", 0)
        if upskill > 0:
            lines.append(f"Upskill (general) in progress: {upskill} months remaining")

        intensive = ind.get("intensive_work_months_remaining", 0)
        if intensive > 0:
            lines.append(f"Intensive work (occ_skill) in progress: {intensive} months remaining")

        training = ind.get("training_months_remaining", 0)
        if training > 0:
            target = ind.get("training_target_occupation", "?")
            lines.append(f"Training for {target}: {training} months remaining")

        # Separate actions into available and unavailable groups.
        # Both are shown in text so weaker models (e.g. Flash) have a
        # textual anchor for what they CAN do — not just what they can't.
        all_actions = macro.get("available_actions", {})
        available_names: list[str] = []
        unavailable: list[tuple[str, str]] = []
        for name, info in all_actions.items():
            if isinstance(info, dict) and info.get("allowed"):
                available_names.append(name)
            elif isinstance(info, dict):
                unavailable.append((name, info.get("reason", "")))

        if available_names:
            lines.extend(["", "## Available Actions (you CAN call these)"])
            for name in available_names:
                lines.append(f"  {name}")

        if unavailable:
            lines.extend(["", "## Unavailable"])
            for action_name, reason in unavailable:
                suffix = f" — {reason}" if reason else ""
                lines.append(f"  {action_name}{suffix}")

        # Economy status
        econ = macro.get("economy_status", "")
        if econ:
            lines.extend([
                "",
                f"# Economy Status",
                f"  {econ}",
            ])

        # Available occupations (only provided when unemployed)
        avail = macro.get("available_occupations", {})
        if avail:
            switch_base = macro.get("switch_base_cost", 2_000)
            lines.append("")
            lines.append("# Available Occupations (you can switch to any of these)")
            for occ_id, detail in avail.items():
                req = detail.get("min_general_skill", 1)
                min_h = detail.get("min_health", 0.0)
                entry = detail.get("entry_cost", 0)
                train = detail.get("training_months", 0)
                total = switch_base + entry
                tiers = detail.get("tiers", [])
                tier_str = " → ".join(
                    f"{t['name']}(×{t['salary_multiplier']:.1f})" for t in tiers
                ) if tiers else "—"
                lines.append(
                    f"  {occ_id}: ${detail['base_monthly_salary']:,.0f}/mo "
                    f"({detail['industry']}, "
                    f"gen≥{req}, health≥{min_h:.1f}, ${total:,.0f}+{train}mo)"
                )
                if tier_str:
                    lines.append(f"    Ladder: {tier_str}")

        lines.extend([
            "",
            f"# Narrative",
            f"{obs.narrative}",
            "",
            f"# Recent Events",
        ])
        events = ind.get("last_month_events", [])
        if events:
            for e in events:
                lines.append(f"- {e}")
        else:
            lines.append("- No notable events this month.")

        if is_retry:
            lines.extend([
                "",
                "Your last action was rejected. Check the reason above and adjust.",
            ])
        elif ind.get("job_status") == "unemployed":
            lines.extend([
                "",
                "You are UNEMPLOYED — no income.",
            ])
        # employed: no extra message needed — available tools speak for themselves.
        return "\n".join(lines)



