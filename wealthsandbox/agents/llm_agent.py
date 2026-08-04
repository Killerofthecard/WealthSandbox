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
    intensive_work_months: int = 3,
    occ_skill_passive_months: int = 12,
) -> str:
    """Build the system prompt from actual configured costs and profile (no magic numbers)."""
    if profile is None:
        from wealthsandbox.profile import AgentProfile
        profile = AgentProfile()
    start_age = profile.age
    initial_cash = profile.initial_cash
    total_months = (end_age - start_age) * 12
    return f"""You are a player in a career simulation called WealthSandBox.

# YOUR OBJECTIVE

Maximise your **total net worth** by age {end_age}.  You start at age
{start_age} — you have **{total_months} months ({total_months // 12} years)**.

## ⚠️ RULE #1: DO NOT GO BANKRUPT

Cash ≤ $0 = game over.  Before spending, always check:
"After this cost + ${living_expense:,.0f} living, is cash still > $0?"

# Game Rules
- You start at age {start_age}, ${initial_cash:,.0f} cash, unemployed,
  general_skill 1, full health (1.0), full energy (1.0).
- Each turn = ONE MONTH.  12 turns = 1 year.
- Episode ends: age {end_age} (success), cash ≤ $0 (bankruptcy), or health ≤ 0 (death).
- **Work is automatic.** You earn your salary every month without calling any tool.
  Call tools only when you want to **change** something.

# Economy Status (shown each turn)
Each turn you will see a qualitative description of the economy — HEALTHY,
SLUGGISH, WEAK, or RECESSION.  This affects your income, layoff risk, and
how easy it is to find a new job.  Use it to guide your decisions:
- HEALTHY: safe to invest in upskilling, switch jobs.
- SLUGGISH: proceed with caution, keep some cash buffer.
- WEAK / RECESSION: conserve cash, avoid unnecessary spending,
  expect income reductions and possible layoffs.

# Your State (shown each turn)
- **Cash** — liquid money.  Reaching $0 = game over.
- **Health** (0.0–1.0) — declines every month, faster as you age:
  * 20–29: −0.03%/mo | 30–39: −0.10%/mo | 40–49: −0.30%/mo | 50+: −0.60%/mo
  * **CRITICAL**: Each occupation has a `min_health` threshold.  If your health
    drops below it while employed, you will be **FORCED TO RESIGN** at the
    start of that month.  You must plan your career path to transition into
    less demanding work BEFORE your health declines too far.
- **Energy** (0.0–1.0) — upskill costs {energy_cost_per_upskill:.0%}, intensive_work
  costs {energy_cost_per_upskill:.0%}.  Training drains 15%/mo.  Recovers 2%/mo
  when resting.  Need ≥{energy_threshold:.0%} to start upskill or intensive_work.
- **General Skill** (1–{max_general_skill}) — transferable capability.  Gates
  occupation entry.  Improved via `upskill` (${upskill_cost:,}, {upskill_months} mo).
- **Occupation Skill** (1–{max_occ_skill} per job) — per-job experience.  Gates
  tier promotion (Junior → Senior).  Grows passively every {occ_skill_passive_months}
  months, or faster via `intensive_work` ({intensive_work_months} mo, no cash cost).
  **NOT carried across occupation switches** — resets to the starting tier.

# Income
- Formula: `base × (1 + sensitivity × (general_skill − 3)) × tier_multiplier`
- Tax: flat {tax_rate:.0%}.  After-tax = gross × {1 - tax_rate:.2f}.
- Tier multiplier increases with occupation skill and tenure.

# Tools (call ONE per month when you need to change)
| Tool | Cost | Time | What it does |
|---|---|---|---|
| `switch_occupation(id)` | ${switch_base_cost:,} + entry | 0–{6} mo training | Switch jobs. General skill carries partially. Occ skill RESETS. |
| `upskill` | ${upskill_cost:,} + {energy_cost_per_upskill:.0%} energy | {upskill_months} mo | General skill +{upskill_skill_boost}. |
| `intensive_work` | {energy_cost_per_upskill:.0%} energy (NO cash) | {intensive_work_months} mo | Occ skill +1 in current job. |
| `quit_job` | None | Immediate | Become unemployed. Income → $0. |

# Skill Transfer on Switch
  * Same industry → ~80% general_skill retained
  * Related → ~40% | Unrelated → ~20% | First job → 100%
  * Industries: tech, finance, healthcare, manufacturing, gov.

# Constraints — Action Rejected If:
1. Cash too low (need action cost + ${living_expense:,.0f} living buffer)
2. General skill below occupation's min requirement
3. Health below occupation's min threshold (both for entry AND continued employment)
4. Energy < {energy_threshold:.0%} (for upskill or intensive_work)
5. Already training / upskilling / doing intensive work
6. General skill or occ skill already at max ({max_general_skill}/{max_occ_skill})
7. Not employed (for quit_job or intensive_work)
→ Rejected actions show: "⚠️ LAST ACTION REJECTED: <reason>"

# Available Occupations
| ID | Industry | Base/mo | Gen≥ | Health≥ | Entry | Train | Tiers (×multiplier) |
|---|---|---|---|---|---|---|---|
| software_engineer | tech | $6,500 | 4 | 0.3 | $8,000 | 4mo | Junior×1.0 → Mid×1.5 → Senior×2.2 → Principal×3.5 |
| data_scientist | tech | $6,800 | 4 | 0.3 | $8,000 | 4mo | Junior×1.0 → Mid×1.5 → Senior×2.2 → Principal×3.5 |
| investment_banker | finance | $7,500 | 5 | 0.4 | $12,000 | 6mo | Analyst×1.0 → Assoc×1.6 → VP×2.5 → MD×4.0 |
| financial_analyst | finance | $5,500 | 3 | 0.3 | $5,000 | 3mo | Junior×1.0 → Mid×1.4 → Senior×1.9 → Lead×2.5 |
| manufacturing_worker | manufacturing | $3,800 | 1 | 0.6 | $0 | 0mo | Apprentice×1.0 → Skilled×1.3 → Supervisor×1.7 → Manager×2.2 |
| nurse | healthcare | $5,200 | 2 | 0.5 | $4,000 | 3mo | Staff×1.0 → Senior×1.5 → Head×2.0 |
| civil_servant | gov | $4,500 | 2 | 0.3 | $1,000 | 2mo | Jr Officer×1.0 → Sr Officer×1.4 → Director×2.0 |

# What To Do
- **Unemployed**: you MUST call switch_occupation to get a job.  Every month
  unemployed costs you ${living_expense:,.0f} with zero income — you will go
  bankrupt.  Pick the best occupation you qualify for and can afford.
- **Employed**: most months call NO tool — you earn automatically.  Call a
  tool only when you want to upskill, switch jobs, or do intensive work.
- Always check Legal Actions ✓/✗ before deciding.  If rejected, try something
  different.
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
    ):
        self.model = model or os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or _build_system_prompt()

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
        """Generate a Decision (reasoning + tool calls) from the current Observation."""
        user_content = self._format_observation(observation)
        self.messages.append({"role": "user", "content": user_content})

        reasoning, tool_calls = self._call_llm()

        # If the model called a tool but left content empty, synthesise a
        # placeholder so the conversation history isn't corrupted by an
        # empty assistant message (which confuses the model next turn).
        if not reasoning.strip() and tool_calls:
            tc = tool_calls[0]
            params = ", ".join(f"{k}={v}" for k, v in tc.parameters.items())
            reasoning = f"[Called {tc.tool_name}({params})]"
        elif not reasoning.strip():
            reasoning = "[No action — continuing with auto-work.]"

        # --- Last-resort guard: detect reasoning / tool-call contradictions ---
        # Some smaller models (e.g. gpt-4.1-mini) occasionally write a perfectly
        # sensible reasoning ("I will not call a tool") but then fire a tool
        # anyway.  When we detect this we discard the tool call and let the
        # agent auto-work — the reasoning was correct, the tool call was noise.
        if tool_calls and _reasoning_says_no_tool(reasoning):
            discarded = tool_calls[0].tool_name
            reasoning += (
                f" [Note: {discarded} was discarded because the reasoning "
                f"said no tool should be called.]"
            )
            tool_calls = []

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

    def _call_llm(self) -> tuple[str, List[ToolCall]]:
        """Single API call: model generates reasoning and optional tool call together.

        This avoids the inconsistency caused by two-stage calling where the
        model could say "no tool" in Stage 1 but call a tool in Stage 2.
        With a single call the reasoning and tool choice share the same
        generation, so they are inherently consistent.
        """
        kwargs = dict(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
            temperature=self.temperature,
        )
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        reasoning = message.content or ""

        # Parse at most one tool call (the environment handles one action/month).
        tool_calls: List[ToolCall] = []
        if message.tool_calls:
            tc = message.tool_calls[0]
            try:
                params = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                params = {}
            tool_calls.append(ToolCall(tool_name=tc.function.name, parameters=params))

        return reasoning, tool_calls

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

        lines.extend([
            "",
            f"# Legal Actions This Month",
        ])

        avail = macro.get("available_actions", {})
        if avail:
            for action_name, info in avail.items():
                allowed = info.get("allowed", info) if isinstance(info, dict) else info
                reason = info.get("reason", "") if isinstance(info, dict) else ""
                if allowed:
                    lines.append(f"  ✓ {action_name}")
                else:
                    suffix = f" — {reason}" if reason else ""
                    lines.append(f"  ✗ {action_name}{suffix}")
        else:
            lines.append("  (no actions available)")

        # Economy status
        econ = macro.get("economy_status", "")
        if econ:
            lines.extend([
                "",
                f"# Economy Status",
                f"  {econ}",
            ])

        lines.extend([
            "",
            f"# Macro Economy",
            f"Industry average monthly salaries:",
        ])
        for industry, salary in macro.get("industry_incomes", {}).items():
            lines.append(f"  {industry}: ${salary:,.0f}")

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
                    f"({detail['industry']}, sens={detail['skill_sensitivity']}, "
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
                "Your last action was REJECTED. Try a DIFFERENT action, or call no tool.",
            ])
        elif ind.get("job_status") == "unemployed":
            lines.extend([
                "",
                "You are UNEMPLOYED — you have no income and your cash is burning every "
                "month.  You MUST call switch_occupation to get a job.  Only occupations "
                "with ✓ in Legal Actions are available to you right now.",
            ])
        else:
            lines.extend([
                "",
                "You are employed and earning automatically.  Call a tool only if you "
                "want to switch jobs, upskill, or do intensive work.",
            ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Contradiction detection (last-resort safety net for small models)
# ---------------------------------------------------------------------------

# Phrases that indicate the agent intends NOT to call a tool this turn.
_NO_TOOL_PHRASES: tuple = (
    "call no tool",
    "not call a tool",
    "not call any tool",
    "will not call",
    "won't call",
    "do nothing",
    "no action",
    "no tool call",
    "without calling",
    "continue working",
    "continue auto-working",
    "not use any tool",
    "will remain",
    "i will stay",
    "save cash",
    "just work",
    "remain unemployed",
)

# Phrases that OVERRIDE the no-tool detection — the model explicitly wants to act.
_TOOL_ACTION_PHRASES: tuple = (
    "i call ",
    "will call ",
    "calling the ",
    "i will switch",
    "i will upskill",
    "i will quit",
    "let me switch",
    "let me upskill",
    "let me quit",
    "going to switch",
    "going to upskill",
    "going to quit",
    "i am switching",
    "i am upskilling",
    "i am quitting",
    "i choose",
    "i want to switch",
    "i want to upskill",
    "i want to quit",
)


def _reasoning_says_no_tool(reasoning: str) -> bool:
    """Return True if the reasoning text clearly says NO tool should be called.

    This is a heuristic safety net for models that self-contradict — they
    write sensible reasoning that says "do nothing" but then fire a tool
    anyway.  The presence of an explicit action phrase overrides the
    no-tool signal.
    """
    text = reasoning.lower()

    # Must match at least one no-tool phrase.
    has_no_tool = any(phrase in text for phrase in _NO_TOOL_PHRASES)
    if not has_no_tool:
        return False

    # Override: if the reasoning also explicitly says to call a tool,
    # the no-tool match was incidental.
    has_action = any(phrase in text for phrase in _TOOL_ACTION_PHRASES)
    if has_action:
        return False

    return True
