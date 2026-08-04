"""Tests for the main environment (simplified — career only)."""

import unittest
from wealthsandbox.env import WealthSandBoxEnv
from wealthsandbox.config import EnvConfig
from wealthsandbox.profile import AgentProfile
from wealthsandbox.types import Action, CareerMove


class TestWealthSandBoxEnv(unittest.TestCase):

    def test_reset(self):
        env = WealthSandBoxEnv(EnvConfig(seed=123))
        obs = env.reset(seed=123)
        self.assertEqual(obs.individual["age"], 20)
        self.assertEqual(obs.individual["occupation_id"], "")
        self.assertFalse(obs.done)

    def test_step_runs(self):
        env = WealthSandBoxEnv(EnvConfig(seed=123))
        env.reset(seed=123)
        # First, get an occupation so auto-income kicks in
        action = Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        )
        obs, reward, done, info = env.step(action=action)
        self.assertIsInstance(reward, float)
        self.assertFalse(done)
        # Auto-work with NONE — income is automatic
        action2 = Action(career_move=CareerMove.NONE)
        obs2, reward2, done2, info2 = env.step(action=action2)
        self.assertGreater(reward2, 0)  # should earn income automatically
        self.assertFalse(done2)

    def test_age_ticks_every_12_months(self):
        env = WealthSandBoxEnv(EnvConfig(profile=AgentProfile(age=20), seed=1))
        env.reset(seed=1)
        # Get an occupation first
        action = Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        )
        env.step(action=action)
        # Age should still be 20 after 1 month
        self.assertEqual(env.micro.state.age, 20)
        # Run 11 more months (total 12) with auto-work
        for _ in range(11):
            env.step(action=Action(career_move=CareerMove.NONE))
        self.assertEqual(env.micro.state.age, 21)

    def test_termination_age(self):
        env = WealthSandBoxEnv(EnvConfig(
            profile=AgentProfile(age=59), end_age=60, seed=1,
        ))
        env.reset(seed=1)
        # Get occupation
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        # Run 11 more (total = 12 months = 1 year) with auto-work
        for _ in range(11):
            _, _, done, _ = env.step(action=Action(career_move=CareerMove.NONE))
        # After 12 months from age 59, should reach 60 and terminate
        _, _, done, info = env.step(action=Action(career_move=CareerMove.NONE))
        # 60 >= 60 means terminated
        self.assertTrue(done)
        self.assertIn(info["termination_reason"], ["age_limit", "bankruptcy", "death"])

    def test_bankruptcy(self):
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        # Agent with no cash and no job can't pay living expenses → bankruptcy
        env.micro.state.cash = 500.0
        _, _, done, info = env.step(action=Action(career_move=CareerMove.NONE))
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "bankruptcy")
        # Cash must be floored at 0 (never negative)
        self.assertEqual(env.micro.state.cash, 0.0)

    def test_apply_tool_calls_quit(self):
        from wealthsandbox.agents.tools import ToolCall
        tcs = [ToolCall(tool_name="quit_job", parameters={})]
        action = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(action.career_move, CareerMove.QUIT_JOB)

    def test_apply_tool_calls_switch(self):
        from wealthsandbox.agents.tools import ToolCall
        tcs = [ToolCall(tool_name="switch_occupation",
                         parameters={"occupation_id": "nurse"})]
        action = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(action.career_move, CareerMove.SWITCH_OCCUPATION)
        self.assertEqual(action.target_occupation_id, "nurse")

    def test_apply_tool_calls_upskill(self):
        from wealthsandbox.agents.tools import ToolCall
        tcs = [ToolCall(tool_name="upskill", parameters={})]
        action = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(action.career_move, CareerMove.UPSKILL)

    # ------------------------------------------------------------------
    # In-month retry (action rejected → retry without consuming the month)
    # ------------------------------------------------------------------

    def test_rejected_action_does_not_advance_month(self):
        """When an action is rejected, the macro calendar should NOT advance."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        months_before = env.macro.total_months

        # Unemployed agent tries to quit — should be rejected
        action = Action(career_move=CareerMove.QUIT_JOB)
        obs, reward, done, info = env.step(action=action)

        self.assertTrue(info.get("action_rejected"))
        self.assertEqual(env.macro.total_months, months_before)  # month NOT advanced

    def test_rejected_action_returns_zero_reward(self):
        """Rejected actions should not yield reward."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)

        action = Action(career_move=CareerMove.QUIT_JOB)
        obs, reward, done, info = env.step(action=action)

        self.assertTrue(info.get("action_rejected"))
        self.assertEqual(reward, 0.0)

    def test_retry_after_rejection_succeeds_in_same_month(self):
        """Agent can retry with a valid action after rejection, within the same month."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        months_before = env.macro.total_months

        # First attempt: try to quit while unemployed → rejected
        action_bad = Action(career_move=CareerMove.QUIT_JOB)
        obs, reward, done, info = env.step(action=action_bad)
        self.assertTrue(info.get("action_rejected"))
        self.assertEqual(env.macro.total_months, months_before)  # still same month

        # Second attempt: valid switch → should succeed and advance month
        action_good = Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        )
        obs2, reward2, done2, info2 = env.step(action=action_good)
        self.assertFalse(info2.get("action_rejected"))
        self.assertEqual(env.macro.total_months, months_before + 1)  # month advanced

    def test_retry_then_none_advances_month(self):
        """After rejection, choosing NONE concludes the month normally."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        months_before = env.macro.total_months

        # Rejected attempt
        env.step(action=Action(career_move=CareerMove.QUIT_JOB))

        # Give up → NONE should conclude the month
        obs, reward, done, info = env.step(action=Action(career_move=CareerMove.NONE))
        self.assertFalse(info.get("action_rejected"))
        self.assertEqual(env.macro.total_months, months_before + 1)

    def test_tick_runs_only_once_across_retries(self):
        """Income should be added only once, even if agent retries multiple times.

        Scenario:
        1. Month 1: get a job (manufacturing_worker) — costs $2,000 to switch,
           $0 income (was unemployed during tick), -$2,000 living.
           → cash = $5,000 - $2,000 - $2,000 = $1,000
        2. Month 2: try to upskill → rejected (need $5,000, only have ~$4,036
           after tick adds income). Then retry NONE.
           → After month: cash = $1,000 + $3,036 (one income) - $2,000 = $2,036

        If tick ran twice, cash would be ~$5,072 instead.
        """
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)

        # Month 1: get a job
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        self.assertEqual(env.micro.state.cash, 1000.0)  # 5000 - 2000 - 2000

        # Month 2: try to upskill (will be rejected — only $1,000 + upcoming income)
        obs, reward, done, info = env.step(action=Action(career_move=CareerMove.UPSKILL))
        self.assertTrue(info.get("action_rejected"))
        self.assertIn("Insufficient cash", info.get("rejection_message", ""))

        # Retry with NONE — conclude the month
        obs, reward, done, info = env.step(action=Action(career_move=CareerMove.NONE))
        self.assertFalse(info.get("action_rejected"))

        # Cash should reflect ONE month's income, not two.
        # The exact amount depends on the macro industry multiplier, but the
        # invariant is: cash = 1000 + one_income - 2000 (living).
        expected = 1000.0 + env.micro.state.monthly_after_tax_income - 2000.0
        self.assertAlmostEqual(env.micro.state.cash, expected, places=2)

    def test_rejection_events_preserved_on_retry(self):
        """Events from a rejected attempt should be visible in the retry observation."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)

        obs, reward, done, info = env.step(action=Action(career_move=CareerMove.QUIT_JOB))
        self.assertTrue(info.get("action_rejected"))
        # Observation should contain rejection info
        self.assertIn("REJECTED", obs.narrative)
        events = obs.individual.get("last_month_events", [])
        self.assertTrue(any("not currently employed" in e.lower() for e in events))

    def test_rejection_message_in_info(self):
        """Info dict should carry a human-readable rejection message."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)

        obs, reward, done, info = env.step(action=Action(career_move=CareerMove.QUIT_JOB))
        self.assertTrue(info.get("action_rejected"))
        self.assertIn("not currently employed", info.get("rejection_message", "").lower())

    def test_history_only_contains_completed_months(self):
        """Rejected months should NOT appear in env.history."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        self.assertEqual(len(env.history), 0)

        # Rejected
        env.step(action=Action(career_move=CareerMove.QUIT_JOB))
        self.assertEqual(len(env.history), 0)  # still 0

        # Successful
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        self.assertEqual(len(env.history), 1)  # now 1


if __name__ == "__main__":
    unittest.main()
