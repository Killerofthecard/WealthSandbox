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
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.QUIT_JOB)
        self.assertEqual(issues, [])

    def test_apply_tool_calls_switch(self):
        from wealthsandbox.agents.tools import ToolCall
        tcs = [ToolCall(tool_name="switch_occupation",
                         parameters={"occupation_id": "nurse"})]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.SWITCH_OCCUPATION)
        self.assertEqual(actions[0].target_occupation_id, "nurse")
        self.assertEqual(issues, [])

    def test_apply_tool_calls_upskill(self):
        from wealthsandbox.agents.tools import ToolCall
        tcs = [ToolCall(tool_name="upskill", parameters={})]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.UPSKILL)
        self.assertEqual(issues, [])

    # --- apply_tool_calls no longer deduplicates, unknown tools surface ---

    def test_apply_tool_calls_keeps_duplicates(self):
        """Two deposits must become two actions — no silent dedup."""
        from wealthsandbox.agents.tools import ToolCall
        tcs = [
            ToolCall(tool_name="deposit", parameters={"amount": 1000}),
            ToolCall(tool_name="deposit", parameters={"amount": 2000}),
        ]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].amount, 1000)
        self.assertEqual(actions[1].amount, 2000)
        self.assertEqual(issues, [])

    def test_apply_tool_calls_unknown_tool_reports_issue(self):
        from wealthsandbox.agents.tools import ToolCall
        tcs = [ToolCall(tool_name="fly_to_mars", parameters={})]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        # The unknown tool yields a NONE placeholder action plus an issue.
        self.assertEqual(actions[0].career_move, CareerMove.NONE)
        self.assertEqual(len(issues), 1)
        self.assertIn("fly_to_mars", issues[0])

    def test_step_rejects_unknown_tool(self):
        from wealthsandbox.agents.tools import ToolCall
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        tcs = [ToolCall(tool_name="fly_to_mars", parameters={})]
        obs, reward, done, info = env.step(tool_calls=tcs)
        self.assertTrue(info.get("action_rejected"))
        self.assertIn("fly_to_mars", info.get("rejection_message", ""))
        self.assertEqual(env.macro.total_months, 0)  # month NOT advanced

    # --- Batch conflict guards (order semantics) ---

    def test_batch_quit_plus_switch_rejected(self):
        from wealthsandbox.agents.tools import ToolCall
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        tcs = [
            ToolCall(tool_name="switch_occupation",
                     parameters={"occupation_id": "nurse"}),
            ToolCall(tool_name="quit_job", parameters={}),
        ]
        obs, reward, done, info = env.step(tool_calls=tcs)
        self.assertTrue(info.get("action_rejected"))
        self.assertIn("quit", info.get("rejection_message", "").lower())
        self.assertEqual(env.macro.total_months, 0)

    def test_batch_two_switches_rejected(self):
        from wealthsandbox.agents.tools import ToolCall
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        tcs = [
            ToolCall(tool_name="switch_occupation",
                     parameters={"occupation_id": "nurse"}),
            ToolCall(tool_name="switch_occupation",
                     parameters={"occupation_id": "manufacturing_worker"}),
        ]
        obs, reward, done, info = env.step(tool_calls=tcs)
        self.assertTrue(info.get("action_rejected"))
        self.assertIn("one occupation", info.get("rejection_message", "").lower())
        self.assertEqual(env.macro.total_months, 0)

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
        """Income should be added only once per month."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)

        # Month 1: get a job
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        cash_after_month1 = env.micro.state.cash

        # Month 2: auto-work
        _, _, _, _ = env.step(action=Action(career_move=CareerMove.NONE))
        income = env.micro.state.monthly_after_tax_income
        expected = cash_after_month1 + income - 2000.0
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

    # --- Bank amount validation triggers rejection (Bug 2 fix) ---

    def test_deposit_amount_zero_rejected(self):
        """deposit(amount=0) should trigger an action rejection."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        # First get a job so we have income and cash
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        obs, reward, done, info = env.step(action=Action(
            career_move=CareerMove.DEPOSIT, amount=0,
        ))
        self.assertTrue(info.get("action_rejected"))
        self.assertIn("greater than zero", info.get("rejection_message", "").lower())

    def test_withdraw_amount_zero_rejected(self):
        """withdraw(amount=0) should trigger an action rejection."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        obs, reward, done, info = env.step(action=Action(
            career_move=CareerMove.WITHDRAW, amount=0,
        ))
        self.assertTrue(info.get("action_rejected"))

    def test_borrow_amount_zero_rejected(self):
        """borrow(amount=0) should trigger an action rejection."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        obs, reward, done, info = env.step(action=Action(
            career_move=CareerMove.BORROW, amount=0,
        ))
        self.assertTrue(info.get("action_rejected"))

    def test_repay_amount_zero_rejected(self):
        """repay(amount=0) should trigger an action rejection."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        obs, reward, done, info = env.step(action=Action(
            career_move=CareerMove.REPAY, amount=0,
        ))
        self.assertTrue(info.get("action_rejected"))

    # --- Tick runs after execution, before observation (Bug 1 + timing fix) ---

    def test_tick_runs_after_execution(self):
        """Observation should include tick effects (income)."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        obs, _, _, _ = env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        # After step: income from tick should be added before observation
        # Initial cash $10,000 - $0 switch - $2,000 expenses + ~$3,230 income
        self.assertGreater(obs.individual["cash"], 10_000)
        self.assertGreater(obs.individual["monthly_after_tax_income"], 0)

    def test_retry_does_not_duplicate_tick(self):
        """Retry after rejection should NOT cause tick to run twice for the same month."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        cash_before = env.micro.state.cash
        health_before = env.micro.state.health

        # Rejected: deposit(amount=0) — agent is unemployed, deposit denied by guard
        env.step(action=Action(career_move=CareerMove.DEPOSIT, amount=0))
        # In new flow, rejection returns before tick AND finalize — nothing changes.
        self.assertEqual(env.micro.state.cash, cash_before)
        self.assertEqual(env.micro.state.health, health_before)
        self.assertEqual(len(env.history), 0)  # not archived

        # Valid step: switch to manufacturing_worker
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        # Tick ran exactly once: income added, expenses deducted
        self.assertGreater(env.micro.state.cash, cash_before - 2_000)  # income > expenses
        self.assertLess(env.micro.state.health, health_before)  # health declined once
        self.assertEqual(len(env.history), 1)  # archived once

    # --- rest / medical_care flow ---

    def test_rest_recovers_health_and_penalizes_income(self):
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        # Baseline normal income for one auto-work month
        env.step(action=Action(career_move=CareerMove.NONE))
        normal_income = env.micro.state.monthly_after_tax_income

        # Lower health so rest is legal (still above the job's 0.6 min_health)
        env.micro.state.health = 0.7
        obs, reward, done, info = env.step(action=Action(career_move=CareerMove.REST))
        self.assertFalse(info.get("action_rejected"))
        # health recovered
        self.assertGreater(env.micro.state.health, 0.7)
        # rest only releases non-work occupancy; the job's own 0.50 footprint
        # remains, so available energy = 1.0 − 0.50 = 0.50
        self.assertAlmostEqual(env.micro.state.energy, 0.50, places=2)
        # rest month earns 20% less
        self.assertLess(env.micro.state.monthly_after_tax_income, normal_income)
        # flag cleared after the month
        self.assertFalse(env.micro.state.resting_this_month)

    def test_medical_care_pays_and_recovers_health(self):
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        env.micro.state.health = 0.7
        cash_before = env.micro.state.cash
        obs, reward, done, info = env.step(action=Action(career_move=CareerMove.MEDICAL_CARE))
        self.assertFalse(info.get("action_rejected"))
        self.assertGreater(env.micro.state.health, 0.7)      # recovered +0.15
        self.assertLess(env.micro.state.cash, cash_before)    # paid $3,000
        self.assertEqual(env.micro.state.medical_care_uses_this_year, 1)

    def test_medical_care_third_use_rejected(self):
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        env.micro.state.medical_care_uses_this_year = 2
        env.micro.state.cash = 50_000
        obs, reward, done, info = env.step(action=Action(career_move=CareerMove.MEDICAL_CARE))
        self.assertTrue(info.get("action_rejected"))
        self.assertIn("times", info.get("rejection_message", "").lower())

    def test_medical_care_uses_reset_each_year(self):
        from wealthsandbox.systems.aging import AgingSystem
        from wealthsandbox.types import AgentState
        aging = AgingSystem(end_age=60)
        state = AgentState(medical_care_uses_this_year=2)
        aging.finalize(state, {"total_months": 12})
        self.assertEqual(state.age, 21)
        self.assertEqual(state.medical_care_uses_this_year, 0)

    # --- Reward decomposition (per-step flow ledger) ---

    def test_step_records_component_flows(self):
        """Each completed month records per-component flows and a summed reward."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        obs, reward, done, info = env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        self.assertFalse(info.get("action_rejected"))
        step = env.history[-1]

        flow = step["flow"]
        self.assertGreater(flow["employment_income"], 0)
        self.assertLess(flow["living_expense"], 0)
        # Reward is the sum of the month's component flows.
        self.assertAlmostEqual(reward, sum(flow.values()), places=2)
        self.assertAlmostEqual(step["reward"], reward, places=2)

    def test_cumulative_flow_accumulates_across_months(self):
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        env.step(action=Action(career_move=CareerMove.NONE))

        cum = env.history[-1]["cumulative_flow"]
        # Two months of employment income and living expense.
        self.assertGreater(cum["employment_income"], 0)
        self.assertLess(cum["living_expense"], 0)
        # Cumulative income must exceed a single month's income.
        latest_income = env.history[-1]["flow"].get("employment_income", 0)
        self.assertGreater(cum["employment_income"], latest_income)

    def test_medical_cost_recorded_as_negative_flow(self):
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        env.micro.state.health = 0.7
        obs, reward, done, info = env.step(action=Action(career_move=CareerMove.MEDICAL_CARE))
        self.assertFalse(info.get("action_rejected"))
        flow = env.history[-1]["flow"]
        self.assertLess(flow["medical_cost"], 0)
        self.assertAlmostEqual(flow["medical_cost"], -env.config.medical_care_cost, places=2)


if __name__ == "__main__":
    unittest.main()
