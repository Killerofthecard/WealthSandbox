"""Tests for tool-call parsing and the LLMAgent Decision data model."""

import unittest

from wealthsandbox.env import WealthSandBoxEnv
from wealthsandbox.agents.tools import ToolCall, Decision, TOOLS
from wealthsandbox.types import Action, CareerMove


class TestApplyToolCalls(unittest.TestCase):

    def test_quit_job_tool(self):
        tcs = [ToolCall("quit_job", {})]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertIsInstance(actions, list)
        self.assertEqual(actions[0].career_move, CareerMove.QUIT_JOB)
        self.assertEqual(issues, [])

    def test_switch_occupation_tool(self):
        tcs = [ToolCall("switch_occupation", {"occupation_id": "nurse"})]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.SWITCH_OCCUPATION)
        self.assertEqual(actions[0].target_occupation_id, "nurse")
        self.assertEqual(issues, [])

    def test_upskill_tool(self):
        tcs = [ToolCall("upskill", {})]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.UPSKILL)
        self.assertEqual(issues, [])

    def test_intensive_work_tool(self):
        tcs = [ToolCall("intensive_work", {})]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.INTENSIVE_WORK)
        self.assertEqual(issues, [])

    def test_rest_tool(self):
        tcs = [ToolCall("rest", {})]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.REST)
        self.assertEqual(issues, [])

    def test_medical_care_tool(self):
        tcs = [ToolCall("medical_care", {})]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.MEDICAL_CARE)
        self.assertEqual(issues, [])

    def test_deposit_tool(self):
        tcs = [ToolCall("deposit", {"amount": 3000})]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.DEPOSIT)
        self.assertEqual(actions[0].amount, 3000)
        self.assertEqual(issues, [])

    def test_withdraw_tool(self):
        tcs = [ToolCall("withdraw", {"amount": 1500})]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.WITHDRAW)
        self.assertEqual(actions[0].amount, 1500)
        self.assertEqual(issues, [])

    def test_defaults_when_no_tools(self):
        actions, issues = WealthSandBoxEnv.apply_tool_calls([])
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].career_move, CareerMove.NONE)
        self.assertEqual(issues, [])

    def test_multiple_tools_in_one_month(self):
        """Agent can call deposit + upskill in the same month."""
        tcs = [
            ToolCall("deposit", {"amount": 3000}),
            ToolCall("upskill", {}),
        ]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(len(actions), 2)
        moves = [a.career_move for a in actions]
        self.assertIn(CareerMove.DEPOSIT, moves)
        self.assertIn(CareerMove.UPSKILL, moves)
        self.assertEqual(issues, [])

    def test_duplicate_tool_not_deduped(self):
        """Two switch calls become two actions (no silent dedup)."""
        tcs = [
            ToolCall("switch_occupation", {"occupation_id": "nurse"}),
            ToolCall("switch_occupation", {"occupation_id": "software_engineer"}),
        ]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].target_occupation_id, "nurse")
        self.assertEqual(actions[1].target_occupation_id, "software_engineer")
        self.assertEqual(issues, [])

    def test_unknown_tool_reports_issue(self):
        tcs = [ToolCall("not_a_real_tool", {})]
        actions, issues = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].career_move, CareerMove.NONE)
        self.assertEqual(len(issues), 1)
        self.assertIn("not_a_real_tool", issues[0])


class TestToolSchemas(unittest.TestCase):

    def test_tools_present(self):
        names = {t["function"]["name"] for t in TOOLS}
        self.assertEqual(names, {"quit_job", "deposit", "withdraw", "borrow", "repay", "switch_occupation", "upskill", "intensive_work", "buy_stock", "sell_stock", "rest", "medical_care"})

    def test_tool_count(self):
        self.assertEqual(len(TOOLS), 12)


class TestDecision(unittest.TestCase):

    def test_decision_creation(self):
        d = Decision(
            reasoning="I am unemployed. I should pick a career.",
            tool_calls=[ToolCall("switch_occupation", {"occupation_id": "software_engineer"})],
        )
        self.assertEqual(d.reasoning, "I am unemployed. I should pick a career.")
        self.assertEqual(len(d.tool_calls), 1)
        self.assertEqual(d.tool_calls[0].tool_name, "switch_occupation")


if __name__ == "__main__":
    unittest.main()
