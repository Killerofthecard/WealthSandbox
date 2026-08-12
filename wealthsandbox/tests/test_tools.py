"""Tests for tool-call parsing and the LLMAgent Decision data model."""

import unittest

from wealthsandbox.env import WealthSandBoxEnv
from wealthsandbox.agents.tools import ToolCall, Decision, TOOLS
from wealthsandbox.types import Action, CareerMove


class TestApplyToolCalls(unittest.TestCase):

    def test_quit_job_tool(self):
        tcs = [ToolCall("quit_job", {})]
        actions = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertIsInstance(actions, list)
        self.assertEqual(actions[0].career_move, CareerMove.QUIT_JOB)

    def test_switch_occupation_tool(self):
        tcs = [ToolCall("switch_occupation", {"occupation_id": "nurse"})]
        actions = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.SWITCH_OCCUPATION)
        self.assertEqual(actions[0].target_occupation_id, "nurse")

    def test_upskill_tool(self):
        tcs = [ToolCall("upskill", {})]
        actions = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.UPSKILL)

    def test_intensive_work_tool(self):
        tcs = [ToolCall("intensive_work", {})]
        actions = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.INTENSIVE_WORK)

    def test_deposit_tool(self):
        tcs = [ToolCall("deposit", {"amount": 3000})]
        actions = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.DEPOSIT)
        self.assertEqual(actions[0].amount, 3000)

    def test_withdraw_tool(self):
        tcs = [ToolCall("withdraw", {"amount": 1500})]
        actions = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(actions[0].career_move, CareerMove.WITHDRAW)
        self.assertEqual(actions[0].amount, 1500)

    def test_defaults_when_no_tools(self):
        actions = WealthSandBoxEnv.apply_tool_calls([])
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].career_move, CareerMove.NONE)

    def test_multiple_tools_in_one_month(self):
        """Agent can call deposit + upskill in the same month."""
        tcs = [
            ToolCall("deposit", {"amount": 3000}),
            ToolCall("upskill", {}),
        ]
        actions = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(len(actions), 2)
        moves = [a.career_move for a in actions]
        self.assertIn(CareerMove.DEPOSIT, moves)
        self.assertIn(CareerMove.UPSKILL, moves)

    def test_duplicate_tool_ignored(self):
        tcs = [
            ToolCall("switch_occupation", {"occupation_id": "nurse"}),
            ToolCall("switch_occupation", {"occupation_id": "software_engineer"}),
        ]
        actions = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].target_occupation_id, "nurse")


class TestToolSchemas(unittest.TestCase):

    def test_tools_present(self):
        names = {t["function"]["name"] for t in TOOLS}
        self.assertEqual(names, {"quit_job", "deposit", "withdraw", "borrow", "repay", "switch_occupation", "upskill", "intensive_work"})

    def test_tool_count(self):
        self.assertEqual(len(TOOLS), 8)


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
