"""Tests for tool-call parsing and the LLMAgent Decision data model."""

import unittest

from wealthsandbox.env import WealthSandBoxEnv
from wealthsandbox.agents.tools import ToolCall, Decision, TOOLS
from wealthsandbox.types import Action, CareerMove


class TestApplyToolCalls(unittest.TestCase):

    def test_quit_job_tool(self):
        tcs = [ToolCall("quit_job", {})]
        action = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(action.career_move, CareerMove.QUIT_JOB)

    def test_switch_occupation_tool(self):
        tcs = [ToolCall("switch_occupation", {"occupation_id": "nurse"})]
        action = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(action.career_move, CareerMove.SWITCH_OCCUPATION)
        self.assertEqual(action.target_occupation_id, "nurse")

    def test_upskill_tool(self):
        tcs = [ToolCall("upskill", {})]
        action = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(action.career_move, CareerMove.UPSKILL)

    def test_intensive_work_tool(self):
        tcs = [ToolCall("intensive_work", {})]
        action = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(action.career_move, CareerMove.INTENSIVE_WORK)

    def test_defaults_when_no_tools(self):
        action = WealthSandBoxEnv.apply_tool_calls([])
        self.assertEqual(action.career_move, CareerMove.NONE)
        self.assertEqual(action.target_occupation_id, "")

    def test_duplicate_tool_ignored(self):
        tcs = [
            ToolCall("switch_occupation", {"occupation_id": "nurse"}),
            ToolCall("switch_occupation", {"occupation_id": "software_engineer"}),
        ]
        action = WealthSandBoxEnv.apply_tool_calls(tcs)
        self.assertEqual(action.target_occupation_id, "nurse")  # first wins


class TestToolSchemas(unittest.TestCase):

    def test_tools_present(self):
        names = {t["function"]["name"] for t in TOOLS}
        self.assertEqual(names, {"quit_job", "switch_occupation", "upskill", "intensive_work"})

    def test_tool_count(self):
        self.assertEqual(len(TOOLS), 4)


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
