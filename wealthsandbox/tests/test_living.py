"""Tests for the LivingExpenseSystem (CPI-driven price-level scaling)."""

import unittest

from wealthsandbox.systems.living import LivingExpenseSystem
from wealthsandbox.types import AgentState


class TestLivingExpenseSystem(unittest.TestCase):

    def test_expense_scales_with_price_level(self):
        sys = LivingExpenseSystem(monthly_living_expense=2_000.0)
        state = AgentState(cash=10_000.0)

        sys.finalize(state, {"price_level": 1.0})
        # 10_000 - 2_000 = 8_000
        self.assertAlmostEqual(state.cash, 8_000.0, places=2)

        sys.finalize(state, {"price_level": 2.5})
        # 8_000 - 2_000*2.5 = 3_000
        self.assertAlmostEqual(state.cash, 3_000.0, places=2)

    def test_expense_defaults_to_one_when_no_price_level(self):
        sys = LivingExpenseSystem(monthly_living_expense=2_000.0)
        state = AgentState(cash=10_000.0)
        sys.finalize(state, {})  # no price_level key -> 1.0
        self.assertAlmostEqual(state.cash, 8_000.0, places=2)

    def test_shortfall_covered_from_savings(self):
        sys = LivingExpenseSystem(monthly_living_expense=2_000.0)
        state = AgentState(cash=500.0, savings=5_000.0)
        sys.finalize(state, {"price_level": 1.0})
        # 500 - 2000 = -1500 -> pull 1500 from savings -> cash 0, savings 3500
        self.assertAlmostEqual(state.cash, 0.0, places=2)
        self.assertAlmostEqual(state.savings, 3_500.0, places=2)


if __name__ == "__main__":
    unittest.main()
