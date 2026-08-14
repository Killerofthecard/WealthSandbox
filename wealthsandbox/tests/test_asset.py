"""Tests for the AssetSystem (stock market) and its integration with the env."""

import unittest
from wealthsandbox.env import WealthSandBoxEnv
from wealthsandbox.config import EnvConfig
from wealthsandbox.profile import AgentProfile
from wealthsandbox.types import Action, CareerMove, AgentState, JobStatus
from wealthsandbox.systems.asset import AssetSystem


class TestAssetSystem(unittest.TestCase):
    """Unit tests for AssetSystem in isolation."""

    def _make_state(self, cash=10_000.0, stock_value=0.0, total_invested=0.0):
        s = AgentState()
        s.cash = cash
        s.stock_value = stock_value
        s.total_invested = total_invested
        return s

    def test_buy_adds_to_stock_value(self):
        sys = AssetSystem()
        s = self._make_state(cash=5_000)
        sys._handle_buy(s, 2_000)
        self.assertEqual(s.cash, 3_000)
        self.assertEqual(s.stock_value, 2_000)
        self.assertEqual(s.total_invested, 2_000)

    def test_sell_reduces_stock_value(self):
        sys = AssetSystem()
        s = self._make_state(cash=5_000, stock_value=4_000, total_invested=4_000)
        sys._handle_sell(s, 2_000)
        self.assertEqual(s.stock_value, 2_000)
        # Sale proceeds go to temp field, moved to pending in finalize()
        self.assertEqual(s._this_month_stock_sales, 2_000)
        self.assertEqual(s.cash, 5_000)  # cash unchanged until settlement
        # total_invested reduced proportionally
        self.assertAlmostEqual(s.total_invested, 2_000)
        # After finalize, pending should be set
        sys.finalize(s, {})
        self.assertEqual(s.pending_settlement, 2_000)
        self.assertEqual(s._this_month_stock_sales, 0.0)

    def test_sell_clamped_to_stock_value(self):
        sys = AssetSystem()
        s = self._make_state(cash=5_000, stock_value=1_000, total_invested=1_000)
        sys._handle_sell(s, 5_000)  # tries to sell more than held
        self.assertEqual(s.stock_value, 0)
        self.assertEqual(s._this_month_stock_sales, 1_000)
        self.assertEqual(s.total_invested, 0)
        # After finalize
        sys.finalize(s, {})
        self.assertEqual(s.pending_settlement, 1_000)

    def test_market_return_applied_in_tick(self):
        """stock_value should change by the market return in tick."""
        sys = AssetSystem()
        s = self._make_state(stock_value=10_000, total_invested=10_000)
        macro = {"sp500_tr": 0.10, "gs10": 4.0}
        sys.tick(s, macro)
        self.assertAlmostEqual(s.stock_value, 11_000)
        self.assertAlmostEqual(s.last_month_stock_return, 0.10)

    def test_negative_return_applied_in_tick(self):
        sys = AssetSystem()
        s = self._make_state(stock_value=10_000, total_invested=10_000)
        macro = {"sp500_tr": -0.20, "gs10": 4.0}
        sys.tick(s, macro)
        self.assertAlmostEqual(s.stock_value, 8_000)

    def test_new_purchase_excluded_from_this_month_return(self):
        """Purchases made in buy_stock do NOT earn this month's return."""
        sys = AssetSystem()
        s = self._make_state(stock_value=10_000, total_invested=10_000)
        # Simulate buying 5_000 this month
        s._this_month_stock_purchases = 5_000
        s.stock_value = 15_000  # 10K old + 5K new

        macro = {"sp500_tr": 0.10, "gs10": 4.0}
        sys.tick(s, macro)
        # Only the pre-existing 10_000 should get the +10% return.
        # 10_000 * 1.10 + 5_000 = 16_000
        self.assertAlmostEqual(s.stock_value, 16_000)
        self.assertEqual(s._this_month_stock_purchases, 0.0)

    def test_settlement_arrives_in_tick(self):
        """Sale proceeds from last month should arrive as cash in tick."""
        sys = AssetSystem()
        s = self._make_state(cash=1_000, stock_value=0)
        s.pending_settlement = 3_000
        macro = {"sp500_tr": 0.0, "gs10": 4.0}
        sys.tick(s, macro)
        self.assertEqual(s.cash, 4_000)
        self.assertEqual(s.pending_settlement, 0)

    def test_force_liquidate_raises_cash_at_discount(self):
        sys = AssetSystem(forced_sale_discount=0.10)
        s = self._make_state(cash=0, stock_value=10_000, total_invested=10_000)
        raised = sys.force_liquidate(s, 5_000)
        # Max raised = 10_000 * (1 - 0.10) = 9_000
        # Need 5_000, so raise 5_000.
        # Stock consumed = 5_000 / 0.9 ≈ 5_555.56
        self.assertAlmostEqual(raised, 5_000)
        self.assertAlmostEqual(s.cash, 5_000)
        self.assertAlmostEqual(s.stock_value, 10_000 - 5_000 / 0.9)
        # Total invested reduced proportionally
        self.assertLess(s.total_invested, 10_000)

    def test_force_liquidate_capped_by_stock_value(self):
        sys = AssetSystem(forced_sale_discount=0.10)
        s = self._make_state(cash=0, stock_value=1_000, total_invested=1_000)
        raised = sys.force_liquidate(s, 10_000)
        # Can only raise 1_000 * 0.9 = 900
        self.assertAlmostEqual(raised, 900)
        self.assertAlmostEqual(s.stock_value, 0)

    def test_force_liquidate_no_stocks(self):
        sys = AssetSystem()
        s = self._make_state(cash=0, stock_value=0)
        raised = sys.force_liquidate(s, 1_000)
        self.assertEqual(raised, 0)

    def test_check_dead_positive_net_worth(self):
        sys = AssetSystem()
        s = self._make_state(cash=1_000)
        self.assertIsNone(sys.check_dead(s))

    def test_check_dead_zero_net_worth(self):
        sys = AssetSystem()
        s = self._make_state(cash=0)
        s.savings = 0
        s.stock_value = 0
        s.pending_settlement = 0
        s.loan_balance = 0
        self.assertEqual(sys.check_dead(s), "bankruptcy")

    def test_check_dead_negative_net_worth(self):
        sys = AssetSystem()
        s = self._make_state(cash=-100, stock_value=50)
        s.loan_balance = 200
        # net = -100 + 50 - 200 = -250
        self.assertEqual(sys.check_dead(s), "bankruptcy")

    def test_check_dead_stocks_count_toward_net_worth(self):
        """Stock value should prevent bankruptcy even when cash is low."""
        sys = AssetSystem()
        s = self._make_state(cash=100, stock_value=5_000)
        self.assertIsNone(sys.check_dead(s))


class TestAssetIntegration(unittest.TestCase):
    """Integration tests: AssetSystem wired into WealthSandBoxEnv."""

    def test_buy_stock_via_env(self):
        """Buy stock through the full env step pipeline."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        # Get a job first for income
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        cash_before = env.micro.state.cash

        # Buy $1,000 of stock
        action = Action(career_move=CareerMove.BUY_STOCK, amount=1_000)
        obs, reward, done, info = env.step(action=action)

        self.assertFalse(info.get("action_rejected"))
        self.assertAlmostEqual(env.micro.state.cash, cash_before + env.micro.state.monthly_after_tax_income - 2_000 - 1_000, places=2)
        self.assertAlmostEqual(env.micro.state.stock_value, 1_000)
        self.assertAlmostEqual(env.micro.state.total_invested, 1_000)

    def test_sell_stock_via_env(self):
        """Sell stock — cash unchanged, pending_settlement populated."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        # Get a job
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        # Buy stock
        env.step(action=Action(career_move=CareerMove.BUY_STOCK, amount=1_000))
        cash_before = env.micro.state.cash

        # Sell stock
        action = Action(career_move=CareerMove.SELL_STOCK, amount=500)
        obs, reward, done, info = env.step(action=action)

        self.assertFalse(info.get("action_rejected"))
        # Settlement is deferred to finalize, so after full step it appears
        self.assertAlmostEqual(env.micro.state.pending_settlement, 500)
        # Remaining stock = $500 marked-to-market by this month's return.
        ret = env.micro.state.last_month_stock_return
        self.assertAlmostEqual(env.micro.state.stock_value, 500 * (1 + ret), places=2)

    def test_sell_settlement_arrives_next_month(self):
        """T+1: sale cash should arrive in the NEXT month's tick."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        env.step(action=Action(career_move=CareerMove.BUY_STOCK, amount=1_000))
        env.step(action=Action(career_move=CareerMove.SELL_STOCK, amount=500))

        # Next month — settlement should arrive
        cash_before = env.micro.state.cash
        obs, reward, done, info = env.step(action=Action(career_move=CareerMove.NONE))
        # Cash should have: previous cash + income - 2000 + 500 settlement
        expected = cash_before + env.micro.state.monthly_after_tax_income - 2_000 + 500
        self.assertAlmostEqual(env.micro.state.cash, expected, places=2)
        self.assertAlmostEqual(env.micro.state.pending_settlement, 0)

    def test_buy_stock_rejected_when_cash_insufficient(self):
        """Buying stock when cash would go below buffer should be rejected."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        # Agent has $100,000 cash. Buy $99,000 → leaves $1,000 < $2,000 buffer.
        action = Action(career_move=CareerMove.BUY_STOCK, amount=99_000)
        obs, reward, done, info = env.step(action=action)
        self.assertTrue(info.get("action_rejected"))

    def test_sell_stock_rejected_when_no_holdings(self):
        """Selling stock with 0 stock_value should be rejected."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        action = Action(career_move=CareerMove.SELL_STOCK, amount=500)
        obs, reward, done, info = env.step(action=action)
        self.assertTrue(info.get("action_rejected"))

    def test_sell_more_than_held_rejected(self):
        """Selling more than stock_value should be rejected by the amount guard."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        env.step(action=Action(career_move=CareerMove.BUY_STOCK, amount=1_000))

        action = Action(career_move=CareerMove.SELL_STOCK, amount=5_000)
        obs, reward, done, info = env.step(action=action)
        self.assertTrue(info.get("action_rejected"))

    def test_force_liquidate_on_living_expense_shortfall(self):
        """When cash+savings are exhausted, stocks should be force-sold.

        Agent quits job first (no income), has only $100 cash and $0 savings,
        so living expenses force stock liquidation.
        """
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        # Get a job and buy stocks
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        env.step(action=Action(career_move=CareerMove.BUY_STOCK, amount=5_000))
        # Quit job so no income arrives
        env.step(action=Action(career_move=CareerMove.QUIT_JOB))

        # Now wipe out cash and savings before next month
        env.micro.state.cash = 100
        env.micro.state.savings = 0
        # Clear _this_month_stock_purchases so tick applies return to ALL stock
        env.micro.state._this_month_stock_purchases = 0.0
        stock_before = env.micro.state.stock_value

        obs, reward, done, info = env.step(action=Action(career_move=CareerMove.NONE))
        # Unemployed → no income → cash ($100) < living expense ($2,000)
        events = obs.individual.get("last_month_events", [])
        self.assertTrue(
            any("EMERGENCY" in e for e in events),
            f"Expected emergency sale event, got: {events}"
        )

    def test_bankruptcy_with_stocks(self):
        """When net worth goes to 0 or below, agent goes bankrupt."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        # Wipe everything
        env.micro.state.cash = 0
        env.micro.state.savings = 0
        env.micro.state.stock_value = 0
        env.micro.state.loan_balance = 0

        obs, reward, done, info = env.step(action=Action(career_move=CareerMove.NONE))
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "bankruptcy")

    def test_stocks_shown_in_observation(self):
        """Observation should include stock info when holdings > 0."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        env.step(action=Action(career_move=CareerMove.BUY_STOCK, amount=1_000))

        obs, _, _, _ = env.step(action=Action(career_move=CareerMove.NONE))
        self.assertGreater(obs.individual.get("stock_value", 0), 0)
        self.assertIn("Stocks", obs.narrative)

    def test_stocks_in_trajectory(self):
        """Trajectory JSON should include stock fields."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        env.step(action=Action(career_move=CareerMove.BUY_STOCK, amount=1_000))

        self.assertEqual(len(env.history), 2)
        self.assertIn("stock_value", env.history[-1])
        self.assertIn("pending_settlement", env.history[-1])
        self.assertIn("total_invested", env.history[-1])

    def test_no_stock_operation_does_not_break_existing_behavior(self):
        """Regression: an episode with NO stock operations behaves identically."""
        env = WealthSandBoxEnv(EnvConfig(seed=1))
        env.reset(seed=1)

        # Get job, work for a few months
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))
        cash1 = env.micro.state.cash

        # Month 2: auto-work
        env.step(action=Action(career_move=CareerMove.NONE))
        self.assertGreater(env.micro.state.cash, 0)
        self.assertEqual(env.micro.state.stock_value, 0)
        self.assertEqual(env.micro.state.pending_settlement, 0)

        # Month 3: deposit some cash
        env.step(action=Action(career_move=CareerMove.DEPOSIT, amount=2_000))
        self.assertGreater(env.micro.state.savings, 0)

        # Should not be dead
        self.assertGreater(env.micro.state.cash, 0)

    def test_2008_october_crash_event(self):
        """Integration: buy stocks through 2008, verify October crash event.

        SP500_TR now lives in the cycle CSV row, so replaying the 2008-2009
        scenario naturally feeds October's crash (-20.2%) to AssetSystem.
        """
        env = WealthSandBoxEnv(EnvConfig(
            seed=42,
            macro_cycle="recession",
            macro_cycle_file="2008_2009.csv",
        ))
        env.reset(seed=42)
        # First get a job
        env.step(action=Action(
            career_move=CareerMove.SWITCH_OCCUPATION,
            target_occupation_id="manufacturing_worker",
        ))

        # Buy stocks every month from Feb through Sep 2008
        for i in range(8):
            env.step(action=Action(career_move=CareerMove.BUY_STOCK, amount=500))

        # Now we are at month 10 (October 2008). The macro_snapshot used
        # in this step's tick has sp500_tr for September 2008. We need one
        # more step to see October's return. But actually, the sp500_tr in
        # macro_snapshot is the value loaded by the PREVIOUS step's macro.step().
        #
        # After step 9 (8th buy), macro.step() advances to 2008-10 and loads
        # sp500_tr for October. That value appears in step 10's tick.

        # Step 10: auto-work, see October event
        obs, _, _, _ = env.step(action=Action(career_move=CareerMove.NONE))

        events = env.micro.state.last_month_events
        stock_events = [e for e in events if "stock holdings" in e.lower()]
        self.assertTrue(len(stock_events) > 0, f"Expected stock event, got events: {events}")

        # October 2008's return is strongly negative (-20.2%).
        self.assertIn("lost", stock_events[0].lower())


if __name__ == "__main__":
    unittest.main()
