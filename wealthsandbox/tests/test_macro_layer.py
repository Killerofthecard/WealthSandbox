"""Tests for MacroLayer continuous mode (single long CSV, freeze on exhaustion)."""

import unittest

from wealthsandbox.config import EnvConfig
from wealthsandbox.macro_layer import MacroLayer


class TestMacroLayerContinuous(unittest.TestCase):
    """Continuous mode reads one long CSV from raw_data/ and freezes at the end."""

    def _make(self, **cfg):
        cfg.setdefault("seed", 42)
        cfg["macro_continuous_file"] = "1986_2025.csv"
        return MacroLayer(EnvConfig(**cfg))

    def test_loads_full_series(self):
        macro = self._make()
        self.assertEqual(macro.current_cycle_label, "continuous")
        self.assertEqual(macro.current_cycle_file, "1986_2025.csv")
        self.assertEqual(len(macro._rows), 480)  # 1986-01 .. 2025-12

    def test_first_row_seeded_at_load(self):
        """Construction seeds 'current' with the first row (no fake default month)."""
        macro = self._make()
        # 1986-01: UNRATE 6.7% -> 0.067, SP500_TR 0.00753337
        self.assertAlmostEqual(macro.unrate, 0.067, places=5)
        self.assertAlmostEqual(macro.sp500_tr, 0.00753337, places=8)

    def test_step_advances_to_next_row(self):
        macro = self._make()
        snap = macro.step()
        self.assertEqual(snap["total_months"], 1)
        # 1986-02: UNRATE 7.2% -> 0.072, SP500_TR 0.05698847
        self.assertAlmostEqual(macro.unrate, 0.072, places=5)
        self.assertAlmostEqual(macro.sp500_tr, 0.05698847, places=8)

    def test_freezes_on_exhaustion(self):
        macro = self._make()
        for _ in range(480):
            macro.step()
        self.assertEqual(macro.total_months, 480)
        # 2025-12: UNRATE 4.4% -> 0.044, SP500_TR 0.0176188
        last_unrate, last_sp = macro.unrate, macro.sp500_tr
        self.assertAlmostEqual(last_unrate, 0.044, places=5)
        self.assertAlmostEqual(last_sp, 0.0176188, places=8)

        # One more step: series exhausted -> freeze (no jump, no raise).
        macro.step()
        self.assertEqual(macro.total_months, 481)
        self.assertEqual(macro.unrate, last_unrate)
        self.assertEqual(macro.sp500_tr, last_sp)

    def test_missing_continuous_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            MacroLayer(EnvConfig(macro_continuous_file="does_not_exist.csv", seed=1))

    # --- CPI / inflation / price level ---

    def test_cpi_seeded_at_load(self):
        """CPI is read from the CSV; price level starts at 1.0, inflation at 0."""
        macro = self._make()
        self.assertAlmostEqual(macro.cpi, 109.6, places=5)  # 1986-01 CPI index
        self.assertEqual(macro.price_level, 1.0)
        self.assertEqual(macro.inflation, 0.0)
        self.assertEqual(macro._cpi_0, 109.6)

    def test_inflation_and_price_level_on_step(self):
        """Month-over-month inflation and cumulative price level track the CPI."""
        macro = self._make()
        snap = macro.step()
        # 1986-02 CPI = 109.3 (mild deflation vs 109.6)
        self.assertAlmostEqual(macro.cpi, 109.3, places=5)
        self.assertAlmostEqual(macro.inflation, 109.3 / 109.6 - 1.0, places=6)
        self.assertAlmostEqual(macro.price_level, 109.3 / 109.6, places=6)
        # Snapshot exposes all three.
        self.assertIn("cpi", snap)
        self.assertIn("inflation", snap)
        self.assertIn("price_level", snap)
        self.assertAlmostEqual(snap["price_level"], 109.3 / 109.6, places=6)

    def test_price_level_accumulates_over_series(self):
        """By end of series the price level is CPI_end / CPI_start (>1, net inflation)."""
        macro = self._make()
        for _ in range(480):
            macro.step()
        # 2025-12 CPI = 324.054, start = 109.6
        self.assertAlmostEqual(macro.price_level, 324.054 / 109.6, places=6)
        self.assertGreater(macro.price_level, 1.0)


if __name__ == "__main__":
    unittest.main()
