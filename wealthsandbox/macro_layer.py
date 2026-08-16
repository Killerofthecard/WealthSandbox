"""Macro-layer: drives external economic conditions from real historical data.

Reads UNRATE, USRECM (NBER recession flag), FEDFUNDS, and SP500_TR from
pre-processed cycle CSVs.  All four live in the SAME row, so the stock return
is always aligned with the labour-market conditions of that exact month — there
is no separate calendar to drift out of sync with.

These parameters flow into CareerSystem (layoff probability, rehire
probability, industry income multipliers) and AssetSystem (stock returns).
The agent does NOT see UNRATE / USREC / FEDFUNDS / SP500_TR directly — it only
feels their effects (layoffs, rehire difficulty, income changes, stock moves).

Time is measured purely as an elapsed-month counter (``total_months``).  There
is no calendar year; the scenario replays a cycle's rows in order.

Set ``macro_cycle`` in EnvConfig to "boom", "normal", or "recession" to lock
a specific cycle type.  Leave empty (default) for random selection.
"""

import csv
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from wealthsandbox.config import EnvConfig


class MacroLayer:
    """Tracks elapsed months and provides UNRATE + USREC + FEDFUNDS + SP500_TR
    to systems, read straight from the selected cycle's rows."""

    def __init__(self, config: EnvConfig):
        self._month_counter: int = 0

        # Current values (consumed by CareerSystem + AssetSystem + BankSystem)
        self.unrate: float = 0.05      # decimal (e.g. 0.054 = 5.4%)
        self.usrecm: int = 0           # 0 or 1
        self.fedfunds: float = 3.0     # % (e.g. 5.25 = 5.25%)
        self.sp500_tr: float = 0.0     # S&P 500 total return this month
        self.cpi: float = 0.0          # CPI index level this month
        self.inflation: float = 0.0    # month-over-month inflation (decimal)
        self.price_level: float = 1.0  # cumulative CPI / base CPI (start = 1.0)
        self._cpi_0: float = 0.0       # CPI at simulation start (price_level denominator)

        # Which cycle is being used
        self.current_cycle_label: str = ""
        self.current_cycle_file: str = ""

        # Cycle data
        self._data_dir: str = config.macro_data_dir
        if not os.path.isabs(self._data_dir):
            # 相对路径锚定到项目根目录（wealthsandbox/ 的上一级），避免受运行时 cwd 影响
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self._data_dir = os.path.join(project_root, self._data_dir)
        self._cycle_override: str = config.macro_cycle  # "" = random
        self._file_override: str = config.macro_cycle_file  # specific CSV file
        self._continuous_file: str = config.macro_continuous_file  # single long CSV
        self._rng = random.Random(config.seed if config.seed is not None else 42)

        # Rows: (unrate, usrecm, fedfunds, sp500_tr, cpi) — one per month, in order.
        self._rows: List[Tuple[float, int, float, float, float]] = []
        self._row_idx: int = 0

        self._load_next_cycle()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_months(self) -> int:
        return self._month_counter

    def step(self) -> Dict[str, Any]:
        """Advance by one month and read the next macro row."""
        self._month_counter += 1

        if self._row_idx < len(self._rows):
            prev_cpi = self.cpi
            self.unrate, self.usrecm, self.fedfunds, self.sp500_tr, self.cpi = self._rows[self._row_idx]
            self._row_idx += 1
            # Month-over-month inflation (negative = deflation).
            if prev_cpi > 0 and self.cpi > 0:
                self.inflation = self.cpi / prev_cpi - 1.0
            else:
                self.inflation = 0.0
            # Cumulative price level relative to the series' first month.
            if self._cpi_0 > 0 and self.cpi > 0:
                self.price_level = self.cpi / self._cpi_0
            else:
                self.price_level = 1.0
        elif self._continuous_file:
            # Continuous mode: series exhausted — freeze on the last row's
            # values instead of chaining a random next cycle.
            pass
        else:
            # Cycle mode: series exhausted — load the next (random) cycle.
            # _load_next_cycle() seeds current values with the new first row
            # and resets price_level / inflation.
            self._load_next_cycle()

        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        """Return current state (consumed by env and systems)."""
        return {
            "total_months": self._month_counter,
            "unrate": self.unrate,
            "usrecm": self.usrecm,
            "fedfunds": self.fedfunds,
            "sp500_tr": self.sp500_tr,
            "cpi": self.cpi,
            "inflation": self.inflation,
            "price_level": self.price_level,
        }

    def reset(self) -> None:
        self._month_counter = 0
        self._row_idx = 0
        self.unrate = 0.05
        self.usrecm = 0
        self.fedfunds = 3.0
        self.sp500_tr = 0.0
        self.cpi = 0.0
        self.inflation = 0.0
        self.price_level = 1.0
        self._cpi_0 = 0.0
        self._load_next_cycle()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_next_cycle(self) -> None:
        """Pick a cycle CSV and load all rows.

        Priority: ``macro_continuous_file`` > ``macro_cycle_file`` >
        ``macro_cycle`` > random.
        """
        self._rows.clear()
        self._row_idx = 0

        path: Optional[str] = None
        pick_label = ""

        # 0. Continuous mode — single long CSV at the data-dir root.
        if self._continuous_file:
            candidate = os.path.join(self._data_dir, self._continuous_file)
            if not os.path.isfile(candidate):
                raise FileNotFoundError(
                    f"macro_continuous_file '{self._continuous_file}' not found "
                    f"under '{self._data_dir}'"
                )
            path = candidate
            pick_label = "continuous"

        # 1. Specific file override
        elif self._file_override:
            for label in ("boom", "normal", "recession"):
                candidate = os.path.join(self._data_dir, label, self._file_override)
                if os.path.isfile(candidate):
                    path = candidate
                    pick_label = label
                    break
            if path is None:
                raise FileNotFoundError(
                    f"macro_cycle_file '{self._file_override}' not found in "
                    f"{{boom,normal,recession}} under '{self._data_dir}'"
                )
        else:
            # 2. Label filter or random
            labels = ("boom", "normal", "recession")
            if self._cycle_override:
                if self._cycle_override not in labels:
                    raise ValueError(
                        f"macro_cycle must be one of {labels}, "
                        f"got '{self._cycle_override}'"
                    )
                labels = (self._cycle_override,)

            cycles: List[Tuple[str, str]] = []
            for label in labels:
                d = os.path.join(self._data_dir, label)
                if os.path.isdir(d):
                    for fname in sorted(os.listdir(d)):
                        if fname.endswith(".csv"):
                            cycles.append((label, os.path.join(d, fname)))
            if not cycles:
                return

            pick_label, path = self._rng.choice(cycles)

        self.current_cycle_label = pick_label
        self.current_cycle_file = os.path.basename(path) if path else ""

        if path:
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ff_str = row.get("FEDFUNDS", "").strip()
                    fedfunds = float(ff_str) if ff_str else 3.0
                    sp_str = row.get("SP500_TR", "").strip()
                    sp500_tr = float(sp_str) if sp_str else 0.0
                    cpi_str = row.get("CPI", "").strip()
                    cpi = float(cpi_str) if cpi_str else 0.0
                    self._rows.append((
                        float(row["UNRATE"]) / 100.0,
                        int(row["USREC"]),
                        fedfunds,
                        sp500_tr,
                        cpi,
                    ))

        # Seed "current" with the first row so the very first month's tick uses
        # real data.  Fixes the off-by-one where month 1 previously ran on the
        # hardcoded defaults and the series' last row was never experienced.
        if self._rows:
            self.unrate, self.usrecm, self.fedfunds, self.sp500_tr, self.cpi = self._rows[0]
            self._row_idx = 1
            self._cpi_0 = self.cpi if self.cpi > 0 else 0.0
            self.price_level = 1.0
            self.inflation = 0.0
