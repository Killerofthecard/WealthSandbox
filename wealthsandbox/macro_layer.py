"""Macro-layer: drives external economic conditions from real historical data.

Reads UNRATE and USRECM (NBER recession flag) from pre-processed cycle CSVs.
These two parameters flow into CareerSystem where they affect layoff probability,
rehire probability, and industry income multipliers.  The agent does NOT see
UNRATE / USRECM directly — it only feels the effects (layoffs, rehire
difficulty, income changes).

Set ``macro_cycle`` in EnvConfig to "boom", "normal", or "recession" to lock
a specific cycle type.  Leave empty (default) for random selection.
"""

import csv
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from wealthsandbox.config import EnvConfig


# ---------------------------------------------------------------------------
# MacroLayer
# ---------------------------------------------------------------------------

class MacroLayer:
    """Tracks calendar time and provides UNRATE + USRECM + stock data to systems."""

    def __init__(self, config: EnvConfig):
        self.year: int = config.start_year
        self.month: int = config.start_month
        self._month_counter: int = 0

        # Current values (consumed by CareerSystem + AssetSystem)
        self.unrate: float = 0.05      # decimal (e.g. 0.054 = 5.4%)
        self.usrecm: int = 0           # 0 or 1
        self.fedfunds: float = 3.0     # % (e.g. 5.25 = 5.25%)
        self.sp500_tr: float = 0.0     # S&P 500 total return this month
        self.gs10: float = 3.0         # 10-year Treasury rate (%)
        self._has_stock_data: bool = False

        # Which cycles are being used
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
        self._rng = random.Random(config.seed if config.seed is not None else 42)
        self._rows: List[Tuple[int, int, float, int, float]] = []  # (year, month, unrate, usrecm, fedfunds)
        self._row_idx: int = 0

        # Stock data index: (year, month) -> (sp500_tr, gs10)
        self._stock_data: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._require_stock: bool = config.require_stock_data
        self._load_stock_data(config)

        self._load_next_cycle()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_months(self) -> int:
        return self._month_counter

    def step(self) -> Dict[str, Any]:
        """Advance the calendar by one month and read next macro row."""
        self._month_counter += 1
        self.month += 1
        if self.month > 12:
            self.month = 1
            self.year += 1

        if self._row_idx < len(self._rows):
            _, _, self.unrate, self.usrecm, self.fedfunds = self._rows[self._row_idx]
            self._row_idx += 1
        else:
            self._load_next_cycle()
            if self._rows:
                _, _, self.unrate, self.usrecm, self.fedfunds = self._rows[0]
                self._row_idx = 1

        # Look up stock data for the NEW (current) month
        key = (self.year, self.month)
        if key in self._stock_data:
            self.sp500_tr, self.gs10 = self._stock_data[key]
            self._has_stock_data = True
        else:
            self.sp500_tr = 0.0
            self.gs10 = 0.0
            self._has_stock_data = False

        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        """Return current state (consumed by env and systems)."""
        return {
            "year": self.year,
            "month": self.month,
            "total_months": self._month_counter,
            "unrate": self.unrate,
            "usrecm": self.usrecm,
            "fedfunds": self.fedfunds,
            "sp500_tr": self.sp500_tr,
            "gs10": self.gs10,
        }

    def reset(self) -> None:
        self._month_counter = 0
        self.year = 2024
        self.month = 1
        self._row_idx = 0
        self.unrate = 0.05
        self.usrecm = 0
        self.fedfunds = 3.0
        self.sp500_tr = 0.0
        self.gs10 = 3.0
        self._has_stock_data = False
        self._load_next_cycle()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_next_cycle(self) -> None:
        """Pick a cycle CSV and load all rows.

        Priority: ``macro_cycle_file`` > ``macro_cycle`` > random.
        """
        self._rows.clear()
        self._row_idx = 0

        path: Optional[str] = None
        pick_label = ""

        # 1. Specific file override
        if self._file_override:
            # Search in all three directories
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
                    self._rows.append((
                        int(row["year"]),
                        int(row["month"]),
                        float(row["UNRATE"]) / 100.0,
                        int(row["USREC"]),
                        fedfunds,
                    ))

        # Validate stock data coverage for loaded rows
        if self._require_stock and self._rows:
            missing: List[str] = []
            for year, month, _, _, _ in self._rows:
                if (year, month) not in self._stock_data:
                    missing.append(f"{year}-{month:02d}")
            if missing:
                raise RuntimeError(
                    f"Stock data missing for {len(missing)} months in "
                    f"'{self.current_cycle_file}' ({pick_label} cycle): "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''}.  "
                    f"Ensure '{self._data_dir}/stock_monthly.csv' covers "
                    f"the scenario date range."
                )

    # ------------------------------------------------------------------
    # Stock data loading
    # ------------------------------------------------------------------

    def _load_stock_data(self, config: EnvConfig) -> None:
        """Load preprocessed stock_monthly.csv into a lookup dict."""
        stock_path = os.path.join(self._data_dir, config.stock_data_file)
        if not os.path.isfile(stock_path):
            if self._require_stock:
                raise FileNotFoundError(
                    f"Stock data file not found: {stock_path}.  "
                    f"Run 'python scripts/prepare_stock_data.py' first, "
                    f"or set require_stock_data=False in EnvConfig."
                )
            return

        with open(stock_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                y = int(row["year"])
                m = int(row["month"])
                tr_str = row.get("SP500_TR", "").strip()
                gs10_str = row.get("GS10", "").strip()
                # Skip rows with missing return (first row is always NaN)
                if not tr_str:
                    continue
                sp500_tr = float(tr_str)
                gs10 = float(gs10_str) if gs10_str else 0.0
                self._stock_data[(y, m)] = (sp500_tr, gs10)
