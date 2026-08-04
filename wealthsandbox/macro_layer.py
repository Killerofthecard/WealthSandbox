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
    """Tracks calendar time and provides UNRATE + USRECM to systems."""

    def __init__(self, config: EnvConfig):
        self.year: int = config.start_year
        self.month: int = config.start_month
        self._month_counter: int = 0

        # Current values (consumed by CareerSystem)
        self.unrate: float = 0.05      # decimal (e.g. 0.054 = 5.4%)
        self.usrecm: int = 0           # 0 or 1

        # Which cycles are being used
        self.current_cycle_label: str = ""
        self.current_cycle_file: str = ""

        # Cycle data
        self._data_dir: str = config.macro_data_dir
        self._cycle_override: str = config.macro_cycle  # "" = random
        self._file_override: str = config.macro_cycle_file  # specific CSV file
        self._rng = random.Random(config.seed if config.seed is not None else 42)
        self._rows: List[Tuple[int, int, float, int]] = []  # (year, month, unrate, usrecm)
        self._row_idx: int = 0

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
            _, _, self.unrate, self.usrecm = self._rows[self._row_idx]
            self._row_idx += 1
        else:
            self._load_next_cycle()
            if self._rows:
                _, _, self.unrate, self.usrecm = self._rows[0]
                self._row_idx = 1

        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        """Return current state (consumed by env and systems)."""
        return {
            "year": self.year,
            "month": self.month,
            "total_months": self._month_counter,
            "unrate": self.unrate,
            "usrecm": self.usrecm,
        }

    def reset(self) -> None:
        self._month_counter = 0
        self.year = 2024
        self.month = 1
        self._row_idx = 0
        self.unrate = 0.05
        self.usrecm = 0
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
                    self._rows.append((
                        int(row["year"]),
                        int(row["month"]),
                        float(row["UNRATE"]) / 100.0,
                        int(row["USREC"]),
                    ))
