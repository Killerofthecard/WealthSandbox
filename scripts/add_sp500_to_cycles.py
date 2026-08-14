#!/usr/bin/env python3
"""Add an SP500_TR column to every macro cycle CSV.

Reads S&P 500 monthly total returns from ``raw_data/stock_monthly.csv`` and
joins them onto each ``raw_data/{boom,normal,recession}/*.csv`` by (year, month),
so the stock return lives in the SAME row as the macro indicators (UNRATE /
USREC / FEDFUNDS).  This guarantees the stock series is aligned to the cycle's
dates — no more cross-calendar lookup in ``MacroLayer``.

Input rows:  year,month,UNRATE,USREC,FEDFUNDS
Output rows: year,month,UNRATE,USREC,FEDFUNDS,SP500_TR
"""

import csv
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(REPO_ROOT, "raw_data")
STOCK_PATH = os.path.join(RAW_DATA, "stock_monthly.csv")
CYCLE_DIRS = ("boom", "normal", "recession")


def load_sp500() -> dict[tuple[int, int], str]:
    """Return {(year, month): SP500_TR} from stock_monthly.csv."""
    out: dict[tuple[int, int], str] = {}
    with open(STOCK_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tr = (row.get("SP500_TR") or "").strip()
            if not tr:
                continue
            out[(int(row["year"]), int(row["month"]))] = tr
    return out


def main() -> None:
    sp500 = load_sp500()
    print(f"Loaded {len(sp500)} (year, month) -> SP500_TR entries.")

    total = 0
    missing = 0
    for label in CYCLE_DIRS:
        d = os.path.join(RAW_DATA, label)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".csv"):
                continue
            path = os.path.join(d, fname)
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                for row in reader:
                    y, m = int(row["year"]), int(row["month"])
                    tr = sp500.get((y, m), "")
                    if not tr:
                        missing += 1
                    rows.append((y, m, row["UNRATE"], row["USREC"],
                                 row.get("FEDFUNDS", ""), tr))

            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["year", "month", "UNRATE", "USREC", "FEDFUNDS", "SP500_TR"])
                for y, m, unrate, usrec, fedfunds, tr in rows:
                    writer.writerow([y, m, unrate, usrec, fedfunds, tr])
            total += 1
            print(f"  {label}/{fname}: {len(rows)} rows")

    print(f"\nWrote SP500_TR to {total} cycle CSVs.")
    if missing:
        print(f"WARNING: {missing} rows had no matching SP500_TR (left empty).")


if __name__ == "__main__":
    main()
