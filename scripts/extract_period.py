"""Extract a continuous multi-year macro series into a single cycle-style CSV.

Reads the four raw FRED / market sources and merges them on (year, month):

    raw_data/UNRATE.csv        -> UNRATE  (percent, e.g. 9.8)
    raw_data/USREC.csv         -> USREC   (0/1 recession flag)
    raw_data/FEDFUNDS.csv      -> FEDFUNDS (percent, e.g. 0.11)
    raw_data/stock_monthly.csv -> SP500_TR (decimal, e.g. -0.202), CPI (index)

Output has the same column layout as the per-cycle CSVs consumed by
MacroLayer (``year,month,UNRATE,USREC,FEDFUNDS,SP500_TR,CPI``), so the result
can be used directly as a long-horizon macro file.

Usage:
    python scripts/extract_period.py [start_year] [end_year]

Defaults to 1985..2025 (inclusive), i.e. 41 years / 492 months.
"""

import csv
import os
import sys

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_data")


def read_date_col(path: str, col: str):
    """Read a FRED-style CSV (observation_date, <col>) into {(year, month): value_str}."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            date = (row.get("observation_date") or "").strip()
            val = (row.get(col) or "").strip()
            if not date or not val or val == ".":
                continue
            y, m = int(date[:4]), int(date[5:7])
            out[(y, m)] = val
    return out


def read_ym_col(path: str, col: str):
    """Read a CSV with integer year,month columns into {(year, month): value_str}."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y = (row.get("year") or "").strip()
            m = (row.get("month") or "").strip()
            val = (row.get(col) or "").strip()
            if not y or not m or not val:
                continue
            out[(int(y), int(m))] = val
    return out


def forward_fill(months, src):
    """Carry the last-known value forward over ``months`` (sorted (year, month) list)."""
    out = {}
    imputed = []
    last = ""
    for k in months:
        v = src.get(k, "")
        if v == "":
            if last != "":
                out[k] = last
                imputed.append(k)
            else:
                out[k] = ""  # leading gap with no prior value — nothing to carry
        else:
            last = v
            out[k] = v
    return out, imputed


def main() -> None:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1985
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2025

    months = [(y, m) for y in range(start, end + 1) for m in range(1, 13)]

    cols = {
        "UNRATE": read_date_col(os.path.join(RAW, "UNRATE.csv"), "UNRATE"),
        "USREC": read_date_col(os.path.join(RAW, "USREC.csv"), "USREC"),
        "FEDFUNDS": read_date_col(os.path.join(RAW, "FEDFUNDS.csv"), "FEDFUNDS"),
        "SP500_TR": read_ym_col(os.path.join(RAW, "stock_monthly.csv"), "SP500_TR"),
        "CPI": read_ym_col(os.path.join(RAW, "stock_monthly.csv"), "CPI"),
    }

    filled = {}
    imputations = []
    for col, src in cols.items():
        filled[col], imp = forward_fill(months, src)
        for k in imp:
            imputations.append((col, k))

    out_path = os.path.join(RAW, f"{start}_{end}.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "month", "UNRATE", "USREC", "FEDFUNDS", "SP500_TR", "CPI"])
        for y, m in months:
            w.writerow([y, m, filled["UNRATE"][(y, m)], filled["USREC"][(y, m)],
                        filled["FEDFUNDS"][(y, m)], filled["SP500_TR"][(y, m)],
                        filled["CPI"][(y, m)]])

    print(f"Wrote {out_path}: {len(months)} months ({start}-01 .. {end}-12).")
    if imputations:
        print(f"Forward-filled {len(imputations)} missing cell(s): {imputations}")
    else:
        print("All four columns fully populated — no gaps.")


if __name__ == "__main__":
    main()
