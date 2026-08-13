#!/usr/bin/env python3
"""Preprocess Shiller stock data for WealthSandBox.

Input: ``raw_data/stock_data.csv`` — actually an .xlsx file with 4 columns:
    Date, S&P Comp. P, Dividend D, Long Interest Rate GS10

Shiller date quirk:
    Date is a float in YYYY.MM format but Excel drops trailing zeros, so
    1961.1 means 1961-10 (October).  Parse::

        year = int(Date)
        month = round((Date - year) * 100)

Calculation:
    SP500_TR[t] = (P[t] + D[t]/12) / P[t-1] - 1     # nominal monthly total return
    GS10 carried through as-is (percentage points).

Output: ``raw_data/stock_monthly.csv`` with columns: year, month, GS10, SP500_TR
"""

import os
import sys

import pandas as pd


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(repo_root, "raw_data", "stock_data.csv")
    output_path = os.path.join(repo_root, "raw_data", "stock_monthly.csv")

    # ------------------------------------------------------------------
    # 1. Read the Excel file (misnamed .csv)
    # ------------------------------------------------------------------
    print(f"Reading: {input_path}")
    df = pd.read_excel(input_path)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # 2. Parse Shiller dates
    # ------------------------------------------------------------------
    year = df["Date"].astype(int)
    month = ((df["Date"] - year) * 100).round().astype(int)
    df["year"] = year
    df["month"] = month

    print(f"  Date range: {year.min()}-{month.min():02d} to {year.max()}-{month.max():02d}")

    # ------------------------------------------------------------------
    # 3. Compute S&P 500 total return (nominal monthly)
    # ------------------------------------------------------------------
    P = df["S&P Comp. P"].values
    D = df["Dividend D"].values

    # SP500_TR[t] = (P[t] + D[t]/12) / P[t-1] - 1
    # First row has no predecessor — NaN
    sp500_tr = [float("nan")]
    for t in range(1, len(P)):
        ret = (P[t] + D[t] / 12.0) / P[t - 1] - 1.0
        sp500_tr.append(round(ret, 8))

    df["SP500_TR"] = sp500_tr
    gs10 = df["Long Interest Rate GS10"]

    # ------------------------------------------------------------------
    # 4. Build output
    # ------------------------------------------------------------------
    out = pd.DataFrame({
        "year": year,
        "month": month,
        "GS10": gs10,
        "SP500_TR": df["SP500_TR"],
    })
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} rows to {output_path}")

    # ------------------------------------------------------------------
    # 5. Quality report
    # ------------------------------------------------------------------
    valid = out["SP500_TR"].dropna()
    print()
    print("=== Quality Report ===")
    print(f"  Rows:                {len(out)}")
    print(f"  Date range:          {out['year'].min()}-{out['month'].min():02d} to "
          f"{out['year'].max()}-{out['month'].max():02d}")
    print(f"  SP500_TR (valid):")
    print(f"    Count:             {len(valid)}")
    print(f"    Min:               {valid.min():.6f}")
    print(f"    Max:               {valid.max():.6f}")
    print(f"    Mean:              {valid.mean():.6f}")
    print(f"  Missing SP500_TR:    {out['SP500_TR'].isna().sum()}")
    print(f"  Missing GS10:        {out['GS10'].isna().sum()}")
    print("======================")


if __name__ == "__main__":
    main()
