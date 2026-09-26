"""Clean raw scraped fights into data/modified_fight_details.csv.

A faithful port of the repo's modify_fights.py, which hardcodes Windows path
separators ('data\\fight_details_date.csv') and so silently writes a file with a
backslash in its name on macOS/Linux instead of the intended one.

Despite the module name this is a full recompute, not a delta: the ratio and
time conversions look at whole columns, so the whole file is rebuilt each run.
That matches how the current dataset was produced — verified byte-identical
against the previously stored modified_fight_details.csv.

Known upstream quirk, preserved deliberately: the row-deletion loop marks the
row AFTER a missing-Winner row (a draw or no-contest) for deletion rather than
that row itself, so each draw costs the following bout. Changing it would make
newly built features inconsistent with every model trained so far.
"""
import argparse
import os
import shutil
import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "fight_details_date.csv")
DST = os.path.join(ROOT, "data", "modified_fight_details.csv")

# ufcstats fighter ids are hex strings; left to inference, one like "1234e567..." reads as a float
ID_COLUMNS = ["Red Fighter ID", "Blue Fighter ID"]

DROP_COLUMNS = ["Red Sig. str. %", "Red Td %", "Blue Sig. str. %", "Blue Td %",
                "Red Sig. str", "Blue Sig. str", "Red Sig. str%", "Blue Sig. str%"]


def convert_ratio(value):
    """'12 of 30' -> (12, 0.4); anything else passes through unchanged."""
    if isinstance(value, str) and " of " in value:
        x, y = value.split(" of ")
        x, y = int(x), int(y)
        return x, x / y if y != 0 else 0
    return value, value


def time_to_minutes(value):
    """'4:30' -> 4.5 minutes."""
    if isinstance(value, str) and ":" in value:
        minutes, seconds = value.split(":")
        return (int(minutes) * 60 + int(seconds)) / 60
    return value


def build(src=SRC, dst=DST, backup=True, log=print):
    df = pd.read_csv(src, low_memory=False, dtype={c: str for c in ID_COLUMNS})
    log(f"read {len(df)} raw rows")

    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, str) and " of " in x).any():
            df[col], pct = zip(*df[col].apply(convert_ratio))
            df[f"{col}%"] = pct
        if df[col].apply(lambda x: isinstance(x, str) and ":" in x).any():
            df[col] = df[col].apply(time_to_minutes)

    # Drop draws/no-contests: rows with no winner.
    df = df[~(df["Winner"].isna() | (df["Winner"].astype(str).str.strip() == ""))]
    df.reset_index(drop=True, inplace=True)

    df = df.drop(columns=DROP_COLUMNS)
    df = df[~df["Title"].str.contains("Women", na=False)]
    df = df[~df["Title"].str.contains("Open", na=False)]

    if backup and os.path.exists(dst):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(dst, f"{dst}.bak-{stamp}")

    df.to_csv(dst, index=False)
    log(f"wrote {dst}: {len(df)} rows, {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--no-backup", action="store_true")
    a = ap.parse_args()
    build(a.src, a.dst, backup=not a.no_backup)
