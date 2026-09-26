"""One-time backfill: add Red/Blue Fighter ID columns to data/fight_details_date.csv.

Reads every ufcstats event page that has fights in the file (about 790 pages),
matches each stored fight to its bout by date and name pair, and writes the two
fighter ids. Refuses to write anything if any fight cannot be matched. Then links
the fighter database to those ids (update_fighters.run).

Rows without corner names (21 bouts from 1994-98 that ufcstats has no stats
tables for; the file stores them as 11-field rows) are written back unchanged.

Rerunning is safe: a file that already has the columns is left alone unless
--force is given.
"""
import argparse
import csv
import datetime
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ufcnet
import update_fighters
from fighter_ids import ID_COLUMNS, assign_ids, event_bouts
from scrape_incremental import CSV_PATH, fetch_event_index


def _get(session, url, cache_dir):
    path = os.path.join(cache_dir, url.rstrip("/").rsplit("/", 1)[-1] + ".html") if cache_dir else None
    if path and os.path.exists(path):
        with open(path) as fh:
            return fh.read()
    html = ufcnet.get(session, url, delay=0.25)
    if path:
        with open(path, "w") as fh:
            fh.write(html)
    return html


def run(force=False, cache_dir=None, log=print):
    with open(CSV_PATH, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        raw = list(reader)
    has_ids = all(c in header for c in ID_COLUMNS)
    if has_ids and not force:
        log("fight file already has fighter id columns; nothing to do (use --force to redo)")
        return 0
    base = [c for c in header if c not in ID_COLUMNS]
    width = len(base)
    named = [i for i, r in enumerate(raw) if len(r) >= width]
    rows = [dict(zip(base, raw[i][:width])) for i in named]

    dates = {r["Date"].strip() for r in rows}
    session = ufcnet.new_session()
    bouts_by_date = {}
    events = [(d, url) for d, url, _ in fetch_event_index(session)
              if d.strftime("%B %d, %Y") in dates]
    log(f"reading {len(events)} event pages")
    for i, (d, url) in enumerate(events, 1):
        bouts_by_date.setdefault(d.strftime("%B %d, %Y"), []).extend(
            event_bouts(_get(session, url, cache_dir)))
        if i % 100 == 0:
            log(f"  {i}/{len(events)}")

    ids = dict(zip(named, assign_ids(rows, bouts_by_date)))  # raises before anything is written

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(CSV_PATH, f"{CSV_PATH}.bak-{stamp}")
    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(base + list(ID_COLUMNS))
        for i, r in enumerate(raw):
            writer.writerow(r[:width] + list(ids[i]) if i in ids else r)
    log(f"wrote fighter ids for {len(ids)} fights")

    update_fighters.run(log=lambda m: log(f"  {m}"))
    return len(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="recompute ids even if the columns exist")
    ap.add_argument("--cache", help="directory to keep fetched event pages in (for retries)")
    a = ap.parse_args()
    if a.cache:
        os.makedirs(a.cache, exist_ok=True)
    run(force=a.force, cache_dir=a.cache)
