"""Scrape only the fights that happened since the last stored one.

data/fight_details_date.csv is stored newest-first, so new bouts are prepended.
Column conventions are matched to the existing file exactly: 'Time Format' and
'Details' stay empty (the site labels them differently and the original scraper
never captured them), and draws carry an empty Winner/Loser.

If the event index parses to zero rows this raises ScrapeError rather than
reporting "0 new fights" — the failure mode that hid a 2.5 year data gap.
"""
import argparse
import csv
import datetime
import os
import shutil
import sys

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ufcnet
import update_fighters
from fighter_ids import ID_COLUMNS, fighter_id
from ufcnet import ScrapeError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "fight_details_date.csv")
EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"

STAT_KEYS = ["Fighter", "KD", "Sig. str.", "Sig. str. %", "Total str.", "Td", "Td %",
             "Sub. att", "Rev.", "Ctrl", "Sig. str", "Head", "Body", "Leg",
             "Distance", "Clinch", "Ground"]


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out.setdefault(k, v)
    return out


def latest_stored_date(path=CSV_PATH):
    """Newest event date already on disk, or None for an empty/absent file."""
    if not os.path.exists(path):
        return None
    newest = None
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                d = datetime.datetime.strptime(row["Date"].strip(), "%B %d, %Y")
            except (ValueError, AttributeError, KeyError):
                continue
            if newest is None or d > newest:
                newest = d
    return newest


def fetch_event_index(session):
    """All completed events as (date, url, name), oldest first."""
    soup = BeautifulSoup(ufcnet.get(session, EVENTS_URL), "html.parser")
    rows = soup.find_all("tr", class_="b-statistics__table-row")
    if not rows:
        raise ScrapeError(
            "event index parsed to zero rows — the page layout changed or the "
            "request was intercepted. Refusing to report this as 'no new fights'.")
    events = []
    for row in rows:
        a = row.find("a", href=True)
        span = row.find("span", class_="b-statistics__date")
        if not a or not span:
            continue
        try:
            d = datetime.datetime.strptime(span.get_text(strip=True), "%B %d, %Y")
        except ValueError:
            continue
        events.append((d, a["href"], a.get_text(strip=True)))
    if not events:
        raise ScrapeError("event index had rows but no parseable events")
    events.sort()
    return events


def parse_fight(html):
    soup = BeautifulSoup(html, "html.parser")

    info = {}
    title = soup.find("i", class_="b-fight-details__fight-title")
    info["Title"] = title.get_text(strip=True) if title else ""
    content = soup.find("div", class_="b-fight-details__content")
    if content:
        for para in content.find_all("p", class_="b-fight-details__text"):
            for item in para.find_all("i"):
                lab = item.find("i", class_="b-fight-details__label")
                if lab:
                    key = lab.get_text(strip=True).rstrip(":")
                    info[key] = item.get_text(strip=True).replace(key + ":", "").strip()

    winner = loser = None
    draw = False
    for person in soup.find_all("div", class_="b-fight-details__person"):
        status = person.find("i", class_="b-fight-details__person-status")
        name = person.find("h3", class_="b-fight-details__person-name")
        if not status or not name:
            continue
        st, nm = status.get_text().strip(), name.get_text().strip()
        if st == "W":
            winner = nm
        elif st == "L":
            loser = nm
        elif st == "D":
            draw = True  # draws keep an empty Winner/Loser, as in the stored file

    red, blue = {}, {}
    red_id = blue_id = None
    for body in soup.find_all("tbody", class_="b-fight-details__table-body"):
        if red_id is None:
            # the Fighter cell links both corners, red first, same order as the stats
            first = body.find("td", class_="b-fight-details__table-col")
            ids = [fighter_id(a.get("href")) for a in first.find_all("a")] if first else []
            ids = [i for i in ids if i]
            if len(ids) >= 2:
                red_id, blue_id = ids[:2]
        thead = body.find_previous("thead")
        if not thead:
            continue
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        d1, d2 = [], []
        for row in body.find_all("tr", class_="b-fight-details__table-row"):
            for cell in row.find_all("td", class_="b-fight-details__table-col"):
                ps = cell.find_all("p", class_="b-fight-details__table-text")
                d1.append(ps[0].get_text(strip=True) if ps else "")
                d2.append(ps[1].get_text(strip=True) if len(ps) > 1 else "")
        # First table wins each key: later per-round tables repeat the headers.
        red = _merge(red, dict(zip(headers, d1)))
        blue = _merge(blue, dict(zip(headers, d2)))

    if not red.get("Fighter"):
        raise ScrapeError("fight page had no fighter statistics table")

    return {"info": info, "winner": winner, "loser": loser, "draw": draw,
            "red": red, "blue": blue, "red_id": red_id, "blue_id": blue_id}


def to_row(fight, date_text, header):
    info = fight["info"]
    draw = fight["draw"]
    row = [
        info.get("Title", ""),
        "" if draw else (fight["winner"] or ""),
        "" if draw else (fight["loser"] or ""),
        draw,
        info.get("Method", ""),
        info.get("Round", ""),
        info.get("Time", ""),
        info.get("Time Format", ""),   # empty by design; site labels it "Time format"
        info.get("Referee", ""),
        info.get("Details", ""),       # empty by design, matches the stored file
        date_text,
    ]
    for side in ("red", "blue"):
        row += [fight[side].get(k, "") for k in STAT_KEYS]
    if all(c in header for c in ID_COLUMNS):
        if not (fight.get("red_id") and fight.get("blue_id")):
            raise ScrapeError(f"no fighter ids on the fight page for {fight['red'].get('Fighter')}")
        row += [fight["red_id"], fight["blue_id"]]
    if len(row) != len(header):
        raise ScrapeError(f"row has {len(row)} fields, file has {len(header)}")
    return row


def run(dry_run=False, full=False, skip_fighters=False, log=print):
    """Scrape new fights into data/fight_details_date.csv. Returns the count added."""
    with open(CSV_PATH, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        existing = list(reader)

    cutoff = None if full else latest_stored_date()
    log(f"latest stored fight: {cutoff.date() if cutoff else '(none — full scrape)'}")

    session = ufcnet.new_session()
    events = fetch_event_index(session)
    log(f"event index: {len(events)} completed events, newest {events[-1][0].date()}")

    todo = [e for e in events if cutoff is None or e[0] > cutoff]
    if not todo:
        log("no events newer than stored data — already up to date")
        return 0
    log(f"{len(todo)} new events to scrape ({todo[0][0].date()} .. {todo[-1][0].date()})")
    if dry_run:
        log("dry run — nothing written")
        return 0

    new_rows = []
    failed = []
    for i, (date, url, name) in enumerate(reversed(todo), 1):  # newest first
        page = BeautifulSoup(ufcnet.get(session, url), "html.parser")
        date_text = ""
        for li in page.find_all("li", class_="b-list__box-list-item"):
            if "Date:" in li.get_text():
                date_text = li.get_text().replace("Date:", "").strip()
                break
        links = list(dict.fromkeys(
            a["href"] for a in page.find_all("a", href=True) if "fight-details" in a["href"]))
        if not links:
            raise ScrapeError(f"event page parsed to zero fights: {url}")
        log(f"  ({i}/{len(todo)}) {date.date()} {name} — {len(links)} fights")
        for link in links:
            try:
                fight = parse_fight(ufcnet.get(session, link))
            except ScrapeError as exc:
                log(f"    !! skipping {link}: {exc}")
                failed.append(link)
                continue
            new_rows.append(to_row(fight, date_text or date.strftime("%B %d, %Y"), header))

    if failed:
        # Writing the event's other fights would advance the cutoff past the
        # failed ones, making them unreachable without --full. Write nothing;
        # the next scheduled run retries the whole event set.
        raise ScrapeError(
            f"{len(failed)} fight(s) failed to parse (see '!! skipping' above) — "
            "nothing written so the next run retries these events")

    if not new_rows:
        raise ScrapeError(f"{len(todo)} new events but zero fights parsed")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{CSV_PATH}.bak-{stamp}"
    shutil.copy2(CSV_PATH, backup)
    log(f"backup written: {os.path.basename(backup)}")

    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(new_rows)
        if not full:
            writer.writerows(existing)
    log(f"added {len(new_rows)} fights ({len(new_rows) + (0 if full else len(existing))} rows total)")

    if not skip_fighters:
        update_fighters.run(log=lambda m: log(f"  {m}"))

    return len(new_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report what would be scraped")
    ap.add_argument("--full", action="store_true", help="rescrape every event from scratch")
    ap.add_argument("--skip-fighters", action="store_true", help="don't update the fighter database")
    a = ap.parse_args()
    try:
        n = run(dry_run=a.dry_run, full=a.full, skip_fighters=a.skip_fighters)
    except ScrapeError as exc:
        print(f"SCRAPE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"new fights: {n}")
