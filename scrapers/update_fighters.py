"""Keep the fighter database in step with the fight data.

Each fighter row carries its ufcstats id (ufcstats_id). A run
  1. links rows to the ids the fight data references; a name shared by two
     fighters is resolved by height/weight against the ufcstats fighter list,
  2. inserts every referenced fighter the database has no row for, by id, so a
     debutant who shares a veteran's name gets a row of his own, and
  3. refills bio fields (height, reach, DOB...) that were blank when a fighter was
     added. ufcstats fills those in later; process_fights_alpha.py drops every fight
     where either fighter lacks a DOB or height, so a stale "--" costs real fights.

process_fights_alpha.py raises KeyError on any fighter the database has never
seen, so this has to run after any scrape that introduces debutants.
"""
import argparse
import csv
import os
import re
import sqlite3
import sys
from datetime import datetime

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ufcnet
from fighter_ids import ID_COLUMNS, fighter_id

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "instance", "detailedfighters.db")
FIGHTS = os.path.join(ROOT, "data", "fight_details_date.csv")
INDEX = "http://ufcstats.com/statistics/fighters?char={c}&page=all"
DETAIL = "http://ufcstats.com/fighter-details/{}"
RECENT_YEARS = 3

KEYMAP = {"SLpM": "SLpM", "Str. Acc.": "Str_Acc", "SApM": "SApM",
          "Str. Def": "Str_Def", "TD Avg.": "TD_Avg", "TD Acc.": "TD_Acc",
          "TD Def.": "TD_Def", "Sub. Avg.": "Sub_Avg", "Height": "Height",
          "Weight": "Weight", "Reach": "Reach", "Stance": "Stance", "DOB": "DOB"}
FLOATS = {"SLpM", "SApM", "TD_Avg", "Sub_Avg"}
COLS = ["name", "record", "SLpM", "Str_Acc", "SApM", "Str_Def", "TD_Avg",
        "TD_Acc", "TD_Def", "Sub_Avg", "Height", "Weight", "Reach", "Stance", "DOB"]
BIO = ["Height", "Weight", "Reach", "Stance", "DOB"]
# The a-z index lists every bio field except DOB (reach as 75.0" rather than the
# detail page's 75", so the index only decides whom to re-read, never what to store).
INDEX_BIO = ["Height", "Weight", "Reach", "Stance"]


def _blank(v):
    return v is None or str(v).strip() in ("", "--")


def _fight_rows(path=FIGHTS):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def ids_by_name(rows):
    """name -> {ufcstats ids} over every fighter the fight rows reference."""
    out = {}
    for row in rows:
        for side, col in zip(("Red Fighter", "Blue Fighter"), ID_COLUMNS):
            name, fid = (row.get(side) or "").strip(), (row.get(col) or "").strip()
            if name and fid:
                out.setdefault(name, set()).add(fid)
    return out


def unidentified_names(rows):
    """Names on rows that predate the id columns (legacy; matched by name)."""
    names = set()
    for row in rows:
        for side, col in zip(("Red Fighter", "Blue Fighter"), ID_COLUMNS):
            name = (row.get(side) or "").strip()
            if name and not (row.get(col) or "").strip():
                names.add(name)
    return names


def ensure_id_column(con):
    cols = [r[1] for r in con.execute("PRAGMA table_info(fighter)")]
    if "ufcstats_id" not in cols:
        con.execute("ALTER TABLE fighter ADD COLUMN ufcstats_id VARCHAR")
        con.commit()


def _parse(html):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h2", class_="b-content__title")
    if not title:
        return None
    txt = title.get_text()
    name = re.search(r"^([^\n]+)", txt.strip()).group(1).strip()
    rec = (re.search(r"Record: (\d+-\d+-\d+ \(.*\))", txt)
           or re.search(r"Record: (\d+-\d+-\d+)", txt))
    stats = {"name": name, "record": rec.group(1).strip() if rec else None}
    for item in soup.find_all("li", class_="b-list__box-list-item"):
        lab = item.find("i", class_="b-list__box-item-title")
        if not lab:
            continue
        label = lab.get_text().strip().rstrip(":")
        if label == "STANCE":
            label = "Stance"
        if label not in KEYMAP:
            continue
        sib = lab.next_sibling
        val = sib.text.strip() if hasattr(sib, "text") else (sib.strip() if sib else "")
        stats[KEYMAP[label]] = val
    return stats


def build_name_index(session, log=print, bios=None):
    """Map "First Last" -> fighter-details URL across the a-z listing.

    If `bios` is a dict, it is filled with ufcstats id -> {name, Height, Weight, Reach, Stance}.
    """
    link_of = {}
    for c in "abcdefghijklmnopqrstuvwxyz":
        soup = BeautifulSoup(ufcnet.get(session, INDEX.format(c=c), delay=0.25), "html.parser")
        rows = soup.find_all("tr", class_="b-statistics__table-row")
        if not rows:
            raise ufcnet.ScrapeError(f"fighter index '{c}' parsed to zero rows")
        for row in rows:
            tds = row.find_all("td")
            links = [a for a in row.find_all("a", href=True) if "fighter-details" in a["href"]]
            if len(tds) < 2 or not links:
                continue
            full = (tds[0].get_text(strip=True) + " " + tds[1].get_text(strip=True)).strip()
            link_of.setdefault(full, links[0]["href"])
            fid = fighter_id(links[0]["href"])
            if bios is not None and fid and len(tds) >= 7:
                bios[fid] = dict(zip(INDEX_BIO, (tds[i].get_text(strip=True) for i in (3, 4, 5, 6))),
                                 name=full)
    log(f"fighter index built: {len(link_of)} names")
    return link_of


def recent_ids(rows, years=RECENT_YEARS):
    """ufcstats ids of fighters with a bout in the last `years` years of the fight data."""
    dated = []
    for row in rows:
        try:
            dated.append((datetime.strptime(row["Date"].strip(), "%B %d, %Y"), row))
        except (KeyError, ValueError):
            continue
    if not dated:
        return set()
    newest = max(d for d, _ in dated)
    cutoff = newest.replace(year=newest.year - years)
    return {(row.get(c) or "").strip() for d, row in dated if d >= cutoff for c in ID_COLUMNS} - {""}


def _values(stats):
    values = []
    for col in COLS:
        v = stats.get(col)
        if col in FLOATS:
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = None
        values.append(v)
    return values


def _insert(con, stats, fid=None):
    next_id = con.execute("SELECT COALESCE(MAX(id),0) FROM fighter").fetchone()[0] + 1
    con.execute(f"INSERT INTO fighter (id,{','.join(COLS)},ufcstats_id) "
                f"VALUES ({','.join(['?'] * (len(COLS) + 2))})",
                [next_id] + _values(stats) + [fid])


def _same_body(row, bio, fields):
    return all(not _blank(row.get(k)) and (row.get(k) or "").strip() == (bio.get(k) or "").strip()
               for k in fields)


def link_ids(con, ids_by_name, index_bios, fetch, log=print):
    """Attach ufcstats ids to fighter rows, inserting a row for every id left over.

    A name with one unlinked row and one unlinked id, and only one fighter of that
    name on ufcstats, links directly. Otherwise each id goes to the row whose
    height and weight (or failing that, height alone) match its fighter-list entry.
    fetch(fid) returns the parsed fighter page for an id with no row.
    """
    names_on_site = {}
    for fid, bio in index_bios.items():
        names_on_site[bio.get("name")] = names_on_site.get(bio.get("name"), 0) + 1
    known = {r[0] for r in con.execute("SELECT ufcstats_id FROM fighter WHERE ufcstats_id IS NOT NULL")}
    linked = inserted = 0
    for name in sorted(ids_by_name):
        ids = sorted(ids_by_name[name] - known)
        if not ids:
            continue
        rows = [dict(zip(["id", "Height", "Weight"], r)) for r in con.execute(
            "SELECT id, Height, Weight FROM fighter WHERE trim(name)=? AND ufcstats_id IS NULL "
            "ORDER BY id", (name,))]
        pairs = []
        if len(rows) == 1 and len(ids) == 1 and names_on_site.get(name, 1) <= 1:
            pairs = [(rows[0], ids[0])]
        else:
            free_rows, free_ids = list(rows), list(ids)
            for fields in (("Height", "Weight"), ("Height",)):
                for fid in list(free_ids):
                    hits = [r for r in free_rows if _same_body(r, index_bios.get(fid, {}), fields)]
                    rivals = [f for f in free_ids if f != fid and
                              any(_same_body(r, index_bios.get(f, {}), fields) for r in hits)]
                    if len(hits) == 1 and not rivals:
                        pairs.append((hits[0], fid))
                        free_rows.remove(hits[0])
                        free_ids.remove(fid)
        for row, fid in pairs:
            con.execute("UPDATE fighter SET ufcstats_id=? WHERE id=?", (fid, row["id"]))
            known.add(fid)
            linked += 1
            if len(ids_by_name[name]) > 1 or names_on_site.get(name, 1) > 1:
                log(f"  linked {name} (row {row['id']}) -> {fid}")
        for fid in sorted(set(ids) - known):
            stats = fetch(fid)
            if not stats:
                log(f"  !! no fighter page for {name} ({fid})")
                continue
            _insert(con, stats, fid)
            known.add(fid)
            inserted += 1
            log(f"  inserted {name} ({fid})")
    con.commit()
    log(f"linked {linked} fighter rows to ufcstats ids; inserted {inserted} new fighters")
    return linked, inserted


def gap_updates(current, fresh):
    """Bio fields that are blank in `current` and have a real value in `fresh`."""
    return {k: fresh[k] for k in BIO if _blank(current.get(k)) and not _blank(fresh.get(k))}


def refresh_candidates(rows, index_bios, recent_ids):
    """Ids worth re-reading: the fighter list now shows a field we lack, or DOB is
    missing for someone who fought recently (the list has no DOB column)."""
    out = []
    for r in rows:
        idx = index_bios.get(r["fid"])
        if idx is None:
            continue
        if any(_blank(r.get(k)) and not _blank(idx.get(k)) for k in INDEX_BIO) or \
                (_blank(r.get("DOB")) and r["fid"] in recent_ids):
            out.append(r["fid"])
    return sorted(out)


def refresh_incomplete(con, index_bios, recent_ids, fetch, log=print):
    """Fill blank bio fields from each candidate's fighter page. Returns rows updated.

    Only rows linked to a ufcstats id are considered: for a name shared by two
    fighters, the id is the only way to know which page belongs to the row.
    fetch(fid) returns the parsed fighter page.
    """
    rows = [dict(zip(["id", "fid"] + BIO, r)) for r in con.execute(
        f"SELECT id, ufcstats_id, {', '.join(BIO)} FROM fighter WHERE ufcstats_id IS NOT NULL")]
    by_fid = {r["fid"]: r for r in rows if any(_blank(r.get(k)) for k in BIO)}
    todo = refresh_candidates(list(by_fid.values()), index_bios, recent_ids)
    log(f"fighters with incomplete bios to re-read: {len(todo)}")
    updated = 0
    for fid in todo:
        fresh = fetch(fid)
        if not fresh:
            continue
        changes = gap_updates(by_fid[fid], fresh)
        if not changes:
            continue
        con.execute(f"UPDATE fighter SET {', '.join(f'{k}=?' for k in changes)} WHERE id=?",
                    list(changes.values()) + [by_fid[fid]["id"]])
        updated += 1
        log(f"  filled {fresh.get('name') or fid}: {', '.join(f'{k}={v}' for k, v in changes.items())}")
    con.commit()
    log(f"filled bio gaps for {updated} fighters")
    return updated


def run(dry_run=False, log=print):
    rows = _fight_rows()
    by_name = ids_by_name(rows)
    con = sqlite3.connect(DB)
    try:
        ensure_id_column(con)
        known = {r[0] for r in con.execute("SELECT ufcstats_id FROM fighter WHERE ufcstats_id IS NOT NULL")}
        have = {r[0].strip() for r in con.execute("SELECT name FROM fighter")}
    finally:
        con.close()
    missing = sorted(unidentified_names(rows) - have)
    unlinked = sum(len(ids - known) for ids in by_name.values())
    log(f"fighter ids not yet in database: {unlinked}; unidentified names missing: {len(missing)}")
    if dry_run:
        return 0

    session = ufcnet.new_session()
    index_bios = {}
    link_of = build_name_index(session, log=log, bios=index_bios)

    def page(url):
        return _parse(ufcnet.get(session, url, delay=0.25))

    con = sqlite3.connect(DB)
    try:
        added, unmatched = 0, []
        for name in missing:  # rows without ids, matched by name as before
            stats = page(link_of[name]) if name in link_of else None
            if not stats:
                unmatched.append(name)
                continue
            _insert(con, stats, fighter_id(link_of[name]))
            added += 1
        con.commit()
        if missing:
            log(f"inserted {added} fighters by name; {len(unmatched)} had no listing")
        if unmatched:
            log(f"  unmatched: {unmatched[:20]}")

        _, inserted = link_ids(con, by_name, index_bios,
                               fetch=lambda fid: page(DETAIL.format(fid)), log=log)
        refresh_incomplete(con, index_bios, recent_ids(rows),
                           fetch=lambda fid: page(DETAIL.format(fid)), log=log)
    finally:
        con.close()
    return added + inserted


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report what is missing, change nothing")
    a = ap.parse_args()
    run(dry_run=a.dry_run)
