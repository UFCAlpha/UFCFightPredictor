import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scrapers"))

import update_fighters as uf  # noqa: E402


def make_db(rows):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE fighter (id INTEGER PRIMARY KEY, name VARCHAR, "
                "Height VARCHAR, Weight VARCHAR, Reach VARCHAR, Stance VARCHAR, DOB VARCHAR, "
                "ufcstats_id VARCHAR)")
    con.executemany("INSERT INTO fighter VALUES (?,?,?,?,?,?,?,?)", rows)
    return con


def test_gap_updates_fills_only_missing_fields_with_real_values():
    current = {"Height": "--", "Weight": "185 lbs.", "Reach": "", "Stance": None, "DOB": "--"}
    fresh = {"Height": "6' 0\"", "Weight": "170 lbs.", "Reach": "75\"", "Stance": "--", "DOB": ""}
    # Weight is known, so it is never overwritten; Stance/DOB are still blank upstream.
    assert uf.gap_updates(current, fresh) == {"Height": "6' 0\"", "Reach": "75\""}


def test_refresh_candidates_uses_index_and_recency():
    rows = [
        {"fid": "rb", "Height": "--", "Weight": "185 lbs.", "Reach": "--", "Stance": "Orthodox", "DOB": "Jul 04, 1990"},
        {"fid": "ot", "Height": "--", "Weight": "--", "Reach": "--", "Stance": "--", "DOB": "--"},
        {"fid": "rn", "Height": "5' 9\"", "Weight": "155 lbs.", "Reach": "70\"", "Stance": "Orthodox", "DOB": "--"},
        {"fid": "cp", "Height": "5' 9\"", "Weight": "155 lbs.", "Reach": "70\"", "Stance": "Orthodox", "DOB": "Jan 01, 1990"},
    ]
    index_bios = {
        "rb": {"Height": "6' 0\"", "Weight": "185 lbs.", "Reach": "75.0\"", "Stance": "Orthodox"},
        "ot": {"Height": "--", "Weight": "--", "Reach": "--", "Stance": ""},
        "rn": {"Height": "5' 9\"", "Weight": "155 lbs.", "Reach": "70.0\"", "Stance": "Orthodox"},
        "cp": {"Height": "5' 9\"", "Weight": "155 lbs.", "Reach": "70.0\"", "Stance": "Orthodox"},
    }
    # rb: index now lists what we lack. rn: DOB is not in the index, but he fought
    # recently, so his page is worth a look. ot: nothing new anywhere.
    assert uf.refresh_candidates(rows, index_bios, recent_ids={"rn"}) == ["rb", "rn"]


def test_refresh_incomplete_updates_db_and_skips_unlinked_rows():
    con = make_db([
        (1, "Robert Bryczek", "--", "185 lbs.", "--", "Orthodox", "Jul 04, 1990", "rb"),
        (2, "Old Timer", "--", "--", "--", "--", "--", None),
    ])
    index_bios = {"rb": {"Height": "6' 0\"", "Weight": "185 lbs.", "Reach": "75.0\"", "Stance": "Orthodox"}}
    pages = {"rb": {"name": "Robert Bryczek", "Height": "6' 0\"", "Weight": "185 lbs.",
                    "Reach": "75\"", "Stance": "Orthodox", "DOB": "Jul 04, 1990"}}
    fetched = []

    def fetch(fid):
        fetched.append(fid)
        return pages[fid]

    n = uf.refresh_incomplete(con, index_bios, recent_ids=set(), fetch=fetch, log=lambda m: None)

    assert n == 1
    assert fetched == ["rb"]
    assert con.execute("SELECT Height, Reach FROM fighter WHERE id=1").fetchone() == ("6' 0\"", "75\"")
