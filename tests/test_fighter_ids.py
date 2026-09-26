"""Fighters are identified by their ufcstats id; names stay as the display/join value."""
import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scrapers"))

import fighter_ids  # noqa: E402
import scrape_incremental  # noqa: E402
import update_fighters as uf  # noqa: E402
from ufcnet import ScrapeError  # noqa: E402

FD = "http://ufcstats.com/fighter-details/"


def fight_page(red, blue):
    """Minimal ufcstats fight page: red corner first in the stats table."""
    def cell(a, b):
        return (f'<td class="b-fight-details__table-col"><p class="b-fight-details__table-text">{a}</p>'
                f'<p class="b-fight-details__table-text">{b}</p></td>')
    names = ('<td class="b-fight-details__table-col">'
             f'<p class="b-fight-details__table-text"><a href="{FD}{red[1]}">{red[0]}</a></p>'
             f'<p class="b-fight-details__table-text"><a href="{FD}{blue[1]}">{blue[0]}</a></p></td>')
    return f"""<html><body>
    <div class="b-fight-details__person"><i class="b-fight-details__person-status">W</i>
      <h3 class="b-fight-details__person-name"><a href="{FD}{red[1]}">{red[0]}</a></h3></div>
    <div class="b-fight-details__person"><i class="b-fight-details__person-status">L</i>
      <h3 class="b-fight-details__person-name"><a href="{FD}{blue[1]}">{blue[0]}</a></h3></div>
    <table><thead><tr><th>Fighter</th><th>KD</th></tr></thead>
    <tbody class="b-fight-details__table-body"><tr class="b-fight-details__table-row">
      {names}{cell(1, 0)}</tr></tbody></table></body></html>"""


def event_page(bouts):
    rows = "".join(
        '<tr class="b-fight-details__table-row"><td><p>'
        f'<a class="b-link b-link_style_black" href="{FD}{ia}">{a}</a></p><p>'
        f'<a class="b-link b-link_style_black" href="{FD}{ib}">{b}</a></p></td></tr>'
        for a, ia, b, ib in bouts)
    return f'<html><body><table><tbody class="b-fight-details__table-body">{rows}</tbody></table></body></html>'


def test_fighter_id_from_href():
    assert fighter_ids.fighter_id(FD + "12ebd7d157e91701") == "12ebd7d157e91701"
    assert fighter_ids.fighter_id(FD + "12ebd7d157e91701/") == "12ebd7d157e91701"
    assert fighter_ids.fighter_id("http://ufcstats.com/event-details/abc") is None
    assert fighter_ids.fighter_id(None) is None


def test_parse_fight_captures_corner_ids():
    fight = scrape_incremental.parse_fight(
        fight_page(("Bruno Silva", "12ebd7d157e91701"), ("Marc-Andre Barriault", "8e9eb3fc86db0f7d")))
    assert fight["red_id"] == "12ebd7d157e91701"
    assert fight["blue_id"] == "8e9eb3fc86db0f7d"


def test_to_row_appends_ids_when_the_file_has_id_columns():
    fight = scrape_incremental.parse_fight(fight_page(("A B", "aaa"), ("C D", "ccc")))
    base = ["x"] * (11 + 2 * len(scrape_incremental.STAT_KEYS))
    row = scrape_incremental.to_row(fight, "May 10, 2025", base + list(fighter_ids.ID_COLUMNS))
    assert row[-2:] == ["aaa", "ccc"]
    # a file that predates the id columns still gets rows of its own width
    assert len(scrape_incremental.to_row(fight, "May 10, 2025", base)) == len(base)


def test_event_bouts_reads_names_and_ids():
    html = event_page([("Bruno Silva", "12eb", "Marc-Andre Barriault", "8e9e")])
    assert fighter_ids.event_bouts(html) == [("Bruno Silva", "12eb", "Marc-Andre Barriault", "8e9e")]


def test_assign_ids_separates_fighters_who_share_a_name():
    rows = [
        {"Date": "May 10, 2025", "Red Fighter": "Marc-Andre Barriault", "Blue Fighter": "Bruno Silva"},
        {"Date": "May 10, 2025", "Red Fighter": "Bruno Silva", "Blue Fighter": "Tagir Ulanbekov"},
    ]
    bouts = {"May 10, 2025": [("Bruno Silva", "mw", "Marc-Andre Barriault", "mab"),
                              ("Tagir Ulanbekov", "tu", "Bruno Silva", "fw")]}
    assert fighter_ids.assign_ids(rows, bouts) == [("mab", "mw"), ("fw", "tu")]


def test_assign_ids_follows_a_rename_through_the_other_corner():
    # the file keeps the name at scrape time; ufcstats now says "King Green"
    rows = [{"Date": "April 22, 2023", "Red Fighter": "Bobby Green", "Blue Fighter": "Jared Gordon"}]
    bouts = {"April 22, 2023": [("King Green", "kg", "Jared Gordon", "jg"),
                                ("King Green", "kg", "Jared Gordon", "jg"),  # listed twice
                                ("Someone", "s1", "Else", "e1")]}
    assert fighter_ids.assign_ids(rows, bouts) == [("kg", "jg")]


def test_assign_ids_will_not_follow_a_rename_when_the_known_name_is_ambiguous():
    rows = [{"Date": "May 10, 2025", "Red Fighter": "Bruno Silva", "Blue Fighter": "Renamed Guy"}]
    bouts = {"May 10, 2025": [("Bruno Silva", "mw", "New Name", "nn"),
                              ("Bruno Silva", "fw", "Other", "ot")]}
    with pytest.raises(ScrapeError, match="Renamed Guy"):
        fighter_ids.assign_ids(rows, bouts)


def test_assign_ids_refuses_to_guess():
    rows = [{"Date": "May 10, 2025", "Red Fighter": "Nobody", "Blue Fighter": "Else"}]
    with pytest.raises(ScrapeError, match="Nobody"):
        fighter_ids.assign_ids(rows, {"May 10, 2025": []})


def make_db(rows):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE fighter (id INTEGER PRIMARY KEY, name VARCHAR, record VARCHAR, "
                "SLpM FLOAT, Str_Acc VARCHAR, SApM FLOAT, Str_Def VARCHAR, TD_Avg FLOAT, "
                "TD_Acc VARCHAR, TD_Def VARCHAR, Sub_Avg FLOAT, Height VARCHAR, Weight VARCHAR, "
                "Reach VARCHAR, Stance VARCHAR, DOB VARCHAR, ufcstats_id VARCHAR)")
    for r in rows:
        con.execute("INSERT INTO fighter (id, name, Height, Weight, DOB, ufcstats_id) VALUES (?,?,?,?,?,?)", r)
    return con


def test_link_ids_matches_rows_and_inserts_the_unseen_namesake():
    con = make_db([
        (1, "Bruno Silva", "5' 4\"", "125 lbs.", "Mar 16, 1990", None),
        (2, "Bruno Silva", "6' 0\"", "185 lbs.", "Jul 13, 1989", None),
        (3, "Victor Valenzuela", "5' 10\"", "155 lbs.", "--", None),
        (4, "Jon Jones", "6' 4\"", "205 lbs.", "Jul 19, 1987", None),
    ])
    ids_by_name = {"Bruno Silva": {"fw", "mw"}, "Victor Valenzuela": {"vv155", "vv170"}, "Jon Jones": {"jj"}}
    index_bios = {"fw": {"Height": "5' 4\"", "Weight": "125 lbs."},
                  "mw": {"Height": "6' 0\"", "Weight": "185 lbs."},
                  "vv155": {"Height": "5' 10\"", "Weight": "155 lbs."},
                  "vv170": {"Height": "5' 9\"", "Weight": "170 lbs."},
                  "jj": {"Height": "6' 4\"", "Weight": "205 lbs."}}
    pages = {"vv170": {"name": "Victor Valenzuela", "record": "14-4-0", "Height": "5' 9\"",
                       "Weight": "170 lbs.", "DOB": "Jan 01, 1995"}}

    uf.link_ids(con, ids_by_name, index_bios, fetch=lambda fid: pages[fid], log=lambda m: None)

    got = dict(con.execute("SELECT id, ufcstats_id FROM fighter WHERE id <= 4"))
    assert got == {1: "fw", 2: "mw", 3: "vv155", 4: "jj"}
    new = con.execute("SELECT name, Weight, ufcstats_id FROM fighter WHERE id > 4").fetchall()
    assert new == [("Victor Valenzuela", "170 lbs.", "vv170")]


def test_refresh_incomplete_reads_duplicate_names_by_id():
    con = make_db([
        (1, "Mike Davis", "--", "--", "--", "c866"),
        (2, "Mike Davis", "6' 0\"", "155 lbs.", "--", "fb3e"),
    ])
    for col in ("Reach", "Stance"):
        con.execute(f"UPDATE fighter SET {col}='Orthodox' WHERE id=2" if col == "Stance"
                    else f"UPDATE fighter SET {col}='72\"' WHERE id=2")
    index_bios = {"c866": {"Height": "--", "Weight": "--", "Reach": "--", "Stance": ""},
                  "fb3e": {"Height": "6' 0\"", "Weight": "155 lbs.", "Reach": "72.0\"", "Stance": "Orthodox"}}
    fetched = []

    def fetch(fid):
        fetched.append(fid)
        return {"DOB": "Oct 07, 1992"}

    n = uf.refresh_incomplete(con, index_bios, recent_ids={"fb3e"}, fetch=fetch, log=lambda m: None)
    assert n == 1 and fetched == ["fb3e"]
    assert con.execute("SELECT DOB FROM fighter WHERE id=2").fetchone() == ("Oct 07, 1992",)


def test_build_features_looks_up_namesakes_by_id(monkeypatch, tmp_path):
    sys.path.insert(0, ROOT)
    import predict_event
    import predict_fights_alpha as pfa

    # two Bruno Silvas in the stats: the veteran (mw) and a one-fight flyweight (fw)
    monkeypatch.setattr(predict_event, "_known_fighters", lambda: {"mw": 10, "fw": 1, "Opp": 5})
    monkeypatch.setattr(pfa, "output_csv_filename", str(tmp_path / "features.csv"))
    calls = []

    def extract(a, b, fid=None, oid=None):
        calls.append((a, b, fid, oid))
        with open(pfa.output_csv_filename, "a") as fh:
            fh.write("x\n")

    monkeypatch.setattr(pfa, "extract_fighter_stats", extract)
    bouts = [("Bruno Silva", "Opp"), ("Bruno Silva", "Other")]
    # Opp's id is not in the stats (no id-bearing rows), so he is found by name
    ids = {bouts[0]: ("mw", "opp-id"), bouts[1]: ("fw", "oth-id")}

    written, skipped = predict_event.build_features(bouts, ids)

    assert calls == [("Bruno Silva", "Opp", "mw", None), ("Opp", "Bruno Silva", None, "mw")]
    assert written == 2
    assert skipped == [(bouts[1], "Bruno Silva: only 1 prior fight; Other: no UFC history in the dataset")]
