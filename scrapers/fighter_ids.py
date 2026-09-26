"""ufcstats fighter ids: the identity of a fighter, where names are not unique.

Six names in the fight data belong to two different fighters (two active Bruno
Silvas, 185 lb and 125 lb, among them). Names stay the display value and the join
key for odds, bets and the site: a (fighter, opponent, date) triple is unique even
for shared names. Anything that keeps per-fighter state (ELO, streaks, bios)
keys on the id instead, stored in these two columns of the fight CSVs.
"""
import re

from bs4 import BeautifulSoup

from ufcnet import ScrapeError

ID_COLUMNS = ("Red Fighter ID", "Blue Fighter ID")
_ID = re.compile(r"/fighter-details/([0-9a-f]+)/?$")


def fighter_id(href):
    """'.../fighter-details/12ebd7d157e91701' -> '12ebd7d157e91701'; None otherwise."""
    m = _ID.search(href or "")
    return m.group(1) if m else None


def event_bouts(html):
    """[(name_a, id_a, name_b, id_b), ...] for every bout on a ufcstats event page."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("tbody", class_="b-fight-details__table-body")
    bouts = []
    for row in body.find_all("tr", class_="b-fight-details__table-row") if body else []:
        links = [a for a in row.find_all("a", class_="b-link_style_black")
                 if fighter_id(a.get("href"))]
        if len(links) >= 2:
            a, b = links[:2]
            bouts.append((a.get_text(strip=True), fighter_id(a["href"]),
                          b.get_text(strip=True), fighter_id(b["href"])))
    return bouts


def assign_ids(rows, bouts_by_date):
    """(red_id, blue_id) for each fight row, matched to its event's bout by the name pair.

    bouts_by_date maps the CSV's date text ("May 10, 2025") to event_bouts() output.
    The file keeps each name as it was when scraped, and ufcstats renames fighters
    ("Bobby Green" is now "King Green"). So when no bout has both names, a bout is
    still accepted if it is the only one that night with either name: the known
    corner fixes the bout, and the renamed corner takes the other id.
    Raises ScrapeError naming every row it cannot place rather than guessing.
    """
    out, unmatched = [], []
    for row in rows:
        red, blue = row["Red Fighter"].strip(), row["Blue Fighter"].strip()
        night = set(bouts_by_date.get(row["Date"].strip(), ()))  # a bout can be listed twice
        hits = [b for b in night if {b[0], b[2]} == {red, blue}]
        if not hits:
            hits = [b for b in night if red in (b[0], b[2]) or blue in (b[0], b[2])]
        if len(hits) != 1:
            unmatched.append(f"{row['Date']}: {red} vs {blue} ({len(hits)} matches)")
            continue
        a, id_a, b, id_b = hits[0]
        if red in (a, b):
            red_id = id_a if red == a else id_b
            blue_id = id_b if red_id == id_a else id_a
        else:
            blue_id = id_a if blue == a else id_b
            red_id = id_b if blue_id == id_a else id_a
        out.append((red_id, blue_id))
    if unmatched:
        raise ScrapeError(f"{len(unmatched)} fight(s) could not be matched to an event bout: "
                          + "; ".join(unmatched[:20]))
    return out
