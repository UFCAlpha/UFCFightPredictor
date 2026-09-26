#!/usr/bin/env python3
"""Predict an upcoming UFC card and size bets with the Kelly criterion.

Feeds data/predicted_data.json, which is what app.py's /get_predicted_data
endpoint serves, plus data/betting_predictions.csv, which betting_alpha.py
reads. Every fight is scored in both corner orientations (A-vs-B and B-vs-A),
as the rest of the pipeline expects: the two runs disagree slightly, so the
betting math averages them, then blends with the devigged market price.

    python predict_event.py                  next upcoming event
    python predict_event.py --list           show the upcoming schedule
    python predict_event.py --event <url>    a specific ufcstats event
    python predict_event.py --odds           also pull odds and size bets
"""
import argparse
import csv
import datetime
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


class _QuietLogger:
    """LightGBM re-warns about parameter aliases on every predict call."""
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): print(msg, file=sys.stderr)
    def debug(self, msg): pass


lgb.register_logger(_QuietLogger())

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scrapers"))

import betting_math
import bet_ledger
import ufcnet
from fighter_ids import fighter_id
from ufcnet import ScrapeError

UPCOMING = "http://ufcstats.com/statistics/events/upcoming?page=all"
MODEL_DIR = os.path.join(ROOT, "saved_models")
PREP_DIR = os.path.join(ROOT, "saved_preprocessing")
PRED_JSON = os.path.join(ROOT, "data", "predicted_data.json")
BET_CSV = os.path.join(ROOT, "data", "betting_predictions.csv")

# Betting strategy (see CLAUDE.md): fractional Kelly, conservative. Edge is
# measured against the DE-VIGGED market probability, which is what MIN_EDGE
# was validated on.
KELLY_FRACTION = 0.05
KELLY_MAX = 0.05
MIN_EDGE = 0.05
BLEND_W = 0.8
BANKROLL = 100.0


# ------------------------------------------------------------------ scraping

def upcoming_events(session):
    soup = BeautifulSoup(ufcnet.get(session, UPCOMING), "html.parser")
    rows = soup.find_all("tr", class_="b-statistics__table-row")
    if not rows:
        raise ScrapeError("upcoming events page parsed to zero rows")
    events = []
    for row in rows:
        a = row.find("a", href=True)
        span = row.find("span", class_="b-statistics__date")
        if not a or not span:
            continue
        try:
            when = datetime.datetime.strptime(span.get_text(strip=True), "%B %d, %Y")
        except ValueError:
            continue
        events.append((when, a["href"], a.get_text(strip=True)))
    if not events:
        raise ScrapeError("upcoming events page had rows but no parseable events")
    events.sort()
    return events


def event_card(session, url):
    """(event name, [(fighter_a, fighter_b), ...], {(a, b): (id_a, id_b)}) for an event page.

    The ids are the fighters' ufcstats ids: a name can belong to two fighters.
    """
    soup = BeautifulSoup(ufcnet.get(session, url), "html.parser")
    title = soup.find("span", class_="b-content__title-highlight")
    name = title.get_text(strip=True) if title else url

    body = soup.find("tbody", class_="b-fight-details__table-body")
    if body is None:
        raise ScrapeError(f"no fight table on {url}")
    bouts, ids = [], {}
    for row in body.find_all("tr", class_="b-fight-details__table-row"):
        links = row.find_all("a", class_="b-link_style_black")
        if len(links) >= 2:
            bout = (links[0].get_text(strip=True), links[1].get_text(strip=True))
            bouts.append(bout)
            ids[bout] = (fighter_id(links[0].get("href")), fighter_id(links[1].get("href")))
    if not bouts:
        raise ScrapeError(f"fight table on {url} parsed to zero bouts")
    return name, bouts, ids


# ------------------------------------------------------------------ features

def _known_fighters():
    """ufcstats id or name -> prior fight count, from the stats the feature builder reads."""
    path = os.path.join(ROOT, "data", "detailed_fighter_stats.csv")
    known = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            known[r.get("ID") or r["Fighter"]] = int(r["totalfights"])
    return known


def build_features(bouts, ids=None):
    """Write both orientations of every bout to data/predict_fights_alpha.csv.

    Returns (rows_written, skipped) where each skip carries the reason. The
    feature builder needs at least two prior bouts per fighter, and the
    training data excludes women's bouts entirely (modify_fights.py drops any
    Title containing "Women"), so those fighters never appear in the stats.
    `ids` maps a bout to its fighters' ufcstats ids (event_card); a fighter is
    looked up by id when the stats carry one for him, else by name.
    """
    import predict_fights_alpha as pfa

    known = _known_fighters()
    with open(pfa.output_csv_filename, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=pfa.fieldnames).writeheader()

    written, skipped = 0, []
    for a, b in bouts:
        id_a, id_b = (ids or {}).get((a, b), (None, None))
        # an id the stats don't carry (fighter with no id-bearing rows) falls back to the name
        id_a, id_b = (i if i in known else None for i in (id_a, id_b))
        reasons = []
        for who, key in ((a, id_a or a), (b, id_b or b)):
            if key not in known:
                reasons.append(f"{who}: no UFC history in the dataset")
            elif known[key] < 2:
                reasons.append(f"{who}: only {known[key]} prior fight")
        if reasons:
            skipped.append(((a, b), "; ".join(reasons)))
            continue
        for first, second, fid, oid in ((a, b, id_a, id_b), (b, a, id_b, id_a)):
            pfa.extract_fighter_stats(first, second, fid, oid)
        with open(pfa.output_csv_filename) as fh:
            after = sum(1 for _ in csv.DictReader(fh))
        if after == written:
            skipped.append(((a, b), "feature builder produced no row"))
        written = after
    return written, skipped


# -------------------------------------------------------------------- model

def load_ensemble():
    label_encoder = joblib.load(os.path.join(PREP_DIR, "label_encoder.joblib"))
    with open(os.path.join(PREP_DIR, "selected_columns.json")) as fh:
        selected = json.load(fh)
    features = [c for c in selected if c != "Result"]
    paths = sorted(glob.glob(os.path.join(MODEL_DIR, "lgbm_model_*.joblib")))
    if not paths:
        raise RuntimeError("no models in saved_models/ — run auto_retrain.py first")
    return [joblib.load(p) for p in paths], features, label_encoder


def predict_rows():
    """[(red, blue, p_red_wins), ...] for every generated orientation."""
    import predict_fights_alpha as pfa
    models, features, label_encoder = load_ensemble()

    df = pd.read_csv(pfa.output_csv_filename, low_memory=False)
    if df.empty:
        raise RuntimeError("no feature rows were generated for this card")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise RuntimeError(f"feature rows are missing model columns: {missing[:5]}")

    win_index = list(label_encoder.classes_).index("win")
    from calibration import calibrate
    probs = calibrate(np.mean([m.predict_proba(df[features]) for m in models], axis=0)[:, win_index])
    return [(df["Red Fighter"].iloc[i], df["Blue Fighter"].iloc[i], float(probs[i]))
            for i in range(len(df))]


# ------------------------------------------------------------------- betting

def _slug_candidates(event_name, when):
    """ufc.com event slugs, most likely first.

    Numbered cards live at /event/ufc-331, but can carry a sponsor prefix
    (cryptocom-ufc-331), so the site index is consulted for those. Fight Nights
    are addressed by date: /event/ufc-fight-night-september-05-2026. Note every
    slug returns HTTP 200 whether or not it exists, so a candidate is only
    accepted once its page actually yields odds.
    """
    import re, requests
    candidates = []
    number = re.search(r"UFC\s+(\d{3})", event_name)
    if number:
        candidates.append(f"ufc-{number.group(1)}")
        try:
            index = requests.get("https://www.ufc.com/events", timeout=30,
                                 headers={"User-Agent": ufcnet.UA})
            found = set(re.findall(r"/event/([a-z0-9\-]+)", index.text))
            candidates += sorted(s for s in found
                                 if s.endswith(f"ufc-{number.group(1)}") and s not in candidates)
        except requests.RequestException:
            pass
    if when is not None:
        candidates.append(f"ufc-fight-night-{when.strftime('%B-%d-%Y').lower()}")
    candidates.append(re.sub(r"[^a-z0-9]+", "-", event_name.lower()).strip("-"))
    return candidates


def _parse_odds_page(html):
    """{(lower_a, lower_b): (odds_a, odds_b)} from a ufc.com event page."""
    soup = BeautifulSoup(html, "html.parser")
    wrappers = soup.find_all(class_="c-listing-fight__odds-wrapper")
    names = []
    for div in soup.find_all("div", {"class": "c-listing-fight__corner-name"}):
        given = div.find("span", {"class": "c-listing-fight__corner-given-name"})
        family = div.find("span", {"class": "c-listing-fight__corner-family-name"})
        if given and family:
            names.append(f"{given.text.strip()} {family.text.strip()}")
        else:
            link = div.find("a")
            names.append(link.text.strip() if link else "")

    out = {}
    for i in range(0, len(names) - 1, 2):
        if i // 2 >= len(wrappers):
            break
        amounts = [e.get_text().replace("\u2212", "-").strip()
                   for e in wrappers[i // 2].find_all(class_="c-listing-fight__odds-amount")]
        if len(amounts) != 2:
            continue
        try:
            out[(names[i].lower(), names[i + 1].lower())] = (int(amounts[0]), int(amounts[1]))
        except ValueError:
            continue
    return out


def fetch_odds(event_name, when=None):
    """Best-effort odds from ufc.com, or {} if the card cannot be located."""
    import requests
    for slug in _slug_candidates(event_name, when):
        try:
            resp = requests.get(f"https://www.ufc.com/event/{slug}", timeout=30,
                                headers={"User-Agent": ufcnet.UA})
        except requests.RequestException:
            continue
        if resp.status_code != 200:
            continue
        odds = _parse_odds_page(resp.text)
        if odds:
            return odds
    return {}


def recommend(bouts, prob_of, odds_map):
    """Kelly stakes where the blended probability clears MIN_EDGE.

    The betting probability is BLEND_W * (two-orientation model average) +
    (1 - BLEND_W) * the de-vigged market probability, and the edge is measured
    against the de-vigged price, all via betting_math.
    """
    picks = []
    for a, b in bouts:
        ab, ba = prob_of.get((a, b)), prob_of.get((b, a))
        if ab is None or ba is None:
            continue
        odds = odds_map.get((a.lower(), b.lower()))
        flip = False
        if odds is None:
            odds = odds_map.get((b.lower(), a.lower()))
            flip = odds is not None
        if odds is None:
            continue
        odds_a, odds_b = (odds[1], odds[0]) if flip else odds

        model_a = (ab + (1 - ba)) / 2
        bet = betting_math.decide_bet(model_a, None, odds_a, odds_b,
                                      blend_w=BLEND_W, min_edge=MIN_EDGE,
                                      fraction=KELLY_FRACTION, cap=KELLY_MAX,
                                      bankroll=BANKROLL)
        if bet is None:
            continue
        name, opponent, price = ((a, b, odds_a) if bet["name_index"] == 0
                                 else (b, a, odds_b))
        picks.append(dict(fighter=name, opponent=opponent, odds=price,
                          model_prob=round(bet["prob"], 4),
                          implied_prob=round(bet["market_prob"], 4),
                          edge=round(bet["edge"], 4), kelly=round(bet["kc"], 4),
                          stake_pct=round(bet["stake"] / BANKROLL * 100, 2)))
    return picks


# -------------------------------------------------------------------- output

def write_outputs(rows, event, bets, event_date):
    predict_data = [{"Red Fighter": r, "Blue Fighter": b,
                     "Probability Win": p, "Probability Lose": 1 - p}
                    for r, b, p in rows]
    payload = {
        "predict_data": predict_data,
        "class_probabilities": {"Win": [p for _, _, p in rows],
                                "Lose": [1 - p for _, _, p in rows]},
        "event": event,
        "event_date": event_date,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if bets:
        payload["bets"] = bets
    with open(PRED_JSON, "w") as fh:
        json.dump(payload, fh)

    with open(BET_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Red Fighter", "Blue Fighter", "Probability Win", "Probability Lose"])
        for r, b, p in rows:
            w.writerow([r, b, p, 1 - p])

    if bets:
        try:
            bet_ledger.record(event, event_date, payload["generated"], bets)
        except Exception as exc:  # the ledger is secondary output; never block predictions
            print(f"warning: bet ledger not updated: {type(exc).__name__}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show upcoming events and exit")
    ap.add_argument("--event", help="ufcstats event-details URL to predict")
    ap.add_argument("--odds", action="store_true", help="pull odds and size bets")
    args = ap.parse_args()

    session = ufcnet.new_session()

    if args.list:
        for when, url, name in upcoming_events(session):
            print(f"{when.date()}  {name}\n            {url}")
        return 0

    when = None
    if args.event:
        url = args.event
        try:
            when = next((w for w, u, _ in upcoming_events(session) if u == url), None)
        except Exception:  # the date is a convenience; event_card below surfaces real network faults
            when = None
    else:
        events = upcoming_events(session)
        when, url, name = events[0]
        print(f"Next event: {name} — {when.date()}")
    event_date = (when or datetime.datetime.now()).strftime("%Y-%m-%d")

    event_name, bouts, ids = event_card(session, url)
    print(f"{event_name}: {len(bouts)} bouts")

    written, skipped = build_features(bouts, ids)
    for (a, b), reason in skipped:
        print(f"  skipped {a} vs {b} — {reason}")
    if written == 0:
        raise RuntimeError("no bout on this card could be turned into features")

    rows = predict_rows()
    prob_of = {(r, b): p for r, b, p in rows}

    bets = []
    if args.odds:
        odds_map = fetch_odds(event_name, when)
        if not odds_map:
            print("  odds unavailable for this card — predictions only")
        else:
            bets = recommend(bouts, prob_of, odds_map)

    write_outputs(rows, event_name, bets, event_date)

    print(f"\n{'FIGHT':<48}{'MODEL PICK':>26}")
    seen = set()
    for a, b in bouts:
        ab, ba = prob_of.get((a, b)), prob_of.get((b, a))
        if ab is None or ba is None or (a, b) in seen:
            continue
        seen.add((a, b))
        # Average the two orientations for a single headline number.
        p = (ab + (1 - ba)) / 2
        favourite, pf = (a, p) if p >= 0.5 else (b, 1 - p)
        print(f"{a + ' vs ' + b:<48}{favourite + ' ' + format(pf, '.1%'):>26}")

    if bets:
        print(f"\n{'BET':<26}{'ODDS':>7}{'BLEND*':>8}{'MARKET':>9}{'EDGE':>7}{'STAKE':>8}")
        for x in bets:
            print(f"{x['fighter']:<26}{x['odds']:>7}{x['model_prob']:>8.1%}"
                  f"{x['implied_prob']:>9.1%}{x['edge']:>7.1%}{x['stake_pct']:>7.2f}%")
        print("\n* Sizing blends the model average with the devigged market probability"
              "\n  (w=0.8), so this differs from the number above. Stake is % of bankroll.")
    elif args.odds:
        print("\nNo bet cleared the 5% edge threshold.")

    print(f"\nWrote {os.path.relpath(PRED_JSON, ROOT)} and {os.path.relpath(BET_CSV, ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScrapeError as exc:
        print(f"SCRAPE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
