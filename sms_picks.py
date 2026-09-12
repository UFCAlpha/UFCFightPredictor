#!/usr/bin/env python3
"""Friday UFC predictions by SMS. Preview by default; --send enables Twilio."""
import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import textwrap
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent
EASTERN = ZoneInfo("America/Toronto")


def in_send_window(now):
    local = now.astimezone(EASTERN)
    return local.weekday() == 4 and local.hour == 21 and local.minute < 15


def weekend_events(events, today):
    """Only future Saturday/Sunday cards in the current Monday-Sunday week."""
    saturday = today + dt.timedelta(days=5 - today.weekday())
    sunday = saturday + dt.timedelta(days=1)
    return sorted(e for e in events if today < e[0].date() and saturday <= e[0].date() <= sunday)


def collect_reports(today):
    """Reuse the live inference and Kelly paths, with an isolated feature CSV.

    Never read yesterday's predicted_data.json or overwrite the public outputs.
    ufcstats is always accessed through predict_event's challenge-aware session.
    """
    import predict_event as predictor

    session = predictor.ufcnet.new_session()
    events = weekend_events(predictor.upcoming_events(session), today)
    if not events:
        return []
    # The legacy feature builder reads input paths relative to the repo.
    import predict_fights_alpha as features

    reports = []
    for when, url, _ in events:
        name, bouts = predictor.event_card(session, url)
        with tempfile.TemporaryDirectory(prefix="ufc-sms-") as scratch:
            original = features.output_csv_filename
            try:
                features.output_csv_filename = str(Path(scratch) / "features.csv")
                written, skipped = predictor.build_features(bouts)
                if not written:
                    raise RuntimeError(f"No predictions generated for {name}")
                rows = predictor.predict_rows()
            finally:
                features.output_csv_filename = original
        validate_rows(bouts, rows, skipped)
        odds_map = predictor.fetch_odds(name, when)
        # Invalid prices cannot be used in Kelly math. Treat these as unavailable.
        odds_map = {pair: prices for pair, prices in odds_map.items()
                    if len(prices) == 2 and all(math.isfinite(p) and abs(p) >= 100 for p in prices)}
        bets = predictor.recommend(bouts, {(a, b): p for a, b, p in rows}, odds_map)
        reports.append(dict(event=name, event_date=when.date().isoformat(), event_url=url,
                            bouts=bouts, rows=rows, skipped=skipped, odds_map=odds_map, bets=bets))
    return reports


def validate_rows(bouts, rows, skipped):
    probabilities = {(a, b): p for a, b, p in rows}
    omitted = {tuple(pair) for pair, _ in skipped}
    expected = {(a, b) for a, b in bouts if (a, b) not in omitted}
    expected |= {(b, a) for a, b in list(expected)}
    if len(probabilities) != len(rows) or set(probabilities) != expected:
        raise RuntimeError("Incomplete or unexpected prediction orientations")
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()):
        raise RuntimeError("Invalid model probabilities")


def format_messages(report):
    probs = {(a, b): p for a, b, p in report["rows"]}
    skipped = {tuple(pair): reason for pair, reason in report["skipped"]}
    lines = ["PREDICTIONS (model probability)"]
    missing_odds = []
    for a, b in report["bouts"]:
        if (a, b) in skipped:
            lines.append(f"{a} vs {b}: unavailable ({skipped[(a, b)]})")
            continue
        p = (probs[a, b] + 1 - probs[b, a]) / 2
        pick, confidence = (a, p) if p >= .5 else (b, 1 - p)
        lines.append(f"{a} vs {b}: {pick} {confidence:.1%}")
        if ((a.lower(), b.lower()) not in report["odds_map"] and
                (b.lower(), a.lower()) not in report["odds_map"]):
            missing_odds.append(f"{a} vs {b}: odds unavailable")
    lines.extend(["", "BETS (5% Kelly, capped at 5% of bankroll)"])
    for bet in report["bets"]:
        lines.append(f"{bet['fighter']} vs {bet['opponent']} {bet['odds']:+d}: "
                     f"Kelly {bet['kelly']:.2%}; stake = bankroll x {bet['stake_pct']:g}%")
    if not report["odds_map"]:
        lines.append("Odds unavailable; no stakes calculated.")
    elif not report["bets"]:
        lines.append("No bets qualify among bouts with available odds.")
    lines.extend(missing_odds)
    lines.append("Stake % already includes fractional Kelly and cap; do not multiply by 5% again.")
    lines.append("Sizing uses 80% model + 20% devigged market. Odds are a snapshot; check before betting.")
    # Bound each request well below Twilio's 1,600-character limit, including
    # UTF-16 surrogate pairs. Preserve complete lines except pathological names.
    blocks, current = [], ""
    for line in lines:
        for piece in textwrap.wrap(line, width=600, replace_whitespace=False) or [""]:
            candidate = current + ("\n" if current else "") + piece
            if len(candidate.encode("utf-16-le")) // 2 > 1250:
                blocks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        blocks.append(current)
    header = f"UFC Alpha | {report['event'][:80]} | {report['event_date']}"
    return [f"{header} ({i}/{len(blocks)})\n{block}" for i, block in enumerate(blocks, 1)]


class TwilioSender:
    """One POST per part, no implicit retries and no sensitive response logging."""
    def __init__(self, config):
        required = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER", "UFC_SMS_TO")
        missing = [key for key in required if not config.get(key)]
        if missing:
            raise ValueError("Missing SMS configuration: " + ", ".join(missing))
        self.account = config["TWILIO_ACCOUNT_SID"]
        self.token = config["TWILIO_AUTH_TOKEN"]
        self.sender = config["TWILIO_FROM_NUMBER"]
        self.recipient = config["UFC_SMS_TO"]
        if not re.fullmatch(r"AC[0-9a-fA-F]{32}", self.account):
            raise ValueError("TWILIO_ACCOUNT_SID must be an AC account SID")
        for key, phone in [("TWILIO_FROM_NUMBER", self.sender), ("UFC_SMS_TO", self.recipient)]:
            if not re.fullmatch(r"\+[1-9][0-9]{7,14}", phone):
                raise ValueError(f"{key} must be an E.164 phone number")

    def __call__(self, body):
        response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{self.account}/Messages.json",
            auth=(self.account, self.token),
            data={"From": self.sender, "To": self.recipient, "Body": body}, timeout=30)
        if response.status_code != 201:
            raise RuntimeError(f"Twilio rejected submission (HTTP {response.status_code}); check Twilio logs")
        payload = response.json()
        if (not re.fullmatch(r"SM[0-9a-fA-F]{32}", payload.get("sid", "")) or
                payload.get("status") not in {"accepted", "queued", "sending", "sent", "delivered"}):
            raise RuntimeError("Twilio did not confirm submission; check Twilio logs")
        return payload["sid"]


@contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another SMS run is already running") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def save_state(path, state):
    """Atomic, durable intent written before each outbound request."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            json.dump(state, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def deliver(path, messages, send, *, before_send=None):
    with file_lock(path.with_suffix(".lock")):
        state = json.loads(path.read_text()) if path.exists() else {
            "parts": [dict(body=body, status="pending") for body in messages]}
        if (not isinstance(state, dict) or not isinstance(state.get("parts"), list)
                or not state["parts"] or any(
                    not isinstance(part, dict) or not isinstance(part.get("body"), str)
                    or part.get("status") not in {"pending", "sending", "submitted"}
                    or (part["status"] == "submitted" and not part.get("sid"))
                    for part in state["parts"])):
            raise RuntimeError(f"Invalid SMS delivery state; inspect {path}")
        # An interrupted POST may have reached Twilio. Never silently retry it.
        if any(part["status"] == "sending" for part in state["parts"]):
            raise RuntimeError(f"SMS delivery uncertain; inspect Twilio logs and {path} before retrying")
        if all(part["status"] == "submitted" for part in state["parts"]):
            return "already submitted"
        if [part["body"] for part in state["parts"]] != messages:
            raise RuntimeError("Forecast changed after partial SMS submission; review before resuming")
        for part in state["parts"]:
            if part["status"] == "submitted":
                continue
            if before_send is not None:
                before_send()
            part["status"] = "sending"
            save_state(path, state)
            sid = send(part["body"])
            part.update(status="submitted", sid=sid)
            save_state(path, state)
        return "submitted"


def verify_card(report):
    import predict_event as predictor
    _, current = predictor.event_card(predictor.ufcnet.new_session(), report["event_url"])
    if {frozenset(bout) for bout in current} != {frozenset(bout) for bout in report["bouts"]}:
        raise RuntimeError("Card changed during predictions; hold SMS and review the updated card")


def run(*, send, scheduled, state_dir, hold_file, now=None):
    now = now or dt.datetime.now(EASTERN)
    if scheduled and not in_send_window(now):
        return 0
    if hold_file.exists():
        print("SMS on hold; remove the hold file after reviewing the card.")
        return 0
    sender = TwilioSender(os.environ) if send else None
    reports = collect_reports(now.astimezone(EASTERN).date())
    if not reports:
        print("No upcoming card this weekend; no SMS.")
        return 0
    for report in reports:
        messages = format_messages(report)
        if not send:
            print("\n\n".join(messages))
            continue
        key = hashlib.sha256((report["event_url"].rstrip("/").split("/")[-1] + "|" +
                              report["event_date"] + "|" + sender.recipient).encode()).hexdigest()
        def before_send():
            verify_card(report)
            # Recheck after the network fetch: a hold can be enabled, or the
            # send window can close, while waiting for the card response.
            if hold_file.exists():
                raise RuntimeError("SMS hold enabled before submission")
            if scheduled and not in_send_window(dt.datetime.now(EASTERN)):
                raise RuntimeError("Friday SMS window closed before submission")
        result = deliver(state_dir / f"{key}.json", messages, sender, before_send=before_send)
        print(f"{report['event']}: {result} to Twilio (delivery not yet confirmed)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--send", action="store_true", help="submit SMS through Twilio")
    mode.add_argument("--dry-run", action="store_true", help="preview only (default)")
    ap.add_argument("--scheduled", action="store_true", help="only run Friday 21:00-21:14 Toronto time")
    ap.add_argument("--check-config", action="store_true", help="validate credentials format without network calls")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    if args.check_config:
        TwilioSender(os.environ)
        print("SMS configuration format is valid (credentials not authenticated).")
        return 0
    state_dir = Path(os.environ.get("UFC_SMS_STATE_DIR", ROOT / "data/sms_delivery"))
    hold_file = Path(os.environ.get("UFC_SMS_HOLD_FILE", ROOT / "data/sms.hold"))
    if args.send:
        with file_lock(state_dir / "run.lock"):
            return run(send=True, scheduled=args.scheduled, state_dir=state_dir, hold_file=hold_file)
    return run(send=False, scheduled=args.scheduled, state_dir=state_dir, hold_file=hold_file)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # requests exceptions can carry URLs/account details; keep logs limited.
        print(f"SMS FAILED: {type(exc).__name__}: " +
              ("network request failed; check provider/network logs" if isinstance(exc, requests.RequestException)
               else str(exc)), file=sys.stderr)
        raise SystemExit(1)
