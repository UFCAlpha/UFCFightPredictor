#!/usr/bin/env python3
"""Friday UFC predictions by Gmail. Preview by default; --send enables email."""
import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
from html import escape

import kalshi_odds
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import smtplib
import ssl
from email.message import EmailMessage
from email.headerregistry import Address
from email.utils import formatdate, make_msgid
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
        name, bouts, ids = predictor.event_card(session, url)
        with tempfile.TemporaryDirectory(prefix="ufc-email-") as scratch:
            original = features.output_csv_filename
            try:
                features.output_csv_filename = str(Path(scratch) / "features.csv")
                written, skipped = predictor.build_features(bouts, ids)
                if not written:
                    raise RuntimeError(f"No predictions generated for {name}")
                rows = predictor.predict_rows()
            finally:
                features.output_csv_filename = original
        validate_rows(bouts, rows, skipped)
        try:
            quotes = kalshi_odds.fetch_quotes(bouts, when.date())
        except (requests.RequestException, ValueError, KeyError, TypeError):
            print("Kalshi quotes unavailable; predictions only.", file=sys.stderr)
            quotes = {}
        odds_map = {pair: tuple(q['odds'] for q in sides) for pair, sides in quotes.items()}
        bets = predictor.recommend(bouts, {(a, b): p for a, b, p in rows}, odds_map)
        reports.append(dict(event=name, event_date=when.date().isoformat(), event_url=url,
                            bouts=bouts, rows=rows, skipped=skipped, odds_map=odds_map, bets=bets,
                            kalshi_quotes=quotes, odds_fetched_at=dt.datetime.now(EASTERN).isoformat(timespec='seconds')))
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
            lines.append("Kalshi odds: " + _kalshi_cell(report, a, b).replace("\n", "; "))
            continue
        p = (probs[a, b] + 1 - probs[b, a]) / 2
        pick, confidence = (a, p) if p >= .5 else (b, 1 - p)
        lines.append(f"{a} vs {b}: {pick} {confidence:.1%}")
        lines.append("Kalshi odds: " + _kalshi_cell(report, a, b).replace("\n", "; "))
        if ((a.lower(), b.lower()) not in report["odds_map"] and
                (b.lower(), a.lower()) not in report["odds_map"]):
            missing_odds.append(f"{a} vs {b}: odds unavailable")
    lines.extend(["", "BETS — Kalshi (5% Kelly, capped at 5% of bankroll)"])
    for bet in report["bets"]:
        lines.append(f"{bet['fighter']} vs {bet['opponent']} {bet['odds']:+.0f}: "
                     f"Kelly {bet['kelly']:.2%}; stake = bankroll x {bet['stake_pct']:g}%")
    if not report["odds_map"]:
        lines.append("Odds unavailable; no stakes calculated.")
    elif not report["bets"]:
        lines.append("No bets qualify among bouts with available odds.")
    lines.extend(missing_odds)
    lines.append("Stake % already includes fractional Kelly and cap; do not multiply by 5% again.")
    lines.append("Sizing uses 80% model + 20% devigged market. Kalshi buy-price snapshot; stakes are before exchange fees. Check prices and available size before betting.")
    header = f"UFC Alpha | {report['event']} | {report['event_date']}"
    return [header + "\n" + "\n".join(lines)]


def _html_table(headings, rows):
    head = ''.join(f'<th scope="col" style="padding:11px 9px;text-align:left;background:#eef1f5;">{escape(h)}</th>' for h in headings)
    body = ''.join('<tr>' + ''.join(
        '<td style="padding:11px 9px;border-bottom:1px solid #e4e7ec;vertical-align:top;">' +
        escape(str(value)).replace('\n', '<br>') + '</td>' for value in row) + '</tr>' for row in rows)
    return f'<table style="width:100%;border-collapse:collapse;font-size:14px;"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _kalshi_cell(report, a, b):
    quotes = report.get('kalshi_quotes', {})
    sides = quotes.get((a.lower(), b.lower()))
    if sides is None:
        reverse = quotes.get((b.lower(), a.lower()))
        sides = list(reversed(reverse)) if reverse else None
    if not sides:
        return 'Unavailable'
    return '\n'.join(f"{label}: {q['price'] * 100:g}¢ ({q['odds']:+.0f})"
                     for label, q in zip(('A', 'B'), sides))


def format_html(report):
    """Email-safe HTML tables, with escaped source data and inline styles."""
    probabilities = {(a, b): p for a, b, p in report['rows']}
    skipped = {tuple(pair): reason for pair, reason in report['skipped']}
    rows, notes = [], []
    for a, b in report['bouts']:
        pick, probability = 'Unavailable', '—'
        if (a, b) in skipped:
            notes.append(f'{a} vs {b}: {skipped[a, b]}')
        else:
            p = (probabilities[a, b] + 1 - probabilities[b, a]) / 2
            pick, confidence = (a, p) if p >= .5 else (b, 1 - p)
            probability = f'{confidence:.1%}'
        rows.append([a, b, pick, probability, _kalshi_cell(report, a, b)])
    fights = _html_table(['Fighter A', 'Fighter B', 'Model Pick', 'Probability', 'Kalshi Odds'], rows)
    bet_rows = [[b['fighter'], b['opponent'], f"{b['odds']:+.0f}", f"{b['kelly']:.2%}",
                 f"bankroll × {b['stake_pct']:g}%"] for b in report['bets']]
    if bet_rows:
        bets = _html_table(['Bet On', 'Opponent', 'Kalshi Odds', 'Full Kelly', 'Stake'], bet_rows)
    else:
        notice = 'No bets qualify among bouts with available Kalshi quotes.' if report['odds_map'] else 'Kalshi odds unavailable; no stakes calculated.'
        bets = f'<p>{notice}</p>'
    notes_html = ''.join(f'<li>{escape(n)}</li>' for n in notes)
    fetched = report.get('odds_fetched_at', 'Not recorded')
    return ('<!doctype html><html><head><meta charset="utf-8"></head>'
            '<body style="margin:0;background:#f4f5f7;color:#17202b;font-family:Arial,Helvetica,sans-serif;">'
            '<div style="max-width:960px;margin:auto;padding:24px 16px;background:#ffffff;">'
            '<p style="font-size:12px;font-weight:bold;color:#bd242b;letter-spacing:2px;">UFC ALPHA</p>'
            f'<h1 style="font-size:24px;margin-bottom:8px;">{escape(report["event"])}</h1>'
            f'<p style="color:#596574;">{escape(report["event_date"])}</p>'
            '<h2 style="font-size:18px;">Fight predictions</h2>' + fights +
            '<p style="font-size:12px;color:#596574;">Probability refers to the model pick. Kalshi A/B prices correspond to Fighter A/B: YES buy price in cents, then American odds.</p>'
            '<h2 style="font-size:18px;margin-top:30px;">Bets · Kalshi</h2>' + bets +
            '<p style="font-size:12px;color:#596574;">Stakes use 5% fractional Kelly, capped at 5% of bankroll. The stake percentage already includes both; do not multiply by 5% again. Sizing blends 80% model with 20% normalized market probability. Estimates are before Kalshi fees and slippage; check current prices and available size before betting.</p>'
            f'<p style="font-size:12px;color:#596574;">Kalshi quotes fetched: {escape(fetched)}</p>' +
            (f'<h3 style="font-size:14px;">Unavailable predictions</h3><ul style="font-size:12px;color:#596574;">{notes_html}</ul>' if notes else '') +
            '</div></body></html>')


class NotSubmittedError(RuntimeError):
    """Known failure before acceptance; retrying cannot duplicate delivery."""


class GmailSender:
    """Gmail SMTP with verified TLS and one recipient. No automatic SMTP retry."""
    def __init__(self, config):
        required = ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD")
        missing = [key for key in required if not config.get(key)]
        if missing:
            raise ValueError("Missing email configuration: " + ", ".join(missing))
        self.sender = config["GMAIL_ADDRESS"].strip()
        self.recipient = (config.get("UFC_EMAIL_TO") or self.sender).strip()
        self.password = "".join(config["GMAIL_APP_PASSWORD"].split())
        if not re.fullmatch(r"[a-zA-Z0-9]{16}", self.password):
            raise ValueError("GMAIL_APP_PASSWORD must be a 16-character Google app password")
        for key, address in [("GMAIL_ADDRESS", self.sender), ("UFC_EMAIL_TO", self.recipient)]:
            try:
                parsed = Address(addr_spec=address)
                if (not address.isascii() or not parsed.username or "." not in parsed.domain
                        or parsed.addr_spec != address):
                    raise ValueError()
            except (ValueError, IndexError):
                raise ValueError(f"{key} must be a single email address without a display name") from None

    def __call__(self, content, *, before_submit=None, html=None):
        subject, body = content.split("\n", 1)
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.sender
        message["To"] = self.recipient
        message["Date"] = formatdate(localtime=False)
        message["Message-ID"] = make_msgid(domain=self.sender.split("@", 1)[1])
        message.set_content(body)
        if html is not None:
            message.add_alternative(html, subtype="html")
        client = None
        try:
            try:
                # Framework Python on macOS may have no system CA bundle.
                # Use Requests' installed roots while still verifying TLS.
                context = ssl.create_default_context(cafile=requests.certs.where())
                client = smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=30)
                client.login(self.sender, self.password)
            except (OSError, smtplib.SMTPException):
                raise NotSubmittedError("Gmail connection/login failed; check network and app password") from None
            if before_submit is not None:
                before_submit()
            try:
                refused = client.send_message(message, from_addr=self.sender, to_addrs=[self.recipient])
                if refused:
                    raise NotSubmittedError("Gmail refused the recipient")
            except (smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused,
                    smtplib.SMTPDataError, smtplib.SMTPNotSupportedError):
                raise NotSubmittedError("Gmail rejected the email; check account and recipient settings") from None
            # Gmail accepted DATA. A Message-ID is a correlation ID, not proof
            # of inbox delivery. Closing the socket cannot undo that acceptance.
            return str(message["Message-ID"])
        finally:
            if client is not None:
                try:
                    client.close()
                except OSError:
                    pass


@contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Email run is already running") from None
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
                    or (part["status"] == "submitted" and not part.get("message_id"))
                    for part in state["parts"])):
            raise RuntimeError(f"Invalid email delivery state; inspect {path}")
        # An interrupted SMTP submission may have reached Gmail. Never silently retry it.
        if any(part["status"] == "sending" for part in state["parts"]):
            raise RuntimeError(f"Email delivery uncertain; inspect Gmail's Sent folder and {path} before retrying")
        if all(part["status"] == "submitted" for part in state["parts"]):
            return "already submitted"
        if [part["body"] for part in state["parts"]] != messages:
            if any(part["status"] == "submitted" for part in state["parts"]):
                raise RuntimeError("Forecast changed after partial email submission; review before resuming")
            state = {"parts": [dict(body=body, status="pending") for body in messages]}
        for part in state["parts"]:
            if part["status"] == "submitted":
                continue
            if before_send is not None:
                before_send()
            part["status"] = "sending"
            save_state(path, state)
            try:
                message_id = send(part["body"])
            except NotSubmittedError:
                part["status"] = "pending"
                save_state(path, state)
                raise
            part.update(status="submitted", message_id=message_id)
            save_state(path, state)
        return "submitted"


def verify_card(report):
    import predict_event as predictor
    _, current, _ = predictor.event_card(predictor.ufcnet.new_session(), report["event_url"])
    if {frozenset(bout) for bout in current} != {frozenset(bout) for bout in report["bouts"]}:
        raise RuntimeError("Card changed during predictions; hold email and review the updated card")


def run(*, send, scheduled, state_dir, hold_file, now=None):
    now = now or dt.datetime.now(EASTERN)
    if scheduled and not in_send_window(now):
        return 0
    if hold_file.exists():
        print("Email on hold; remove the hold file after reviewing the card.")
        return 0
    sender = GmailSender(os.environ) if send else None
    reports = collect_reports(now.astimezone(EASTERN).date())
    if not reports:
        print("No upcoming card this weekend; no email.")
        return 0
    for report in reports:
        messages = format_messages(report)
        if not send:
            print("\n\n".join(messages))
            continue
        key = hashlib.sha256((report["event_url"].rstrip("/").split("/")[-1] + "|" +
                              report["event_date"] + "|" + sender.recipient).encode()).hexdigest()
        def check_send_allowed():
            if hold_file.exists():
                raise NotSubmittedError("Email hold enabled before submission")
            if scheduled and not in_send_window(dt.datetime.now(EASTERN)):
                raise NotSubmittedError("Friday email window closed before submission")
        def before_send():
            verify_card(report)
            check_send_allowed()
        def submit(content):
            # Login is another network round trip. Check the hold/window again
            # after authentication and before sending any message data.
            return sender(content, before_submit=check_send_allowed, html=format_html(report))
        result = deliver(state_dir / f"{key}.json", messages, submit, before_send=before_send)
        print(f"{report['event']}: {result} to Gmail (delivery not yet confirmed)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--send", action="store_true", help="submit email through Gmail")
    mode.add_argument("--dry-run", action="store_true", help="preview only (default)")
    ap.add_argument("--scheduled", action="store_true", help="only run Friday 21:00-21:14 Toronto time")
    ap.add_argument("--check-config", action="store_true", help="validate credentials format without network calls")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    if args.check_config:
        GmailSender(os.environ)
        print("Email configuration format is valid (credentials not authenticated).")
        return 0
    state_dir = Path(os.environ.get("UFC_EMAIL_STATE_DIR", ROOT / "data/email_delivery"))
    hold_file = Path(os.environ.get("UFC_EMAIL_HOLD_FILE", ROOT / "data/email.hold"))
    if args.send:
        with file_lock(state_dir / "run.lock"):
            return run(send=True, scheduled=args.scheduled, state_dir=state_dir, hold_file=hold_file)
    return run(send=False, scheduled=args.scheduled, state_dir=state_dir, hold_file=hold_file)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # requests exceptions can carry URLs/account details; keep logs limited.
        print(f"Email FAILED: {type(exc).__name__}: " +
              ("network request failed; check provider/network logs" if isinstance(exc, (requests.RequestException, smtplib.SMTPException, OSError))
               else str(exc)), file=sys.stderr)
        raise SystemExit(1)
