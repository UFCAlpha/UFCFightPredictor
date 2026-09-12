"""Delivery tests use fake network boundaries; no email is sent."""
import datetime as dt
import importlib
import json
from types import SimpleNamespace

import pytest
import requests


@pytest.fixture
def picks():
    return importlib.import_module("email_picks")


def test_friday_window_uses_eastern_time_in_summer_and_winter(picks):
    assert picks.in_send_window(dt.datetime.fromisoformat("2026-09-12T01:00:00+00:00"))
    assert picks.in_send_window(dt.datetime.fromisoformat("2026-12-05T02:00:00+00:00"))
    for stamp in ["2026-09-12T00:59:00+00:00", "2026-09-12T01:15:00+00:00",
                  "2026-09-13T01:00:00+00:00", "2026-12-05T01:00:00+00:00"]:
        assert not picks.in_send_window(dt.datetime.fromisoformat(stamp))


def test_selects_only_this_weekends_cards(picks):
    events = [(dt.datetime(2026, 9, day), f"url{day}", f"Card {day}")
              for day in [19, 13, 11, 12]]
    today = dt.date(2026, 9, 11)
    assert [e[1] for e in picks.weekend_events(events, today)] == ["url12", "url13"]
    assert picks.weekend_events(events[:1], today) == []
    assert picks.weekend_events(events, dt.date(2026, 9, 13)) == []


@pytest.fixture
def report():
    return dict(event="UFC Test", event_date="2026-09-12", event_url="url12",
                bouts=[("A", "B"), ("C", "D"), ("E", "F")],
                rows=[("A", "B", .7), ("B", "A", .4), ("C", "D", .4), ("D", "C", .6)],
                skipped=[(("E", "F"), "no UFC history in the dataset")],
                odds_map={("b", "a"): (130, -150)},
                bets=[dict(fighter="A", opponent="B", odds=-150, kelly=.125, stake_pct=.625)])


def test_message_averages_orientations_and_uses_final_stake_once(picks, report):
    text = "\n".join(picks.format_messages(report))
    assert "A vs B: A 65.0%" in text
    assert "C vs D: D 60.0%" in text
    assert "C vs D: odds unavailable" in text
    assert "E vs F: unavailable" in text
    assert "Kelly 12.50%" in text and "bankroll x 0.625%" in text
    assert "UFC Test" in text and "2026-09-12" in text


def test_no_odds_is_distinct_from_no_qualifying_bets(picks, report):
    report["bets"] = []
    assert "No bets qualify" in "\n".join(picks.format_messages(report))
    report["odds_map"] = {}
    text = "\n".join(picks.format_messages(report))
    assert "Odds unavailable; no stakes calculated" in text
    assert "No bets qualify" not in text


def test_long_unicode_card_stays_in_one_email(picks, report):
    report["bouts"] = [(f"Fighter {i} " + "🥊" * 50, f"Opponent {i}") for i in range(30)]
    report["rows"] = []
    report["bets"] = []
    report["skipped"] = [(bout, "no history") for bout in report["bouts"]]
    messages = picks.format_messages(report)
    assert len(messages) == 1
    for a, b in report["bouts"]:
        assert f"{a} vs {b}" in messages[0]


def test_delivery_is_persisted_and_duplicate_run_sends_nothing(picks, tmp_path):
    path = tmp_path / "delivery.json"
    calls = []
    def send(body):
        calls.append(body)
        return "<test@example.com>"
    assert picks.deliver(path, ["one", "two"], send) == "submitted"
    assert picks.deliver(path, ["changed forecast"], send) == "already submitted"
    assert calls == ["one", "two"]
    assert json.loads(path.read_text())["parts"][1]["message_id"].startswith("<test@")


def test_timeout_blocks_retries_and_preserves_accepted_parts(picks, tmp_path):
    path = tmp_path / "delivery.json"
    calls = []
    def send(body):
        calls.append(body)
        if body == "two":
            raise requests.Timeout("uncertain")
        return "<test@example.com>"
    with pytest.raises(requests.Timeout):
        picks.deliver(path, ["one", "two", "three"], send)
    with pytest.raises(RuntimeError, match="uncertain"):
        picks.deliver(path, ["new"], send)
    assert calls == ["one", "two"]
    state = json.loads(path.read_text())
    assert state["parts"][0]["status"] == "submitted"
    assert state["parts"][1]["status"] == "sending"
    assert state["parts"][2]["status"] == "pending"


def test_concurrent_delivery_cannot_double_send(picks, tmp_path):
    path = tmp_path / "delivery.json"
    def send(body):
        with pytest.raises(RuntimeError, match="running"):
            picks.deliver(path, [body], lambda _: pytest.fail("duplicate send"))
        return "<test@example.com>"
    assert picks.deliver(path, ["one"], send) == "submitted"


@pytest.fixture
def gmail_config(monkeypatch):
    config = dict(GMAIL_ADDRESS="picks@gmail.com", GMAIL_APP_PASSWORD="abcd efgh ijkl mnop",
                  UFC_EMAIL_TO="recipient@example.com")
    for key, value in config.items():
        monkeypatch.setenv(key, value)
    return config


@pytest.fixture
def smtp_server(picks, monkeypatch):
    class SMTP:
        def __init__(self, host, port, *, context, timeout):
            import ssl
            assert (host, port, timeout) == ("smtp.gmail.com", 465, 30)
            assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
            self.messages = []
            self.closed = False
        def login(self, address, password):
            assert (address, password) == ("picks@gmail.com", "abcdefghijklmnop")
        def send_message(self, message, *, from_addr, to_addrs):
            assert from_addr == "picks@gmail.com"
            assert to_addrs == [message["To"]]
            self.messages.append(message)
            return {}
        def close(self):
            self.closed = True
    server = SMTP("smtp.gmail.com", 465, context=__import__("ssl").create_default_context(), timeout=30)
    monkeypatch.setattr(picks.smtplib, "SMTP_SSL", lambda *a, **kw: (SMTP(*a, **kw), server)[1])
    return server


def test_gmail_sends_one_unicode_email_over_verified_tls(picks, gmail_config, smtp_server):
    receipt = picks.GmailSender(gmail_config)("UFC Alpha | Test\nA vs B: A 65.0%\nJosé wins")
    message = smtp_server.messages[0]
    assert message["Subject"] == "UFC Alpha | Test"
    assert message["From"] == "picks@gmail.com"
    assert message["To"] == "recipient@example.com"
    assert "José wins" in message.get_content()
    assert message["Message-ID"] == receipt
    assert smtp_server.closed


def test_recipient_defaults_to_sender(picks, gmail_config, smtp_server):
    del gmail_config["UFC_EMAIL_TO"]
    picks.GmailSender(gmail_config)("Picks\nFull card")
    assert smtp_server.messages[0]["To"] == "picks@gmail.com"


@pytest.mark.parametrize("field,value", [("GMAIL_ADDRESS", "bad"),
    ("UFC_EMAIL_TO", "a@example.com\nBcc: victim@example.com"),
    ("UFC_EMAIL_TO", "a@example.com,b@example.com"), ("GMAIL_APP_PASSWORD", "short")])
def test_gmail_rejects_invalid_configuration(picks, gmail_config, field, value):
    gmail_config[field] = value
    with pytest.raises(ValueError):
        picks.GmailSender(gmail_config)


def test_authentication_failure_is_retryable_and_password_is_not_logged(picks, gmail_config, smtp_server, tmp_path):
    def fail(*args):
        raise picks.smtplib.SMTPAuthenticationError(535, b"private provider details")
    smtp_server.login = fail
    path = tmp_path / "receipt.json"
    with pytest.raises(picks.NotSubmittedError) as error:
        picks.deliver(path, ["Picks\nCard"], picks.GmailSender(gmail_config))
    assert "private" not in str(error.value) and "abcdefghijklmnop" not in str(error.value)
    assert json.loads(path.read_text())["parts"][0]["status"] == "pending"
    assert smtp_server.closed


def test_timeout_during_smtp_submission_stays_uncertain(picks, gmail_config, smtp_server, tmp_path):
    smtp_server.send_message = lambda *a, **kw: (_ for _ in ()).throw(TimeoutError())
    path = tmp_path / "receipt.json"
    with pytest.raises(TimeoutError):
        picks.deliver(path, ["Picks\nCard"], picks.GmailSender(gmail_config))
    with pytest.raises(RuntimeError, match="uncertain"):
        picks.deliver(path, ["Picks\nCard"], picks.GmailSender(gmail_config))
    assert smtp_server.closed


def test_hold_and_outside_window_skip_before_prediction(picks, tmp_path, monkeypatch):
    monkeypatch.setattr(picks, "collect_reports", lambda *a: pytest.fail("predictions ran"))
    hold = tmp_path / "hold"
    hold.touch()
    assert picks.run(send=True, scheduled=False, state_dir=tmp_path, hold_file=hold) == 0
    hold.unlink()
    assert picks.run(send=True, scheduled=True, state_dir=tmp_path, hold_file=hold,
                   now=dt.datetime.fromisoformat("2026-09-12T02:00:00+00:00")) == 0


def test_dry_run_prints_without_credentials_or_delivery_state(picks, tmp_path, monkeypatch, report, capsys):
    monkeypatch.setattr(picks, "collect_reports", lambda *a: [report])
    assert picks.run(send=False, scheduled=False, state_dir=tmp_path / "state",
                   hold_file=tmp_path / "hold") == 0
    assert "A vs B: A 65.0%" in capsys.readouterr().out
    assert not (tmp_path / "state").exists()


@pytest.fixture
def fake_predictor(monkeypatch):
    import sys
    import predict_event
    scratch = SimpleNamespace(output_csv_filename="original.csv")
    monkeypatch.setitem(sys.modules, "predict_fights_alpha", scratch)
    monkeypatch.setattr(predict_event.ufcnet, "new_session", lambda: object())
    monkeypatch.setattr(predict_event, "upcoming_events", lambda _: [
        (dt.datetime(2026, 9, 12), "url12", "UFC Test")])
    monkeypatch.setattr(predict_event, "event_card", lambda *a: ("UFC Test", [("A", "B")]))
    def features(bouts):
        assert scratch.output_csv_filename != "original.csv"
        return 2, []
    monkeypatch.setattr(predict_event, "build_features", features)
    monkeypatch.setattr(predict_event, "predict_rows", lambda: [("A", "B", .8), ("B", "A", .2)])
    import kalshi_odds
    monkeypatch.setattr(kalshi_odds, "fetch_quotes", lambda *a: {
        ("a", "b"): [dict(price=.6, odds=-150), dict(price=100/230, odds=130)]})
    monkeypatch.setattr(predict_event, "fetch_odds", lambda *a: pytest.fail("sportsbook odds must not size Kalshi bets"))
    return predict_event, scratch


def test_collection_uses_real_kelly_and_restores_feature_output(picks, fake_predictor):
    reports = picks.collect_reports(dt.date(2026, 9, 11))
    report = reports[0]
    assert report["event_date"] == "2026-09-12"
    assert report["bets"][0]["fighter"] == "A"
    assert report["bets"][0]["stake_pct"] == 1.95
    assert fake_predictor[1].output_csv_filename == "original.csv"


@pytest.mark.parametrize("rows", [[("A", "B", .8)], [("A", "B", float("nan")), ("B", "A", .2)]])
def test_bad_prediction_output_aborts(picks, fake_predictor, monkeypatch, rows):
    monkeypatch.setattr(fake_predictor[0], "predict_rows", lambda: rows)
    with pytest.raises(RuntimeError):
        picks.collect_reports(dt.date(2026, 9, 11))
    assert fake_predictor[1].output_csv_filename == "original.csv"


def test_empty_scrape_propagates_failure(picks, fake_predictor, monkeypatch):
    def fail(_):
        raise fake_predictor[0].ScrapeError("empty index")
    monkeypatch.setattr(fake_predictor[0], "upcoming_events", fail)
    with pytest.raises(fake_predictor[0].ScrapeError):
        picks.collect_reports(dt.date(2026, 9, 11))


def test_changed_card_aborts_before_network_send(picks, fake_predictor, report):
    with pytest.raises(RuntimeError, match="Card changed"):
        picks.verify_card(report)


def test_no_weekend_event_never_builds_features(picks, fake_predictor, monkeypatch):
    monkeypatch.setattr(fake_predictor[0], "upcoming_events", lambda _: [
        (dt.datetime(2026, 9, 19), "url19", "Next Week")])
    monkeypatch.setattr(fake_predictor[0], "build_features", lambda _: pytest.fail("built features"))
    assert picks.collect_reports(dt.date(2026, 9, 11)) == []


def test_partial_delivery_cannot_resume_a_different_forecast(picks, tmp_path):
    path = tmp_path / "delivery.json"
    path.write_text(json.dumps({"parts": [
        {"body": "old one", "status": "submitted", "message_id": "<test@example.com>"},
        {"body": "old two", "status": "pending"}]}))
    with pytest.raises(RuntimeError, match="changed"):
        picks.deliver(path, ["new one", "new two"], lambda _: pytest.fail("stale send"))


def test_malformed_journal_cannot_trigger_a_resend(picks, tmp_path):
    path = tmp_path / "delivery.json"
    path.write_text(json.dumps({"parts": [{"body": "one", "status": "typo"}]}))
    with pytest.raises(RuntimeError, match="Invalid"):
        picks.deliver(path, ["one"], lambda _: pytest.fail("resend"))


def test_run_submits_complete_card_once_and_dry_run_does_not_consume_it(
        picks, fake_predictor, gmail_config, smtp_server, tmp_path, capsys):
    kwargs = dict(scheduled=False, state_dir=tmp_path / "receipts", hold_file=tmp_path / "hold",
                  now=dt.datetime.fromisoformat("2026-09-11T21:00:00-04:00"))
    picks.run(send=False, **kwargs)
    assert smtp_server.messages == []
    picks.run(send=True, **kwargs)
    picks.run(send=True, **kwargs)
    assert len(smtp_server.messages) == 1
    content = smtp_server.messages[0].get_body(preferencelist=("plain",)).get_content()
    assert "A vs B: A 80.0%" in content
    assert "stake = bankroll x 1.95%" in content
    assert "already submitted" in capsys.readouterr().out


def test_wrapper_loads_local_config_and_defaults_to_scheduled_send(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    root = Path(__file__).resolve().parents[1]
    shutil.copy(root / "run_email_scheduled.sh", tmp_path)
    interpreter = tmp_path / "fake-python"
    interpreter.write_text('#!/bin/bash\nif [ "$1" = "-c" ]; then exit 0; fi\nprintf "%s\\n" "$PWD" "$UFC_EMAIL_TO" "$@"\n')
    interpreter.chmod(0o700)
    (tmp_path / ".email.env").write_text(f"UFC_PYTHON='{interpreter}'\nUFC_EMAIL_TO='recipient@example.com'\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("UFC_", "TWILIO_"))}
    result = subprocess.run(["/bin/bash", str(tmp_path / "run_email_scheduled.sh")],
                            cwd="/tmp", env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(tmp_path), "recipient@example.com", str(tmp_path / "email_picks.py"),
                                          "--send", "--scheduled"]
    result = subprocess.run(["/bin/bash", str(tmp_path / "run_email_scheduled.sh"), "--dry-run"],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines()[-1] == "--dry-run"
    assert "--send" not in result.stdout


def test_installer_render_does_not_install_or_read_credentials(tmp_path, monkeypatch):
    import plistlib
    import setup_email_launchd
    monkeypatch.setattr(setup_email_launchd.subprocess, "run", lambda *a, **k: pytest.fail("installation attempted"))
    path = tmp_path / "job.plist"
    assert setup_email_launchd.main(["--output", str(path)]) == 0
    job = plistlib.loads(path.read_bytes())
    assert job["ProgramArguments"] == ["/bin/bash", str(setup_email_launchd.ROOT / "run_email_scheduled.sh")]
    assert job["StartInterval"] <= 60
    assert "RunAtLoad" not in job and "KeepAlive" not in job


@pytest.mark.parametrize("change", ["hold", "window"])
def test_hold_or_window_change_during_card_fetch_prevents_send(picks, fake_predictor, monkeypatch, tmp_path, change, gmail_config, smtp_server):
    hold = tmp_path / "hold"
    window_open = [True]
    monkeypatch.setattr(picks, "in_send_window", lambda _: window_open[0])
    def check(_):
        if change == "hold":
            hold.touch()
        else:
            window_open[0] = False
    monkeypatch.setattr(picks, "verify_card", check)
    smtp_server.send_message = lambda *a, **kw: pytest.fail("Email sent after hold/window changed")
    with pytest.raises(RuntimeError, match="hold|window"):
        picks.run(send=True, scheduled=True, state_dir=tmp_path / "state", hold_file=hold,
                now=dt.datetime.fromisoformat("2026-09-11T21:00:00-04:00"))


def test_preflight_failure_can_retry_without_uncertain_email_delivery(picks, tmp_path):
    path = tmp_path / "delivery.json"
    def preflight():
        raise RuntimeError("card fetch failed")
    with pytest.raises(RuntimeError, match="card fetch failed"):
        picks.deliver(path, ["one"], lambda _: pytest.fail("submitted"), before_send=preflight)
    assert picks.deliver(path, ["one"], lambda _: "<test@example.com>") == "submitted"


def test_rendered_job_preserves_custom_config_location(tmp_path, monkeypatch):
    import plistlib
    import setup_email_launchd
    config = tmp_path / "private config.env"
    monkeypatch.setenv("UFC_EMAIL_CONFIG", str(config))
    output = tmp_path / "job.plist"
    setup_email_launchd.main(["--output", str(output)])
    job = plistlib.loads(output.read_bytes())
    assert job["EnvironmentVariables"]["UFC_EMAIL_CONFIG"] == str(config)


def test_known_failed_send_can_use_refreshed_forecast(picks, tmp_path):
    path = tmp_path / "receipt.json"
    def fail(_):
        raise picks.NotSubmittedError("login rejected")
    with pytest.raises(picks.NotSubmittedError):
        picks.deliver(path, ["old forecast"], fail)
    received = []
    picks.deliver(path, ["fresh forecast"], lambda body: received.append(body) or "<new@example.com>")
    assert received == ["fresh forecast"]


def test_hold_enabled_during_gmail_login_stops_submission(picks, fake_predictor, gmail_config, smtp_server, tmp_path):
    hold = tmp_path / "hold"
    smtp_server.login = lambda *a: hold.touch()
    with pytest.raises(picks.NotSubmittedError, match="hold"):
        picks.run(send=True, scheduled=False, state_dir=tmp_path / "state", hold_file=hold,
                now=dt.datetime.fromisoformat("2026-09-11T21:00:00-04:00"))
    assert smtp_server.messages == []


def test_gmail_rejection_is_retryable(picks, gmail_config, smtp_server, tmp_path):
    def reject(*a, **kw):
        raise picks.smtplib.SMTPDataError(550, b"rejected")
    smtp_server.send_message = reject
    path = tmp_path / "receipt.json"
    with pytest.raises(picks.NotSubmittedError):
        picks.deliver(path, ["Picks\nCard"], picks.GmailSender(gmail_config))
    assert json.loads(path.read_text())["parts"][0]["status"] == "pending"


def test_close_error_after_acceptance_does_not_erase_receipt(picks, gmail_config, smtp_server, tmp_path):
    smtp_server.close = lambda: (_ for _ in ()).throw(OSError("closed"))
    path = tmp_path / "receipt.json"
    assert picks.deliver(path, ["Picks\nCard"], picks.GmailSender(gmail_config)) == "submitted"
    assert json.loads(path.read_text())["parts"][0]["status"] == "submitted"


def test_gmail_loads_ca_bundle_without_system_certificates(picks, gmail_config, smtp_server, monkeypatch, tmp_path):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "missing-certs"))
    def connect(*args, **kwargs):
        assert kwargs["context"].get_ca_certs(), "Gmail TLS must have trusted root certificates"
        return smtp_server
    monkeypatch.setattr(picks.smtplib, "SMTP_SSL", connect)
    picks.GmailSender(gmail_config)("Picks\nCard")


def test_html_email_has_fight_table_then_bets(picks, report, gmail_config, smtp_server):
    from bs4 import BeautifulSoup
    report['kalshi_quotes'] = {('a','b'): [dict(price=.6, odds=-150), dict(price=.41, odds=143.9)]}
    html = picks.format_html(report)
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    assert [th.get_text() for th in tables[0].find_all('th')] == [
        'Fighter A','Fighter B','Model Pick','Probability','Kalshi Odds']
    cells = [td.get_text(' ',strip=True) for td in tables[0].find_all('tr')[1].find_all('td')]
    assert cells[:4] == ['A','B','A','65.0%']
    assert 'A: 60¢' in cells[4] and 'B: 41¢' in cells[4]
    assert 'bankroll × 0.625%' in tables[1].get_text(' ',strip=True)
    assert 'Unavailable' in tables[0].get_text()
    picks.GmailSender(gmail_config)(picks.format_messages(report)[0], html=html)
    message = smtp_server.messages[0]
    assert message.get_content_type() == 'multipart/alternative'
    assert '<table' in message.get_body(preferencelist=('html',)).get_content()
    assert 'A vs B: A 65.0%' in message.get_body(preferencelist=('plain',)).get_content()


def test_html_escapes_scraped_names(picks, report):
    report['event'] = '<img src=x onerror=alert(1)>'
    report['bouts'] = [('<script>', 'B')]
    report['rows'] = [('<script>','B',.7),('B','<script>',.3)]
    report['skipped'] = []
    html = picks.format_html(report)
    assert '<script>' not in html and '<img src=x' not in html
    assert '&lt;script&gt;' in html


def test_kalshi_failure_keeps_predictions_but_never_uses_sportsbook_stakes(picks, fake_predictor, monkeypatch):
    def fail(*args):
        raise requests.Timeout('unavailable')
    monkeypatch.setattr(picks.kalshi_odds, 'fetch_quotes', fail)
    report = picks.collect_reports(dt.date(2026,9,11))[0]
    assert len(report['rows']) == 2
    assert report['bets'] == [] and report['odds_map'] == {}
    assert 'Kalshi odds unavailable' in picks.format_html(report)
