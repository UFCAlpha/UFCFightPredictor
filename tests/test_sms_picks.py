"""Delivery tests use fake network boundaries; no SMS is sent."""
import datetime as dt
import importlib
import json
from types import SimpleNamespace

import pytest
import requests


@pytest.fixture
def sms():
    return importlib.import_module("sms_picks")


def test_friday_window_uses_eastern_time_in_summer_and_winter(sms):
    assert sms.in_send_window(dt.datetime.fromisoformat("2026-09-12T01:00:00+00:00"))
    assert sms.in_send_window(dt.datetime.fromisoformat("2026-12-05T02:00:00+00:00"))
    for stamp in ["2026-09-12T00:59:00+00:00", "2026-09-12T01:15:00+00:00",
                  "2026-09-13T01:00:00+00:00", "2026-12-05T01:00:00+00:00"]:
        assert not sms.in_send_window(dt.datetime.fromisoformat(stamp))


def test_selects_only_this_weekends_cards(sms):
    events = [(dt.datetime(2026, 9, day), f"url{day}", f"Card {day}")
              for day in [19, 13, 11, 12]]
    today = dt.date(2026, 9, 11)
    assert [e[1] for e in sms.weekend_events(events, today)] == ["url12", "url13"]
    assert sms.weekend_events(events[:1], today) == []
    assert sms.weekend_events(events, dt.date(2026, 9, 13)) == []


@pytest.fixture
def report():
    return dict(event="UFC Test", event_date="2026-09-12", event_url="url12",
                bouts=[("A", "B"), ("C", "D"), ("E", "F")],
                rows=[("A", "B", .7), ("B", "A", .4), ("C", "D", .4), ("D", "C", .6)],
                skipped=[(("E", "F"), "no UFC history in the dataset")],
                odds_map={("b", "a"): (130, -150)},
                bets=[dict(fighter="A", opponent="B", odds=-150, kelly=.125, stake_pct=.625)])


def test_message_averages_orientations_and_uses_final_stake_once(sms, report):
    text = "\n".join(sms.format_messages(report))
    assert "A vs B: A 65.0%" in text
    assert "C vs D: D 60.0%" in text
    assert "C vs D: odds unavailable" in text
    assert "E vs F: unavailable" in text
    assert "Kelly 12.50%" in text and "bankroll x 0.625%" in text
    assert "UFC Test" in text and "2026-09-12" in text


def test_no_odds_is_distinct_from_no_qualifying_bets(sms, report):
    report["bets"] = []
    assert "No bets qualify" in "\n".join(sms.format_messages(report))
    report["odds_map"] = {}
    text = "\n".join(sms.format_messages(report))
    assert "Odds unavailable; no stakes calculated" in text
    assert "No bets qualify" not in text


def test_long_unicode_card_is_split_without_losing_bouts(sms, report):
    report["bouts"] = [(f"Fighter {i} " + "🥊" * 50, f"Opponent {i}") for i in range(30)]
    report["rows"] = []
    report["bets"] = []
    report["skipped"] = [(bout, "no history") for bout in report["bouts"]]
    messages = sms.format_messages(report)
    assert len(messages) > 1
    assert all(len(m.encode("utf-16-le")) // 2 <= 1600 for m in messages)
    text = "\n".join(messages)
    for a, b in report["bouts"]:
        assert f"{a} vs {b}" in text


def test_delivery_is_persisted_and_duplicate_run_sends_nothing(sms, tmp_path):
    path = tmp_path / "delivery.json"
    calls = []
    def send(body):
        calls.append(body)
        return "SM" + "1" * 32
    assert sms.deliver(path, ["one", "two"], send) == "submitted"
    assert sms.deliver(path, ["changed forecast"], send) == "already submitted"
    assert calls == ["one", "two"]
    assert json.loads(path.read_text())["parts"][1]["sid"].startswith("SM")


def test_timeout_blocks_retries_and_preserves_accepted_parts(sms, tmp_path):
    path = tmp_path / "delivery.json"
    calls = []
    def send(body):
        calls.append(body)
        if body == "two":
            raise requests.Timeout("uncertain")
        return "SM" + "1" * 32
    with pytest.raises(requests.Timeout):
        sms.deliver(path, ["one", "two", "three"], send)
    with pytest.raises(RuntimeError, match="uncertain"):
        sms.deliver(path, ["new"], send)
    assert calls == ["one", "two"]
    state = json.loads(path.read_text())
    assert state["parts"][0]["status"] == "submitted"
    assert state["parts"][1]["status"] == "sending"
    assert state["parts"][2]["status"] == "pending"


def test_concurrent_delivery_cannot_double_send(sms, tmp_path):
    path = tmp_path / "delivery.json"
    def send(body):
        with pytest.raises(RuntimeError, match="running"):
            sms.deliver(path, [body], lambda _: pytest.fail("duplicate send"))
        return "SM" + "1" * 32
    assert sms.deliver(path, ["one"], send) == "submitted"


def test_twilio_request_and_response_are_validated(sms, monkeypatch):
    config = dict(TWILIO_ACCOUNT_SID="AC" + "1" * 32, TWILIO_AUTH_TOKEN="secret",
                  TWILIO_FROM_NUMBER="+14165550100", UFC_SMS_TO="+14165550101")
    def post(url, **kwargs):
        assert url == "https://api.twilio.com/2010-04-01/Accounts/AC" + "1" * 32 + "/Messages.json"
        assert kwargs["auth"] == (config["TWILIO_ACCOUNT_SID"], "secret")
        assert kwargs["data"] == {"From": "+14165550100", "To": "+14165550101", "Body": "hello"}
        assert kwargs["timeout"] == 30
        return SimpleNamespace(status_code=201, json=lambda: {"sid": "SM" + "2" * 32, "status": "queued"})
    monkeypatch.setattr(requests, "post", post)
    assert sms.TwilioSender(config)("hello") == "SM" + "2" * 32


@pytest.mark.parametrize("response", [
    SimpleNamespace(status_code=401, json=lambda: {"message": "private details"}),
    SimpleNamespace(status_code=201, json=lambda: {}),
    SimpleNamespace(status_code=201, json=lambda: {"sid": "SM" + "2" * 32, "status": "failed"}),
])
def test_twilio_failure_is_not_reported_as_success(sms, monkeypatch, response):
    config = dict(TWILIO_ACCOUNT_SID="AC" + "1" * 32, TWILIO_AUTH_TOKEN="secret",
                  TWILIO_FROM_NUMBER="+14165550100", UFC_SMS_TO="+14165550101")
    monkeypatch.setattr(requests, "post", lambda *a, **kw: response)
    with pytest.raises(RuntimeError) as err:
        sms.TwilioSender(config)("hello")
    assert "secret" not in str(err.value) and "private details" not in str(err.value)


def test_hold_and_outside_window_skip_before_prediction(sms, tmp_path, monkeypatch):
    monkeypatch.setattr(sms, "collect_reports", lambda *a: pytest.fail("predictions ran"))
    hold = tmp_path / "hold"
    hold.touch()
    assert sms.run(send=True, scheduled=False, state_dir=tmp_path, hold_file=hold) == 0
    hold.unlink()
    assert sms.run(send=True, scheduled=True, state_dir=tmp_path, hold_file=hold,
                   now=dt.datetime.fromisoformat("2026-09-12T02:00:00+00:00")) == 0


def test_dry_run_prints_without_credentials_or_delivery_state(sms, tmp_path, monkeypatch, report, capsys):
    monkeypatch.setattr(sms, "collect_reports", lambda *a: [report])
    assert sms.run(send=False, scheduled=False, state_dir=tmp_path / "state",
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
    monkeypatch.setattr(predict_event, "fetch_odds", lambda *a: {("a", "b"): (-150, 130)})
    return predict_event, scratch


def test_collection_uses_real_kelly_and_restores_feature_output(sms, fake_predictor):
    reports = sms.collect_reports(dt.date(2026, 9, 11))
    report = reports[0]
    assert report["event_date"] == "2026-09-12"
    assert report["bets"][0]["fighter"] == "A"
    assert report["bets"][0]["stake_pct"] == 1.95
    assert fake_predictor[1].output_csv_filename == "original.csv"


@pytest.mark.parametrize("rows", [[("A", "B", .8)], [("A", "B", float("nan")), ("B", "A", .2)]])
def test_bad_prediction_output_aborts(sms, fake_predictor, monkeypatch, rows):
    monkeypatch.setattr(fake_predictor[0], "predict_rows", lambda: rows)
    with pytest.raises(RuntimeError):
        sms.collect_reports(dt.date(2026, 9, 11))
    assert fake_predictor[1].output_csv_filename == "original.csv"


def test_empty_scrape_propagates_failure(sms, fake_predictor, monkeypatch):
    def fail(_):
        raise fake_predictor[0].ScrapeError("empty index")
    monkeypatch.setattr(fake_predictor[0], "upcoming_events", fail)
    with pytest.raises(fake_predictor[0].ScrapeError):
        sms.collect_reports(dt.date(2026, 9, 11))


def test_changed_card_aborts_before_network_send(sms, fake_predictor, report):
    with pytest.raises(RuntimeError, match="Card changed"):
        sms.verify_card(report)


def test_no_weekend_event_never_builds_features(sms, fake_predictor, monkeypatch):
    monkeypatch.setattr(fake_predictor[0], "upcoming_events", lambda _: [
        (dt.datetime(2026, 9, 19), "url19", "Next Week")])
    monkeypatch.setattr(fake_predictor[0], "build_features", lambda _: pytest.fail("built features"))
    assert sms.collect_reports(dt.date(2026, 9, 11)) == []


def test_partial_delivery_cannot_resume_a_different_forecast(sms, tmp_path):
    path = tmp_path / "delivery.json"
    path.write_text(json.dumps({"parts": [
        {"body": "old one", "status": "submitted", "sid": "SM" + "1" * 32},
        {"body": "old two", "status": "pending"}]}))
    with pytest.raises(RuntimeError, match="changed"):
        sms.deliver(path, ["new one", "new two"], lambda _: pytest.fail("stale send"))


def test_malformed_journal_cannot_trigger_a_resend(sms, tmp_path):
    path = tmp_path / "delivery.json"
    path.write_text(json.dumps({"parts": [{"body": "one", "status": "typo"}]}))
    with pytest.raises(RuntimeError, match="Invalid"):
        sms.deliver(path, ["one"], lambda _: pytest.fail("resend"))


def test_run_submits_complete_card_once_and_dry_run_does_not_consume_it(
        sms, fake_predictor, monkeypatch, tmp_path, capsys):
    for key, value in dict(TWILIO_ACCOUNT_SID="AC" + "1" * 32, TWILIO_AUTH_TOKEN="secret",
                           TWILIO_FROM_NUMBER="+14165550100", UFC_SMS_TO="+14165550101").items():
        monkeypatch.setenv(key, value)
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs["data"]["Body"])
        return SimpleNamespace(status_code=201, json=lambda: {"sid": "SM" + "2" * 32, "status": "queued"})
    monkeypatch.setattr(requests, "post", post)
    kwargs = dict(scheduled=False, state_dir=tmp_path / "receipts", hold_file=tmp_path / "hold",
                  now=dt.datetime.fromisoformat("2026-09-11T21:00:00-04:00"))
    sms.run(send=False, **kwargs)
    assert calls == []
    sms.run(send=True, **kwargs)
    sms.run(send=True, **kwargs)
    assert len(calls) == 1
    assert "A vs B: A 80.0%" in calls[0]
    assert "stake = bankroll x 1.95%" in calls[0]
    assert "already submitted" in capsys.readouterr().out


def test_wrapper_loads_local_config_and_defaults_to_scheduled_send(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    root = Path(__file__).resolve().parents[1]
    shutil.copy(root / "run_sms_scheduled.sh", tmp_path)
    interpreter = tmp_path / "fake-python"
    interpreter.write_text('#!/bin/bash\nif [ "$1" = "-c" ]; then exit 0; fi\nprintf "%s\\n" "$PWD" "$UFC_SMS_TO" "$@"\n')
    interpreter.chmod(0o700)
    (tmp_path / ".sms.env").write_text(f"UFC_PYTHON='{interpreter}'\nUFC_SMS_TO='+14165550101'\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("UFC_", "TWILIO_"))}
    result = subprocess.run(["/bin/bash", str(tmp_path / "run_sms_scheduled.sh")],
                            cwd="/tmp", env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(tmp_path), "+14165550101", str(tmp_path / "sms_picks.py"),
                                          "--send", "--scheduled"]
    result = subprocess.run(["/bin/bash", str(tmp_path / "run_sms_scheduled.sh"), "--dry-run"],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines()[-1] == "--dry-run"
    assert "--send" not in result.stdout


def test_installer_render_does_not_install_or_read_credentials(tmp_path, monkeypatch):
    import plistlib
    import setup_sms_launchd
    monkeypatch.setattr(setup_sms_launchd.subprocess, "run", lambda *a, **k: pytest.fail("installation attempted"))
    path = tmp_path / "job.plist"
    assert setup_sms_launchd.main(["--output", str(path)]) == 0
    job = plistlib.loads(path.read_bytes())
    assert job["ProgramArguments"] == ["/bin/bash", str(setup_sms_launchd.ROOT / "run_sms_scheduled.sh")]
    assert job["StartInterval"] <= 60
    assert "RunAtLoad" not in job and "KeepAlive" not in job


@pytest.mark.parametrize("change", ["hold", "window"])
def test_hold_or_window_change_during_card_fetch_prevents_send(sms, fake_predictor, monkeypatch, tmp_path, change):
    for key, value in dict(TWILIO_ACCOUNT_SID="AC" + "1" * 32, TWILIO_AUTH_TOKEN="secret",
                           TWILIO_FROM_NUMBER="+14165550100", UFC_SMS_TO="+14165550101").items():
        monkeypatch.setenv(key, value)
    hold = tmp_path / "hold"
    window_open = [True]
    monkeypatch.setattr(sms, "in_send_window", lambda _: window_open[0])
    def check(_):
        if change == "hold":
            hold.touch()
        else:
            window_open[0] = False
    monkeypatch.setattr(sms, "verify_card", check)
    monkeypatch.setattr(requests, "post", lambda *a, **kw: pytest.fail("SMS sent after hold/window changed"))
    with pytest.raises(RuntimeError, match="hold|window"):
        sms.run(send=True, scheduled=True, state_dir=tmp_path / "state", hold_file=hold,
                now=dt.datetime.fromisoformat("2026-09-11T21:00:00-04:00"))


def test_preflight_failure_can_retry_without_uncertain_twilio_delivery(sms, tmp_path):
    path = tmp_path / "delivery.json"
    def preflight():
        raise RuntimeError("card fetch failed")
    with pytest.raises(RuntimeError, match="card fetch failed"):
        sms.deliver(path, ["one"], lambda _: pytest.fail("submitted"), before_send=preflight)
    assert sms.deliver(path, ["one"], lambda _: "SM" + "1" * 32) == "submitted"


def test_rendered_job_preserves_custom_config_location(tmp_path, monkeypatch):
    import plistlib
    import setup_sms_launchd
    config = tmp_path / "private config.env"
    monkeypatch.setenv("UFC_SMS_CONFIG", str(config))
    output = tmp_path / "job.plist"
    setup_sms_launchd.main(["--output", str(output)])
    job = plistlib.loads(output.read_bytes())
    assert job["EnvironmentVariables"]["UFC_SMS_CONFIG"] == str(config)
