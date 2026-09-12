# Friday picks by SMS (PAR-8)

`sms_picks.py` generates predictions for this week's upcoming Saturday/Sunday
UFC cards using the saved ensemble, then sends the predictions and Kelly stakes
through Twilio. It does not retrain. Keep the existing Monday/Friday retrain job
running against the same checkout and model files.

Automatic sending starts Friday at **21:00 America/Toronto**, following daylight
saving time (EST in winter, EDT in summer). A Mac launchd job checks every minute;
the script only accepts scheduled runs from 21:00 through 21:14. This gives a
short catch-up window if the Mac wakes late. Outside that window it does nothing.
The Mac must be awake and the user session active. A run missed entirely is not
sent on Saturday. Cards dated Friday or already past are excluded because the
source supplies dates, not reliable start times.

## Setup

Install in the permanent checkout that receives retrained models, not an obsolete
development worktree. Run from the repository root:

```bash
cp .sms.env.example .sms.env
chmod 600 .sms.env
```

Edit `.sms.env` locally with your Twilio account SID, auth token, SMS-capable
sender, and destination phone number. Both phone numbers include `+` and country
code. This is a shell configuration file; quote values. Do not commit it.
`UFC_SMS_CONFIG` can point the wrapper at a different file. Set it when running
the installer too; the installed job preserves its absolute path. Put credentials
and interpreter overrides in that file, since launchd does not inherit your
terminal's environment. `.venv/bin/python`
is preferred; set `UFC_PYTHON` if necessary.

```bash
./run_sms_scheduled.sh --check-config  # format check only, no network or SMS
./run_sms_scheduled.sh --dry-run       # fresh predictions/odds printed; no SMS
.venv/bin/python setup_sms_launchd.py --output /tmp/ufc-sms.plist  # inspect job
.venv/bin/python setup_sms_launchd.py --install                  # enable job
```

The installer validates configuration format before replacing an existing job.
It does not authenticate credentials or send a test message. Install only one
scheduler against one checkout and one delivery-state directory.

For a Linux cron host with the pipeline dependencies, use the same wrapper
instead of launchd (replace the repository path):

```cron
* * * * * /bin/bash /absolute/path/UFCFightPredictor/run_sms_scheduled.sh >> /absolute/path/UFCFightPredictor/logs/sms_cron.log 2>&1
```

Create `logs/` first. The Python gate handles Toronto time regardless of the cron
host's timezone. Run `./run_sms_scheduled.sh --send` only when you intentionally
want an immediate manual send; it bypasses the Friday clock gate but still uses
future cards in this week's weekend, the hold file, and duplicate protection.

## Weigh-ins and the hold switch

Sending is automatic. The script fetches the current UFCStats card and checks
that the matchups have not changed immediately before each SMS submission. It
does **not** certify weigh-ins, detect medical issues, or monitor injury news.
Review those yourself and use the hold switch for an abnormal card:

```bash
touch data/sms.hold   # prevents previews and sends
rm data/sms.hold      # release after reviewing the card
```

The hold remains active until removed. `UFC_SMS_HOLD_FILE` overrides its location.
A hold added after one part was accepted cannot recall that part. Release during
the Friday window allows a later scheduler check to run; otherwise wait until
next Friday or explicitly use `--send`.

## Message contents and bankroll

Every modeled bout appears once, with the probability averaged across both
red/blue orientations. Bouts that cannot be modeled (including women's bouts
and fighters with insufficient history) are labeled unavailable with a reason.
Missing odds are explicit and distinct from priced fights with no qualifying bet.

Bets use the existing `predict_event.recommend` / `betting_math` policy unchanged:
80% model / 20% de-vigged market blend, minimum 5% edge, no prices longer than
+200, 5% fractional Kelly, 5% bankroll cap, and no minimum stake.
For example:

```text
A vs B: A 65.0%
A vs B -150: Kelly 12.50%; stake = bankroll x 0.625%
```

For a $1,000 bankroll the second line means $6.25. The stake percentage already
includes the Kelly fraction and cap: do not multiply by 5% again. Use your current
bankroll at the time of betting. No balance is stored or inferred from the paper
ledger. Odds are a generation-time snapshot, not a promised execution price.

Long reports are numbered parts. Twilio may further segment each part and bill
for the segments. The integration uses Twilio's
[Messages API](https://www.twilio.com/docs/messaging/api/message-resource).
An accepted submission is logged as **submitted**, not delivered; delivery
confirmation and failures are visible in the Twilio console.

## State, failures, and recovery

`data/sms_delivery/` contains private delivery journals (message bodies, status,
and Twilio SIDs). `.sms.env`, journals, and the hold file are ignored by Git.
`UFC_SMS_STATE_DIR` can relocate the journal directory. Preserve it across deploys:
deleting journals removes duplicate protection. Do not use separate journals
for multiple schedulers sending to the same recipient.

Each event/date/recipient has one journal. Accepted parts are not sent again.
Concurrent runs are locked out. Fresh prediction features use a temporary file;
the SMS job does not overwrite prediction exports or record hypothetical bets
in the live ledger.

Any scrape or inference failure aborts without using cached forecasts. Missing
odds permit predictions with an explicit unavailable notice. A card changed
while predictions were being generated aborts sending. A partial delivery is
never resumed against changed message contents.

Card checks and hold/window checks run before recording outbound intent; their
failures can be retried by the next scheduler check without an uncertain receipt.
Before each Twilio request the journal records `sending`. If a timeout, crash, or provider
error interrupts it, that part remains uncertain and **is not
automatically retried**. This deliberately requires review even for explicit
provider rejections. To recover:

1. Enable the hold and inspect `logs/sms_error.log` and the event journal.
2. Check Twilio's messaging logs. If accepted, set that part to `submitted` and
   add its `sid`. If you establish no request was accepted, set it to `pending`.
   Preserve other accepted parts and their bodies.
3. Release the hold and rerun. If the forecast changed after partial submission,
   inspect the differences before making any manual correction; do not erase
   accepted receipts to force a resend.

A failed write before a POST prevents sending; a crash after a POST can leave
an uncertain receipt. This avoids claiming exactly-once network delivery.

```bash
tail -n 50 logs/sms_error.log
launchctl print gui/$(id -u)/com.ufcpredictor.sms
launchctl bootout gui/$(id -u)/com.ufcpredictor.sms  # disable the job
```

The journal lock uses `flock` and supports macOS/Linux. Windows is not supported.
