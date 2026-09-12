# Friday picks by Gmail (PAR-8)

`email_picks.py` emails each upcoming weekend UFC card's predictions and Kelly
stake percentages through your Gmail account. Each card is one complete email,
including unavailable predictions and missing odds. It uses the saved ensemble
and existing betting policy; it does not retrain or overwrite prediction exports.

## Configure Gmail

1. Enable [Google 2-Step Verification](https://support.google.com/accounts/answer/185833).
2. Create a [Google app password](https://myaccount.google.com/apppasswords), named
   `UFC Alpha`. This is a separate 16-character password, not your Google login
   password. App passwords may be unavailable on some managed or protected accounts.
3. From the permanent repository checkout that receives retrained models:

   ```bash
   cp .email.env.example .email.env
   chmod 600 .email.env
   ```

4. Edit `.email.env` locally:

   ```bash
   GMAIL_ADDRESS='your.account@gmail.com'
   GMAIL_APP_PASSWORD='your 16-character app password'
   UFC_EMAIL_TO=''  # blank sends to GMAIL_ADDRESS itself
   ```

   Replace the password placeholder with the actual app password. Spaces in
   Google's displayed app password are accepted. Set `UFC_EMAIL_TO` to a single
   different email address if desired. Do not paste the password into chat or
   commit this file. Configuration is sourced as shell code, so quote values.

The connection uses `smtp.gmail.com:465` with verified TLS. No additional Python
package is needed for SMTP. See [Google's SMTP settings](https://support.google.com/mail/answer/7104828).

## Preview and enable

Run from the repository root:

```bash
./run_email_scheduled.sh --check-config  # local format validation; no network
./run_email_scheduled.sh --dry-run       # fresh predictions and odds; no email
.venv/bin/python setup_email_launchd.py --output /tmp/ufc-email.plist
.venv/bin/python setup_email_launchd.py --install
```

Install against the permanent checkout, not an old development worktree. Keep
the existing Monday/Friday retrain job running against that same checkout.
The installer validates configuration format but does not authenticate Gmail
or send an email. To intentionally send the current weekend's picks immediately:

```bash
./run_email_scheduled.sh --send
```

That bypasses the Friday clock gate but respects the hold and duplicate protection.
The default command without arguments is `--send --scheduled`.

`UFC_EMAIL_CONFIG` can select a different configuration file. Set it when running
the installer as well: the generated job preserves the absolute path. Put all
credentials and interpreter overrides in the file, because launchd does not
inherit terminal environment variables. `.venv/bin/python` is preferred; set
`UFC_PYTHON` in the configuration if needed.

## Friday schedule and hold

The launchd job checks every minute and permits sending Friday **21:00–21:14
America/Toronto**, following daylight saving time. The first check after 9 PM
starts the run. The short catch-up window accommodates waking the Mac late;
a run missed entirely is not sent on Saturday. The Mac must be awake and the
user session active.

Only future Saturday/Sunday cards in the current week are selected. Friday
cards and same-day cards are excluded because UFCStats supplies dates, not
reliable start times. Off weeks send nothing.

The current card is fetched again before sending. Matchup changes abort the
send, but the script does not monitor weight misses, medical issues, or injury
news. Review those yourself and use the hold for an abnormal card:

```bash
touch data/email.hold   # stop previews and sends
rm data/email.hold      # release after reviewing the card
```

The hold remains active until removed and is checked again after Gmail login.
It cannot recall an email already submitted. Override the path with
`UFC_EMAIL_HOLD_FILE` if needed.

For a Linux cron host, create `logs/` and install this instead of launchd,
replacing the absolute repository path:

```cron
* * * * * /bin/bash /absolute/path/UFCFightPredictor/run_email_scheduled.sh >> /absolute/path/UFCFightPredictor/logs/email_cron.log 2>&1
```

The Python gate follows Toronto time independently of the host timezone.
Use only one scheduler and one delivery-state directory.

## Stakes and delivery

Predictions average both corner orientations. Bets retain the existing 80% model /
20% de-vigged market blend, minimum 5% edge, +200 maximum underdog price, 5%
fractional Kelly, 5% bankroll cap, and no minimum stake.

For example, `Kelly 12.50%; stake = bankroll x 0.625%` means $6.25 on a $1,000
bankroll. The stake percentage already includes fractional Kelly and the cap;
do not multiply it by 5% again. Use your current bankroll and check current odds
before betting. No account balance is stored or inferred from the paper ledger.

Private journals in `data/email_delivery/` prevent repeated sends for the same
event/date/recipient and lock out concurrent runs. Preserve this directory across
deployments; deleting receipts removes duplicate protection. `UFC_EMAIL_STATE_DIR`
can relocate it. Configuration, holds, and journals are ignored by Git.

Connection/login failures and explicit SMTP rejections remain retryable. An
interrupted or timed-out submission is marked uncertain and is not automatically
retried, since Gmail may already have accepted it. Successful submission records
a Message-ID for correlation; it is not proof of inbox delivery. Check Gmail's
Sent folder, the destination inbox, and any bounce messages.

If delivery is uncertain, enable the hold and inspect the receipt and Gmail's
Sent folder. If accepted, mark the entry `submitted` and record its `message_id`.
Only reset it to `pending` when you establish it was not accepted. An absent
message alone may not prove that. Preserve accepted receipts. Then release the
hold and rerun when appropriate. Forecasts can refresh after a known rejection;
accepted forecasts are not resent merely because odds or probabilities changed.

```bash
tail -n 50 logs/email_error.log
launchctl print gui/$(id -u)/com.ufcpredictor.email
launchctl bootout gui/$(id -u)/com.ufcpredictor.email  # disable
```

This replaces the earlier Twilio implementation. No SMS scheduler was installed
during development. If you independently installed `com.ufcpredictor.sms`, disable
it before installing this job. macOS and Linux are supported; Windows is not.
