#!/usr/bin/env python3
"""Render or install the Friday SMS launchd job (macOS). No SMS is sent here."""
import argparse
from pathlib import Path
import plistlib
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parent
LABEL = "com.ufcpredictor.sms"


def job(repo):
    return {
        "Label": LABEL,
        "ProgramArguments": ["/bin/bash", str(repo / "run_sms_scheduled.sh")],
        "WorkingDirectory": str(repo),
        "EnvironmentVariables": {
            "UFC_SMS_CONFIG": str((repo / os.environ.get("UFC_SMS_CONFIG", ".sms.env")).resolve()),
        },
        # The Python gate uses America/Toronto, independent of the Mac's timezone.
        # Polling permits a short wake-from-sleep catch-up window, never a late SMS.
        "StartInterval": 60,
        "StandardOutPath": str(repo / "logs/sms_out.log"),
        "StandardErrorPath": str(repo / "logs/sms_error.log"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="render a plist for inspection without installing")
    mode.add_argument("--install", action="store_true", help="install and enable Friday SMS")
    args = parser.parse_args(argv)
    if args.output:
        args.output.write_bytes(plistlib.dumps(job(ROOT)))
        print(f"Wrote {args.output}; scheduler not installed.")
        return 0
    if sys.platform != "darwin":
        parser.error("launchd requires macOS; use the cron example in docs/sms-picks.md")
    config = Path(job(ROOT)["EnvironmentVariables"]["UFC_SMS_CONFIG"])
    if not config.is_file():
        parser.error("Create .sms.env or set UFC_SMS_CONFIG to a configuration file before installing")
    # Validate configuration before changing any existing job. This does not send.
    subprocess.run(["/bin/bash", str(ROOT / "run_sms_scheduled.sh"), "--check-config"], check=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    destination = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    destination.parent.mkdir(parents=True, exist_ok=True)
    domain = f"gui/{os.getuid()}"
    existing = subprocess.run(["launchctl", "print", f"{domain}/{LABEL}"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if existing.returncode == 0:
        subprocess.run(["launchctl", "bootout", f"{domain}/{LABEL}"], check=True)
    destination.write_bytes(plistlib.dumps(job(ROOT)))
    subprocess.run(["launchctl", "bootstrap", domain, str(destination)], check=True)
    print("Installed: Friday 21:00 America/Toronto (15-minute catch-up window).")
    print(f"Hold: touch {ROOT / 'data/sms.hold'}")
    print(f"Uninstall: launchctl bootout {domain}/{LABEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
