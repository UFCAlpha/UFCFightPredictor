#!/usr/bin/env bash
# Cron/launchd entry point. No arguments: send during Friday's Eastern window.
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
CONFIG="${UFC_SMS_CONFIG:-$REPO/.sms.env}"
if [ -f "$CONFIG" ]; then
    set -a
    source "$CONFIG"
    set +a
fi

resolve_python() {
    local candidate
    for candidate in "${UFC_PYTHON:-}" "$REPO/.venv/bin/python" /Users/aalex_xuu/anaconda3/bin/python3; do
        [ -n "$candidate" ] && [ -x "$candidate" ] || continue
        if "$candidate" -c 'import pandas, lightgbm, sklearn, bs4, requests' 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
        echo "SMS: configured interpreter is missing pipeline dependencies; skipping" >&2
    done
    return 1
}
if ! PY="$(resolve_python)"; then
    echo "SMS: no usable Python. Create .venv or set UFC_PYTHON in .sms.env." >&2
    exit 1
fi
if [ "$#" -eq 0 ]; then
    set -- --send --scheduled
fi
exec "$PY" "$REPO/sms_picks.py" "$@"
