#!/usr/bin/env python3
"""Scrape new fights, rebuild features, retrain the ensemble, validate, publish.

Run by launchd twice a week (see setup_launchd.sh). Every step either succeeds
or raises: a step that produces nothing is a failure, not a quiet no-op. The
previous version of this pipeline logged "0 new fights added" and exited 0 for
two and a half years while the scraper was being served an anti-bot page, so
"nothing happened" is never treated as success here.

    python auto_retrain.py                      full pipeline
    python auto_retrain.py --skip-training      scrape and process only
    python auto_retrain.py --skip-scrape        rebuild and retrain existing data
    python auto_retrain.py --force-full-scrape  rescrape every event
    python auto_retrain.py --dry-run            preview, change nothing
"""
import argparse
import datetime
import glob
import json
import logging
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scrapers"))

LOG_DIR = os.path.join(ROOT, "logs")
MODEL_DIR = os.path.join(ROOT, "saved_models")
PREP_DIR = os.path.join(ROOT, "saved_preprocessing")
FEATURES = os.path.join(ROOT, "data", "detailed_fights.csv")

MIN_ACCURACY = 0.60      # documented gate: below this the new models are rejected
FEATURE_SEED = 42        # process_fights_alpha.py picks red/blue at random

log = logging.getLogger("auto_retrain")


# ---------------------------------------------------------------- infrastructure

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"auto_retrain_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    log.setLevel(logging.INFO)
    for handler in (logging.FileHandler(path), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return path


def banner(title):
    log.info("=" * 70)
    log.info(title)
    log.info("=" * 70)


def run_script(script, seed=None):
    """Run a repo script in a subprocess; raise if it fails."""
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=ROOT)
    if seed is None:
        cmd = [sys.executable, script]
    else:
        cmd = [sys.executable, "-c",
               f"import random, runpy; random.seed({seed}); "
               f"runpy.run_path({script!r}, run_name='__main__')"]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        for line in tail:
            log.error(f"  {line}")
        raise RuntimeError(f"{script} exited {proc.returncode}")
    return proc.stdout


# ---------------------------------------------------------------------- steps

def step_scrape(force_full, dry_run):
    banner("STEP 1: SCRAPING NEW FIGHTS")
    import scrape_incremental
    n = scrape_incremental.run(dry_run=dry_run, full=force_full, log=log.info)
    log.info(f"✓ Scraping completed: {n} new fights added")
    return n


def step_process():
    banner("STEP 2: PROCESSING FIGHT DATA")
    from utils import incremental_processing
    df = incremental_processing.build(log=log.info)
    log.info(f"✓ Processing completed: {len(df)} rows")


def step_grade_ledger():
    """Settle public-ledger picks whose results just landed. Never fails the run."""
    banner("STEP 2b: GRADING THE BET LEDGER")
    try:
        import bet_ledger
        n = bet_ledger.grade()
        log.info(f"✓ Ledger graded: {n} entries settled")
    except Exception as exc:
        log.warning(f"Ledger grading skipped: {type(exc).__name__}: {exc}")


def step_features():
    banner("STEP 3: FEATURE ENGINEERING")
    before = os.path.getmtime(FEATURES) if os.path.exists(FEATURES) else 0
    run_script(os.path.join(ROOT, "process_fights_alpha.py"), seed=FEATURE_SEED)
    if not os.path.exists(FEATURES) or os.path.getmtime(FEATURES) <= before:
        raise RuntimeError("process_fights_alpha.py did not rewrite detailed_fights.csv")
    import pandas as pd
    df = pd.read_csv(FEATURES, low_memory=False)
    log.info(f"✓ Features rebuilt: {len(df)} fights, {len(df.columns)} columns")


def backup_models():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(MODEL_DIR, f"backup_{stamp}")
    os.makedirs(dest, exist_ok=True)
    copied = 0
    for path in glob.glob(os.path.join(MODEL_DIR, "*.joblib")):
        shutil.copy2(path, dest)
        copied += 1
    prep = os.path.join(dest, "saved_preprocessing")
    if os.path.isdir(PREP_DIR):
        shutil.copytree(PREP_DIR, prep, dirs_exist_ok=True)
        copied += len(os.listdir(prep))
    if copied == 0:
        raise RuntimeError("nothing to back up — saved_models/ is empty")
    log.info(f"✓ Backed up {copied} files to {os.path.relpath(dest, ROOT)}")
    return dest


def restore_models(backup):
    for path in glob.glob(os.path.join(backup, "*.joblib")):
        shutil.copy2(path, MODEL_DIR)
    prep = os.path.join(backup, "saved_preprocessing")
    if os.path.isdir(prep):
        shutil.copytree(prep, PREP_DIR, dirs_exist_ok=True)
    # Older backups predate the manifest. Never retain metadata from a failed
    # refit while restoring an older model generation.
    manifest = os.path.join(PREP_DIR, "training_manifest.json")
    if not os.path.exists(os.path.join(prep, "training_manifest.json")) and os.path.exists(manifest):
        os.remove(manifest)
    log.info(f"✓ Restored models from {os.path.relpath(backup, ROOT)}")


def step_train():
    banner("STEP 4: MODEL TRAINING")
    backup = backup_models()
    # ml_ensemble.py dumps its five models one at a time at the very end, so a
    # crash partway through leaves a mixed set on disk. Always roll back.
    try:
        run_script(os.path.join(ROOT, "ml_ensemble.py"))
        models = glob.glob(os.path.join(MODEL_DIR, "lgbm_model_*.joblib"))
        if not models:
            raise RuntimeError("training wrote no models")
    except Exception:
        restore_models(backup)
        raise
    log.info(f"✓ Training completed: {len(models)} models written")
    return backup


def evaluate_saved_ensemble():
    """Evaluate candidates only; production refits have already seen the holdout."""
    if os.path.exists(os.path.join(PREP_DIR, "training_manifest.json")):
        from production_refit import load_candidate_evaluation
        report = load_candidate_evaluation(FEATURES, MODEL_DIR, PREP_DIR)
        return report["evaluation"]["accuracy"], report["evaluation"]["n"]
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score

    label_encoder = joblib.load(os.path.join(PREP_DIR, "label_encoder.joblib"))
    with open(os.path.join(PREP_DIR, "selected_columns.json")) as fh:
        selected = json.load(fh)
    features = [c for c in selected if c != "Result"]

    paths = sorted(glob.glob(os.path.join(MODEL_DIR, "lgbm_model_*.joblib")))
    models = [joblib.load(p) for p in paths]

    df = pd.read_csv(FEATURES, low_memory=False)
    holdout = df.iloc[int(len(df) * 0.95):]
    probs = np.mean([m.predict_proba(holdout[features]) for m in models], axis=0)
    y = label_encoder.transform(holdout["Result"])
    return accuracy_score(y, probs.argmax(axis=1)), len(holdout)


def step_validate(backup):
    banner("STEP 5: VALIDATION")
    try:
        accuracy, n = evaluate_saved_ensemble()
        log.info(f"Candidate holdout accuracy: {accuracy:.4f} over {n} fights "
                 f"(threshold {MIN_ACCURACY:.2f})")
        if not 0 <= accuracy <= 1 or accuracy < MIN_ACCURACY:
            raise RuntimeError(f"validation failed: {accuracy:.4f} < {MIN_ACCURACY:.2f}")
    except Exception:
        restore_models(backup)
        raise
    log.info("✓ Candidate validation passed — ready for production refit")
    return accuracy


def step_refit(backup):
    banner("STEP 5b: PRODUCTION REFIT INCLUDING LATEST COMPLETED FIGHTS")
    from production_refit import refit_production_ensemble
    try:
        return refit_production_ensemble(FEATURES, MODEL_DIR, PREP_DIR,
                                        min_accuracy=MIN_ACCURACY, log=log.info)
    except Exception:
        restore_models(backup)
        raise


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-training", action="store_true", help="scrape and process only")
    ap.add_argument("--skip-scrape", action="store_true", help="use the data already on disk")
    ap.add_argument("--force-full-scrape", action="store_true", help="rescrape every event")
    ap.add_argument("--dry-run", action="store_true", help="preview without changing anything")
    args = ap.parse_args()

    log_path = setup_logging()
    started = datetime.datetime.now()
    log.info("╔" + "=" * 68 + "╗")
    log.info("║" + "UFC FIGHT PREDICTOR AUTO-RETRAIN".center(68) + "║")
    log.info("║" + f"Started: {started:%Y-%m-%d %H:%M:%S}".center(68) + "║")
    log.info("╚" + "=" * 68 + "╝")

    try:
        if args.skip_scrape:
            banner("STEP 1: SCRAPING NEW FIGHTS")
            log.info("↷ Skipped (--skip-scrape); using data already on disk")
            new_fights = None
        else:
            new_fights = step_scrape(args.force_full_scrape, args.dry_run)

        if args.dry_run:
            banner("DRY RUN - no further steps")
            log.info(f"✓ Preview complete. Log file: {os.path.relpath(log_path, ROOT)}")
            return 0

        if new_fights == 0:
            banner("NO NEW FIGHTS - Skipping remaining steps")
            log.info("Data is current; nothing to rebuild.")
            banner("STEP 6: NOTIFICATION")
            log.info("✓ Auto-retraining success: no new fights to process")
            log.info(f"Log file: {os.path.relpath(log_path, ROOT)}")
            return 0

        step_process()
        step_grade_ledger()
        step_features()

        if args.skip_training:
            banner("STEP 4: MODEL TRAINING")
            log.info("↷ Skipped (--skip-training)")
            accuracy = None
        else:
            backup = step_train()
            accuracy = step_validate(backup)
            step_refit(backup)

        banner("STEP 6: NOTIFICATION")
        summary = "✓ Auto-retraining success"
        if new_fights:
            summary += f": {new_fights} new fights"
        if accuracy is not None:
            summary += f", pre-refit candidate holdout accuracy {accuracy:.4f}"
        log.info(summary)
        log.info(f"Elapsed: {datetime.datetime.now() - started}")
        log.info(f"Log file: {os.path.relpath(log_path, ROOT)}")
        return 0

    except Exception as exc:
        banner("STEP 6: NOTIFICATION")
        log.error(f"✗ Auto-retraining FAILED: {type(exc).__name__}: {exc}")
        log.error(f"Log file: {os.path.relpath(log_path, ROOT)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
