# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

UFC Fight Predictor is a machine learning system that predicts UFC fight outcomes and recommends bets using the Kelly Criterion. Website: https://ufcalpha.com/ (frontend on Cloudflare Pages, API on Render)

The tier-2 walk-forward export over 2024-01-01 → 2026-08-30 scores 756 fights at 66.4% accuracy (AUC 0.695) and turns 170 bets into +26.6% on a $1,000 paper bankroll with an 11.5% max drawdown. The public site (`frontend/`) recomputes every window from `frontend/src/data/backtest.json`.

## Key Commands

### Auto-Retraining Pipeline (Full Automation)
```bash
python auto_retrain.py                      # scrape → process → train → validate → production refit
python auto_retrain.py --skip-training      # Scrape and process only
python auto_retrain.py --skip-scrape        # Rebuild and retrain from data already on disk
python auto_retrain.py --force-full-scrape  # Rescrape every event from scratch
python auto_retrain.py --dry-run            # Preview without changing anything
```

### Individual Pipeline Components
```bash
python scrapers/scrape_incremental.py       # Scrape new fights (also updates the fighter DB)
python scrapers/update_fighters.py          # Fill in fighters missing from the DB
python utils/incremental_processing.py      # Clean raw fights → modified_fight_details.csv
python process_fights_alpha.py              # Feature engineering (180+ features)
python ml_ensemble.py                       # Train/save evaluation candidates (no final production refit)
```

`ml_alpha_date.py` trains a single model and prints metrics, but **never writes to
`saved_models/`** and ends in a blocking `plt.show()`. It is an exploration script.
`ml_ensemble.py` produces the evaluation candidates. Use `auto_retrain.py`
for production: after validation it runs `production_refit.py`'s refit step,
including the newest eligible completed fights in the saved ensemble.

### Predictions & Testing
```bash
python predict_event.py                     # Predict the next event
python predict_event.py --list              # Show the upcoming schedule
python predict_event.py --event <url>       # Predict a specific ufcstats event
python predict_event.py --odds              # Also pull odds and size bets with Kelly
python betting_alpha.py                     # Betting recs from data/betting_predictions.csv
cd testing && python testing_time_period.py # Run backtesting
```

### Flask API & Frontend
```bash
python app.py                               # Start Flask server (localhost:5000)
cd frontend && npm start                    # Start React dev server (localhost:3000)
cd frontend && npm run build                # Production build
cd frontend && CI=true npx react-scripts test --watchAll=false   # Frontend tests

# Public site data: replay the walk-forward cache into the JSON the frontend bundles,
# then commit and push so Cloudflare Pages rebuilds. publish_site.py does all three and
# runs automatically after every scheduled retrain (see run_scheduled.sh); it commits only
# frontend/src/data, and only when the content changed (the export timestamp alone never
# triggers a commit). It must be run on main.
python publish_site.py                      # export → commit → push
python publish_site.py --no-push            # same, commit locally only
python publish_site.py --roll               # also scrape new odds and extend the window (below)

# By hand, the same three steps are:
python testing/export_site_data.py
git add frontend/src/data && git commit -m "Refresh site data" && git push

# The export reads test_results/.tier2_full_cache/ (one pred_YYYY-MM-DD.csv per walk-forward
# retrain, 2024-01-01 → 2026-08-30, committed). To extend or rebuild it:
python testing/build_walk_forward_cache.py --start 2024-01-01 --end 2026-08-30 --cache test_results/.tier2_full_cache
# The newest pred_*.csv only covers fights that existed when it was built and the builder
# reuses existing files, so to extend the window delete the newest one first, then rebuild
# and export with the new --end. `publish_site.py --roll` does this after running
# scrapers/scrape_new_odds.py and re-sorting the odds CSV; it skips the rebuild when no
# newer priced fight appeared. The window is otherwise fixed: the scheduled run never rolls it.
# Before launch, replace MEMBERSHIP_URL in frontend/src/constants.js (an inert example.com
# placeholder) with the real Patreon or Discord invite; /join is the only page that links out to it.
```

### Scheduler Management (macOS launchd)
```bash
./setup_launchd.sh                          # Interactive launchd setup
tail -f logs/launchd_error.log              # Scheduler-level failures (job never started)
ls -t logs/auto_retrain_*.log | head -1 | xargs cat  # Latest run log
ls -t logs/publish_site_*.log | head -1 | xargs cat  # Latest site publish log
```

The launchd job runs `run_scheduled.sh`, which runs `auto_retrain.py` and then, only if
that succeeded, `publish_site.py`. launchd starts with a minimal PATH, so the script
resolves the interpreter explicitly rather than trusting `python3` (which would find
`/usr/bin/python3` and none of the deps): `$UFC_PYTHON`, then `.venv/bin/python`, then a
pinned Anaconda interpreter as a last resort, taking the first that can import the pipeline
packages. `.venv` is the one that should win — it is the isolated environment the scheduled
retrain trains in, so an unrelated `conda install` cannot shift the library versions under
it. If no candidate imports, the script exits 1 with the reason rather than running the
pipeline against a broken environment.

Rebuild `.venv` with:

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # then: .venv/bin/pip install pytest
```

The push uses the `origin` SSH remote, so the launchd session needs a working key (check
`ssh -T git@github.com` once from a fresh login); the key at `~/.ssh/id_ed25519` is
unencrypted and pinned in `~/.ssh/config`, so the push works without an ssh-agent.

**A scheduled run with no new events never reaches training.** `auto_retrain.py` stops
after step 1 when the scrape returns 0 fights, which is the common case between events. To
exercise the train-and-validate path on demand, use `--skip-scrape`.

## Architecture

### Data Pipeline Flow
```
ufcstats.com → scrapers/scrape_incremental.py → data/fight_details_date.csv
    → utils/incremental_processing.py → data/modified_fight_details.csv
    → process_fights_alpha.py → data/detailed_fights.csv (180+ features)
    → ml_ensemble.py → saved_models/lgbm_model_*.joblib + saved_preprocessing/
    → predict_event.py → data/predicted_data.json
    → app.py (Flask API) → frontend/ (React)
```

### Key Files by Function
- **Orchestration**: `auto_retrain.py` — runs the full pipeline on schedule, with backup, validation and rollback
- **Publishing**: `publish_site.py` — re-exports `frontend/src/data`, commits it if the content changed, pushes main; `--roll` extends the walk-forward window. `run_scheduled.sh` chains it after the retrain
- **HTTP**: `scrapers/ufcnet.py` — ufcstats session that clears the anti-bot challenge
- **Scraping**: `scrapers/scrape_incremental.py` — detects last stored date, scrapes only newer events
- **Fighters**: `scrapers/update_fighters.py` — adds debutants to `instance/detailedfighters.db`
- **Processing**: `utils/incremental_processing.py` — data cleaning, format conversion
- **Features**: `process_fights_alpha.py` — ELO, per-minute stats, weighted averages
- **Training**: `ml_ensemble.py` — tunes/saves five evaluation candidates, preprocessing, calibration, and a hash-bound holdout report
- **Production refit**: `production_refit.py` — refits validated settings on all recent eligible completed fights, including the former holdout; calibrates on chronological out-of-fold predictions
- **Prediction**: `predict_event.py` — event predictions with Kelly betting
- **API**: `app.py` — Flask REST endpoints

### Model Files
- `saved_models/lgbm_model_0.joblib` … `lgbm_model_4.joblib` — five ensemble members. They are **not** corner-specific: each is trained on the same augmented data with its own Optuna-sampled hyperparameters, and inference averages `predict_proba` across all five (see `load_ensemble.py`).
- `saved_preprocessing/calibrator.joblib` — the temperature calibrator `load_ensemble.py` applies at inference; members are trained with distinct seeds
- `saved_preprocessing/training_manifest.json` — candidate holdout metrics, dataset/artifact hashes, and final production training dates/counts. Metrics are explicitly **pre-refit**; never evaluate the refitted production models on their former holdout as if it were unseen.
- `saved_preprocessing/selected_columns.json` — the exact feature list the models expect; **always** select columns through this rather than recomputing the prune.
- `saved_preprocessing/label_encoder.joblib` — classes are `['loss', 'win']`, so **class 1 = the red corner wins**.
- `saved_models/backup_YYYYMMDD_HHMMSS/` — timestamped backups written before each retrain, including a copy of `saved_preprocessing/`.

### Data Files
- `data/fight_details_date.csv` — raw scraped fights, stored **newest-first**; new rows are prepended
- `data/modified_fight_details.csv` — cleaned/processed fights (also newest-first)
- `data/detailed_fights.csv` — feature-engineered dataset, oldest-first (full recompute each run)
- `data/detailed_fighter_stats.csv` — per-fighter career state, read by `predict_fights_alpha.py`
- `data/predicted_data.json` — latest predictions; served by `/get_predicted_data`
- `data/fight_results_with_odds.csv` — historical closing odds through 2026-08-29; must stay sorted
  oldest-first because the backtests walk it in file order (re-sort after any backfill)
- `data/bet_ledger.json` — live picks written by `predict_event.py --odds`, graded by
  `auto_retrain.py`; copied into the site by `testing/export_site_data.py`
- `test_results/.tier2_full_cache/` — the walk-forward prediction cache the site export reads (six `pred_YYYY-MM-DD.csv`, one per retrain, 2024-01-01 → 2026-08-30); rebuild with `testing/build_walk_forward_cache.py`

## Important Patterns

### ufcstats.com serves an anti-bot challenge
Every page is fronted by a JavaScript proof-of-work interstitial: find `n` such that
`sha256(f"{nonce}:{n}")` starts with a run of zeros, POST it to `/__c`, then retry. One
cookie then covers the rest of the session. `scrapers/ufcnet.py` handles this; always
fetch ufcstats pages through it rather than calling `requests.get` directly.

**A parse that yields nothing is a failure, not "no new data."** This is the single most
important invariant in the scraper. The pre-2026 version of this pipeline used a plain
`requests.get`, parsed the challenge page, found zero rows, and logged
`✓ Scraping completed: 0 new fights added` on every scheduled run for two and a half
years. Keep raising `ScrapeError` on empty parses; "already up to date" must only be
reachable after successfully parsing the event index.

### Calibration matters more than accuracy
For betting, the useful property is that higher confidence means more often right — that
is what Kelly sizing consumes. A stale model degrades here long before its headline
accuracy moves. When the models were last left untrained for two years, accuracy held
around 64% while the 70%+ confidence band fell to 64% — no better than its weakest picks.
After the February 2026 retrain, accuracy on that year's holdout was flat (62.9%) but the bands became monotonic again and the
70%+ band reached 79%; the current headline numbers above come from the later tier-2 model over 2024–26. **Judge a retrain on AUC, log loss, Brier and the calibration
bands, not on accuracy alone.**

### Feature Engineering
For each fighter stat, generates: per-minute rates, accuracy %, differentials, weighted
moving averages (recent fights weighted higher), career totals. Features for a fight use
only bouts that preceded it, so there is no lookahead.

`process_fights_alpha.py` chooses each fight's red/blue orientation with `random.choice`,
so it is **not reproducible unless seeded**. It seeds itself at 42 (line 7) and
`auto_retrain.py` seeds it again through its runner; `data/detailed_fights.csv` is
byte-identical across runs as a result. The two fighter-stat exports are ordered by
`sorted(all_fighters)` for the same reason — set iteration order follows `PYTHONHASHSEED`,
and without the sort every run rewrote both files in a new order and left the repo dirty
after each scheduled retrain.

### Red/Blue symmetry is a hard requirement
Training augments every row with a Red↔Blue-swapped copy, so the retained feature set must
be closed under that swap. The correlation prune in `ml_ensemble.py` is order-dependent and
will otherwise drop only one side of a pair; `pd.concat` then unions the two frames and
**silently fills the orphaned columns with NaN**. `ml_ensemble.py` now drops both halves of
any such pair and asserts that mirroring did not change the column set. Do not remove that
guard.

### Betting Strategy
Fractional Kelly with conservative defaults; all sizing paths (`predict_event.py`,
`betting_alpha.py`, the backtests) route through `betting_math.py`:
- 5% Kelly fraction, 5% max cap, **no floor** — Kelly ≤ 0 means no bet
- Requires ≥5% edge, measured against the **de-vigged** market probability of the chosen side
- The betting probability blends model and market: 0.8 × (two-orientation model average)
  + 0.2 × de-vigged market prob. The displayed headline probability stays the plain model
  average; `closerToOdds` survives only as a legacy comparison path in the backtests
  (`strategy[4] = None`)
- Picks priced longer than +200 are skipped (`max_dog_odds`, adopted 2026-09 after the 2024-26 walk-forward study)

### Auto-Retraining System
Runs Monday & Friday at 2:00 AM via launchd:
1. Incremental scrape of new events; new fighters added to the fighter DB
2. New rows prepended to `fight_details_date.csv` (backup written first); pending entries in
   `data/bet_ledger.json` are graded against the new results (a grading error is logged, never fatal)
3. Full feature recomputation (needed for ELO consistency)
4. Backup of `saved_models/` + `saved_preprocessing/`, then training via `ml_ensemble.py`
5. Validation of the candidate ensemble on its chronological holdout (the last 5% of
   fights, excluded from candidate fitting). Accuracy below 60% rejects the candidate;
   validation exceptions also restore the backup. `ml_ensemble.py` alone saves candidates.
   After passing, **step 5b refits production on the latest eligible completed fights,
   including that former holdout**, with the selected columns, hyperparameters, tree
   counts, and seeds frozen. The existing oldest-history exclusion is preserved.
   Production calibration uses expanding chronological folds, keeping each fight date
   together. Failures restore the previous models and preprocessing. Holdout scores in
   the manifest describe the pre-refit candidate, not the refitted production model.
   Current fighter statistics already include all processed bouts through the latest
   event; the final refit additionally lets the model learn from those recent outcomes.
6. `publish_site.py` (via `run_scheduled.sh`, only after a successful retrain) re-exports the
   site data, commits `frontend/src/data` if anything but the timestamp changed, and pushes
   main so Cloudflare Pages rebuilds. It never rolls the backtest window on its own.

Any step that fails, or produces nothing, exits non-zero.

### Known quirks
- The cleaning step drops every `Title` containing "Women", so **women's bouts are absent
  from training and cannot be predicted**. `predict_event.py` reports them as fighters with
  no history.
- `Time Format` and `Details` are empty for every row: ufcstats labels them differently and
  the original scraper never captured them. New rows match this on purpose.
- The `shap` block in `ml_ensemble.py` is optional diagnostics, guarded by a try/except so
  a shap-less environment skips it. It *is* pinned in `requirements.txt` and installed in
  `.venv`, so the block does run on a scheduled retrain; `summary_plot` is called with
  `show=False` and `auto_retrain.py` forces `MPLBACKEND=Agg`, so nothing blocks on a render.

### Leakage fixes to preserve
Several are load-bearing and easy to undo by accident:
- `ml_ensemble.py` computes the correlation matrix on **training rows only**; computing it
  over the full frame lets feature selection see the test set.
- Optuna CV runs on the **un-augmented** training set. With the mirrored copies appended,
  `TimeSeriesSplit` folds validate on swapped duplicates of fights already trained on.
- `predict_fights_alpha.py` divides differential features by `sqrSum(totalfights)` exactly
  as `process_fights_alpha.py` does at training time. These drifted apart once and produced
  training/serving skew.
- `tests/test_no_data_leakage.py` and `validation/` exist to guard this; run them after
  touching the feature or training path.

### File Naming
- `*_alpha.py` — Alpha model variants
- `*_date.py` — Date-aware processing
- `ml_*.py` — Machine learning scripts
- `predict_*.py` — Prediction scripts

## React Frontend Structure

The public site is static: every page reads `frontend/src/data/backtest.json` and
`ledger.json`, which `testing/export_site_data.py` writes. No public page calls the Flask
API. Brand and paywall constants live in `frontend/src/constants.js` (`SITE_NAME`,
`MEMBERSHIP_URL`).

`frontend/src/aggregate.js` — browser twin of the export's summary sections; every page
recomputes for the URL window (`?from=&to=`).

Components in `frontend/src/components/`:
- `Home.js` — landing page: headline stats, how it works, calibration proof, market table, membership CTA
- `Results.js` — the full walk-forward report
- `Bets.js` — bet log with Backtest / Live segments
- `Methodology.js` — pipeline explainer
- `Join.js` — membership page
- `PeriodFilter.js` — presets + custom range
- `Navbar.js`, `Footer.js`, `StatTile.js`, `ScrollToTop.js` — chrome
- `charts/` — recharts wrappers: `CalibrationChart`, `MonthlyAccuracyChart`, `BankrollChart`, shared `chartTheme`
- `FightersPage.js` / `FightersDropdown.js` — unrouted legacy components

## Dependencies

Python: Flask, Flask-CORS, Flask-SQLAlchemy, pandas, numpy, lightgbm, scikit-learn, scipy, joblib,
optuna, beautifulsoup4, requests, matplotlib. `shap` is optional (diagnostics only).

Frontend: React 18, Tailwind CSS
