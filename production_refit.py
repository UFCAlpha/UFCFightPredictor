"""Refit validated candidates on recent eligible history, including the holdout.

The holdout report belongs to the candidate, never to the final production fit.
Hyperparameters and selected columns stay frozen. Production calibration uses
expanding chronological folds, keeping all fights on a date in the same fold.
The caller must back up artifacts and restore them if publication fails.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.base import clone
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from calibration import apply_temperature

MANIFEST = "training_manifest.json"


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _artifact_hashes(model_dir, prep_dir):
    models = sorted(Path(model_dir).glob("lgbm_model_*.joblib"))
    if len(models) != 5:
        raise ValueError("expected exactly five ensemble members")
    files = models + [Path(prep_dir) / name for name in
                      ("selected_columns.json", "label_encoder.joblib", "calibrator.joblib")]
    return {p.name: _sha256(p) for p in files}


def _inputs(features_path, model_dir, prep_dir):
    frame = pd.read_csv(features_path, low_memory=False)
    dates = pd.to_datetime(frame["Date"])
    if dates.isna().any() or not dates.is_monotonic_increasing:
        raise ValueError("fight dates must be present and chronologically sorted")
    if dates.max().date() > datetime.now().date():
        raise ValueError("training data contains future fights")
    encoder = joblib.load(Path(prep_dir) / "label_encoder.joblib")
    if list(encoder.classes_) != ["loss", "win"]:
        raise ValueError("expected class 1 to mean red wins")
    selected = json.loads((Path(prep_dir) / "selected_columns.json").read_text())
    columns = [c for c in selected if c != "Result"]
    X = frame[columns]
    y = pd.Series(encoder.transform(frame["Result"]), index=frame.index)
    models = [joblib.load(p) for p in sorted(Path(model_dir).glob("lgbm_model_*.joblib"))]
    # LightGBM replaces spaces in pandas column names with underscores.
    stored_names = [c.replace(" ", "_") for c in columns]
    if len(models) != 5 or any(list(m.feature_name_) != stored_names for m in models):
        raise ValueError("ensemble members do not match saved selected columns")
    return dates, X, y, models


def mirror_features(X):
    swap = {c: c.replace("Red", "Blue") if "Red" in c else c.replace("Blue", "Red")
            for c in X.columns if "Red" in c or "Blue" in c}
    mirrored = X.rename(columns=swap).copy()
    if set(mirrored.columns) != set(X.columns):
        raise ValueError("selected feature set must be Red/Blue symmetric")
    for c in X.columns:
        if "oppdiff" in c:
            mirrored[c] = -X[c]
    return mirrored[X.columns]


def _fit(models, X, y):
    augmented = pd.concat([X, mirror_features(X)], ignore_index=True)
    labels = pd.concat([y, 1-y], ignore_index=True)
    return [clone(m).fit(augmented, labels) for m in models]


def _predict(models, X):
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    if not np.isfinite(p).all():
        raise ValueError("ensemble produced non-finite probabilities")
    return p


def _save_manifest(prep_dir, report):
    target = Path(prep_dir) / MANIFEST
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


def record_candidate_evaluation(features_path, model_dir, prep_dir):
    """Called by ml_ensemble after saving its holdout-excluding candidates."""
    hashes = _artifact_hashes(model_dir, prep_dir)
    existing = Path(prep_dir) / MANIFEST
    if existing.exists():
        previous = json.loads(existing.read_text())
        if previous.get("stage") == "production" and previous.get("artifact_sha256") == hashes:
            raise ValueError("production models cannot be reevaluated as unseen-holdout candidates")
    dates, X, y, models = _inputs(features_path, model_dir, prep_dir)
    split = int(len(X) * .95)
    first = int(split * .3)  # keep ml_ensemble's intentional old-history exclusion
    p = _predict(models, X.iloc[split:])
    a = joblib.load(Path(prep_dir) / "calibrator.joblib")["a"]
    q = apply_temperature(p, a)
    labels = y.iloc[split:]
    report = {
        "stage": "candidate", "created_at": datetime.now(timezone.utc).isoformat(),
        "data_sha256": _sha256(features_path), "artifact_sha256": hashes,
        "candidate": {"rows": split-first, "first_row": first, "holdout_first_row": split,
                      "first_fight": dates.iloc[first].date().isoformat(),
                      "last_fight": dates.iloc[split-1].date().isoformat()},
        "evaluation": {
            "scope": "candidate holdout before production refit", "n": len(labels),
            "first_fight": dates.iloc[split].date().isoformat(),
            "last_fight": dates.iloc[-1].date().isoformat(),
            "accuracy": float(accuracy_score(labels, p >= .5)),
            "log_loss": float(log_loss(labels, q, labels=[0, 1])),
            "brier": float(brier_score_loss(labels, q)),
            "auc": float(roc_auc_score(labels, q)) if labels.nunique() == 2 else None,
        },
    }
    _save_manifest(prep_dir, report)
    return report


def load_candidate_evaluation(features_path, model_dir, prep_dir):
    report = json.loads((Path(prep_dir) / MANIFEST).read_text())
    if report.get("stage") != "candidate":
        raise ValueError("need fresh candidate models; production models have already seen the holdout")
    if report["data_sha256"] != _sha256(features_path):
        raise ValueError("fight data changed since candidate evaluation")
    if report["artifact_sha256"] != _artifact_hashes(model_dir, prep_dir):
        raise ValueError("model or preprocessing artifacts changed since candidate evaluation")
    return report


def refit_production_ensemble(features_path, model_dir, prep_dir, min_accuracy=.60, log=print):
    report = load_candidate_evaluation(features_path, model_dir, prep_dir)
    accuracy = report["evaluation"]["accuracy"]
    if not np.isfinite(accuracy) or accuracy < min_accuracy:
        raise ValueError(f"candidate validation failed: {accuracy} < {min_accuracy}")
    dates, X, y, models = _inputs(features_path, model_dir, prep_dir)
    first = report["candidate"]["first_row"]
    dates, X, y = dates.iloc[first:], X.iloc[first:], y.iloc[first:]
    log(f"Refitting production on {len(X)} fights through {dates.iloc[-1].date()} "
        f"(including {report['evaluation']['n']} former holdout fights)")

    oof_p, oof_y, folds = [], [], []
    unique_dates = dates.unique()
    for i, (tr, va) in enumerate(TimeSeriesSplit(n_splits=5).split(unique_dates), 1):
        train = dates.isin(unique_dates[tr])
        valid = dates.isin(unique_dates[va])
        fitted = _fit(models, X.loc[train], y.loc[train])
        oof_p.append(_predict(fitted, X.loc[valid]))
        oof_y.append(y.loc[valid].to_numpy())
        folds.append({"train_end": dates.loc[train].max().date().isoformat(),
                      "validation_start": dates.loc[valid].min().date().isoformat(),
                      "validation_end": dates.loc[valid].max().date().isoformat(),
                      "train_rows": int(train.sum()), "validation_rows": int(valid.sum())})
        log(f"Production calibration fold {i}/5: {train.sum()} train, {valid.sum()} validation")
    probabilities, labels = np.concatenate(oof_p), np.concatenate(oof_y)
    fitted_temperature = minimize_scalar(
        lambda a: log_loss(labels, apply_temperature(probabilities, a), labels=[0, 1]),
        bounds=(.25, 4.), method="bounded")
    if not fitted_temperature.success or not np.isfinite(fitted_temperature.fun):
        raise ValueError("production calibration failed")
    a = float(fitted_temperature.x)
    log(f"Production temperature: {a:.4f}; fitting five final ensemble members")
    production = _fit(models, X, y)
    _predict(production, X.tail(20))
    # Detect concurrent dataset/model changes before publishing any artifacts.
    load_candidate_evaluation(features_path, model_dir, prep_dir)
    for i, model in enumerate(production):
        joblib.dump(model, Path(model_dir) / f"lgbm_model_{i}.joblib")
    joblib.dump({"a": a}, Path(prep_dir) / "calibrator.joblib")
    report["candidate_artifact_sha256"] = report["artifact_sha256"]
    report["artifact_sha256"] = _artifact_hashes(model_dir, prep_dir)
    report["stage"] = "production"
    report["production"] = {
        "fitted_at": datetime.now(timezone.utc).isoformat(), "rows": len(X),
        "augmented_rows": 2*len(X), "first_fight": dates.iloc[0].date().isoformat(),
        "last_fight": dates.iloc[-1].date().isoformat(), "temperature": a,
        "calibration_rows": len(labels), "calibration_folds": folds,
        "calibration_oof_log_loss": float(fitted_temperature.fun),
    }
    _save_manifest(prep_dir, report)
    log("Production refit saved; reported holdout metrics remain the pre-refit candidate's")
    return report
