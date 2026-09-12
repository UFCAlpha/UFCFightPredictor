"""Production refits must include recent fights without relabeling training scores as holdout scores."""
import importlib
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

import auto_retrain


@pytest.fixture
def candidate(tmp_path):
    models = tmp_path / "models"
    prep = tmp_path / "prep"
    models.mkdir()
    prep.mkdir()
    x = np.where(np.arange(80) % 2, 2., -2.)
    frame = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=40).repeat(2),
        "Red x": x, "Blue x": -x, "x oppdiff": 2*x,
        "Result": np.where(x > 0, "win", "loss"),
    })
    features = tmp_path / "features.csv"
    frame.to_csv(features, index=False)
    encoder = LabelEncoder().fit(["loss", "win"])
    joblib.dump(encoder, prep / "label_encoder.joblib")
    joblib.dump({"a": 1.}, prep / "calibrator.joblib")
    columns = ["Red x", "Blue x", "x oppdiff"]
    (prep / "selected_columns.json").write_text(json.dumps(columns + ["Result"]))
    for i in range(5):
        model = lgb.LGBMClassifier(n_estimators=3, num_leaves=3, min_child_samples=1,
                                  verbosity=-1, n_jobs=1, random_state=42+i)
        model.fit(frame.iloc[22:76][columns], encoder.transform(frame.iloc[22:76].Result))
        joblib.dump(model, models / f"lgbm_model_{i}.joblib")
    return features, models, prep


def refit_module():
    assert importlib.util.find_spec("production_refit") is not None, "production refit is not implemented"
    return importlib.import_module("production_refit")


def snapshot(candidate):
    _, models, prep = candidate
    return {str(p): p.read_bytes() for root in (models, prep) for p in root.iterdir()}


def test_refit_includes_latest_rows_and_preserves_pre_refit_evaluation(candidate):
    module = refit_module()
    before = module.record_candidate_evaluation(*candidate)
    report = module.refit_production_ensemble(*candidate, log=lambda _: None)
    assert before["evaluation"]["n"] == 4
    assert report["evaluation"] == before["evaluation"]
    assert report["stage"] == "production"
    assert report["production"]["rows"] == 58
    assert report["production"]["last_fight"] == "2025-02-09"
    assert report["production"]["augmented_rows"] == 116
    _, models, prep = candidate
    for p in models.glob("lgbm_model_*.joblib"):
        model = joblib.load(p)
        root = model.booster_.dump_model()["tree_info"][0]["tree_structure"]
        assert root["internal_count"] == 116
    assert np.isfinite(joblib.load(prep / "calibrator.joblib")["a"])
    for fold in report["production"]["calibration_folds"]:
        assert fold["train_end"] < fold["validation_start"]
    assert report["production"]["calibration_folds"][-1]["validation_end"] == "2025-02-09"
    with pytest.raises(ValueError, match="candidate"):
        module.refit_production_ensemble(*candidate)


def test_failed_holdout_cannot_refit_or_modify_artifacts(candidate):
    module = refit_module()
    features, _, _ = candidate
    frame = pd.read_csv(features)
    frame.loc[76:, "Result"] = frame.loc[76:, "Result"].map({"win": "loss", "loss": "win"})
    frame.to_csv(features, index=False)
    report = module.record_candidate_evaluation(*candidate)
    assert report["evaluation"]["accuracy"] == 0
    before = snapshot(candidate)
    with pytest.raises(ValueError, match="validation"):
        module.refit_production_ensemble(*candidate)
    assert snapshot(candidate) == before


@pytest.mark.parametrize("changed", ["data", "model", "calibrator"])
def test_refit_rejects_changes_since_evaluation(candidate, changed):
    module = refit_module()
    module.record_candidate_evaluation(*candidate)
    features, models, prep = candidate
    path = {"data": features, "model": models / "lgbm_model_0.joblib",
            "calibrator": prep / "calibrator.joblib"}[changed]
    path.write_bytes(path.read_bytes() + b"\n")
    before = snapshot(candidate)
    with pytest.raises(ValueError, match="changed"):
        module.refit_production_ensemble(*candidate)
    assert snapshot(candidate) == before


def test_orphaned_corner_features_fail_before_writes(candidate):
    module = refit_module()
    module.record_candidate_evaluation(*candidate)
    frame = pd.DataFrame({"Red x": [1., 2.]})
    with pytest.raises(ValueError, match="symmetr"):
        module.mirror_features(frame)


def test_mirroring_swaps_corner_values_and_negates_differentials():
    module = refit_module()
    frame = pd.DataFrame({"Red x": [3.], "Blue x": [1.], "x oppdiff": [2.]})
    assert module.mirror_features(frame).iloc[0].tolist() == [1., 3., -2.]


def test_production_models_cannot_be_reported_as_unseen_holdout(candidate, monkeypatch):
    module = refit_module()
    module.record_candidate_evaluation(*candidate)
    module.refit_production_ensemble(*candidate, log=lambda _: None)
    features, models, prep = candidate
    monkeypatch.setattr(auto_retrain, "FEATURES", str(features))
    monkeypatch.setattr(auto_retrain, "MODEL_DIR", str(models))
    monkeypatch.setattr(auto_retrain, "PREP_DIR", str(prep))
    with pytest.raises(ValueError, match="already seen"):
        auto_retrain.evaluate_saved_ensemble()
    with pytest.raises(ValueError, match="production models"):
        module.record_candidate_evaluation(*candidate)


def test_final_refit_failure_restores_candidate_and_metadata(candidate, monkeypatch):
    module = refit_module()
    features, models, prep = candidate
    monkeypatch.setattr(auto_retrain, "MODEL_DIR", str(models))
    monkeypatch.setattr(auto_retrain, "PREP_DIR", str(prep))
    monkeypatch.setattr(auto_retrain, "FEATURES", str(features))
    model_path = models / "lgbm_model_0.joblib"
    original = model_path.read_bytes()
    backup = auto_retrain.backup_models()

    def broken(*args, **kwargs):
        model_path.write_bytes(b"partial refit")
        (prep / "training_manifest.json").write_text('{"stage":"production"}')
        raise RuntimeError("write failed")

    monkeypatch.setattr(module, "refit_production_ensemble", broken)
    with pytest.raises(RuntimeError, match="write failed"):
        auto_retrain.step_refit(backup)
    assert model_path.read_bytes() == original
    assert not (prep / "training_manifest.json").exists()


def test_validation_exception_restores_previous_models(candidate, monkeypatch):
    features, models, prep = candidate
    monkeypatch.setattr(auto_retrain, "MODEL_DIR", str(models))
    monkeypatch.setattr(auto_retrain, "PREP_DIR", str(prep))
    original = (models / "lgbm_model_0.joblib").read_bytes()
    backup = auto_retrain.backup_models()
    (models / "lgbm_model_0.joblib").write_bytes(b"bad candidate")
    def broken():
        raise RuntimeError("validation crashed")
    monkeypatch.setattr(auto_retrain, "evaluate_saved_ensemble", broken)
    with pytest.raises(RuntimeError, match="validation crashed"):
        auto_retrain.step_validate(backup)
    assert (models / "lgbm_model_0.joblib").read_bytes() == original
