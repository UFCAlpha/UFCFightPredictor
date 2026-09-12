import csv
import os
import json
import pandas as pd
import sys
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import log_loss
import optuna
import os

file_path = os.path.join("data", "detailed_fights.csv")
# file_path = "predict_fights_alpha.csv"

df = pd.read_csv(file_path)

label_encoder = LabelEncoder()
df["Result"] = label_encoder.fit_transform(df["Result"])

selected_columns = df.columns.tolist()

columns_to_remove = ["Red Fighter", "Blue Fighter", "Title", "Date"]
selected_columns = [col for col in selected_columns if col not in columns_to_remove]

low_importance_to_remove = [
    
]
selected_columns = [col for col in selected_columns if col not in low_importance_to_remove]
# selected_columns = [col for col in selected_columns if 'red' not in col.lower() and 'blue' not in col.lower()]
# oppdiff (Red-minus-Blue matchup) columns re-admitted 2026-08-31: the strip
# below dated to 2024-01 (pre-leakage-fixes). The swap augmentation already
# sign-flips them correctly, and they give the trees direct resolution on
# close matchups. Re-strip by uncommenting if a retrain degrades calibration.
# selected_columns = [col for col in selected_columns if 'oppdiff' not in col]

split_index = int(len(df) * 0.95)
# correlations computed on training rows only, so feature selection can't see the test set
corr_matrix = df[selected_columns].iloc[:split_index].corr().abs()

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]


def _mirror(column):
    """The Red/Blue counterpart of a feature name."""
    if column.startswith("Red "):
        return "Blue " + column[len("Red "):]
    if column.startswith("Blue "):
        return "Red " + column[len("Blue "):]
    return column


# The training set below is augmented with Red/Blue-swapped copies of every row,
# so the retained feature set has to be closed under that swap. The triangular
# correlation scan above is order-dependent and drops only one side of a pair --
# leaving e.g. "Blue Body% defense" with no "Red Body% defense". Renaming then
# produces a column the original frame lacks, pd.concat unions the two frames,
# and the orphaned columns are silently filled with NaN for half the rows. Drop
# both halves of any such pair so the set stays symmetric.
_asymmetric = {_mirror(c) for c in to_drop
               if _mirror(c) != c and _mirror(c) in selected_columns} - set(to_drop)
if _asymmetric:
    print(f"pruning {len(_asymmetric)} column(s) to keep Red/Blue symmetry: {sorted(_asymmetric)}")
    to_drop = to_drop + sorted(_asymmetric)
to_drop = [c for c in to_drop if c != "Result"]

# Drop highly correlated features
df.drop(to_drop, axis=1, inplace=True)

selected_columns = [column for column in selected_columns if column not in to_drop]

df = df[selected_columns]

X = df.drop(["Result"], axis=1)
y = df["Result"]

last_index = int(len(df) * 1)
X_train, X_test = X[:split_index], X[split_index:last_index]
y_train, y_test = y[:split_index], y[split_index:last_index]

seed = 42
prune_index = int(len(X_train) * 0.3)


X_train = X_train[prune_index:]
y_train = y_train[prune_index:]

# win_count = y_train.value_counts()[1]  # Assuming 'win' is encoded as 1
# loss_count = y_train.value_counts()[0]  # Assuming 'loss' is encoded as 0

# print(f"Number of wins in train: {win_count}")
# print(f"Number of losses in train: {loss_count}")

# win_count2 = y_test.value_counts()[1]  # Assuming 'win' is encoded as 1
# loss_count2 = y_test.value_counts()[0]  # Assuming 'loss' is encoded as 0

# print(f"Number of wins in test: {win_count2}")
# print(f"Number of losses in test: {loss_count2}")
X_train_swapped = X_train.copy()
y_train_swapped = y_train.copy()

swap_columns = {}
for column in X_train.columns:
    if "Red" in column:
        swap_columns[column] = column.replace("Red", "Blue")
    elif "Blue" in column:
        swap_columns[column] = column.replace("Blue", "Red")


X_train_swapped.rename(columns=swap_columns, inplace=True)
for column in X_train.columns:
    if "oppdiff" in column:
        X_train_swapped[column] = X_train[column] * -1

y_train_swapped = y_train_swapped.apply(lambda x: 0 if x == 1 else 1)

# Guard the invariant the swap depends on: mirroring must not invent columns.
if set(X_train_swapped.columns) != set(X_train.columns):
    raise RuntimeError(
        "Red/Blue swap changed the feature set -- concat would pad with NaN. "
        f"only in swapped: {sorted(set(X_train_swapped.columns) - set(X_train.columns))}")

X_train_extended = pd.concat([X_train, X_train_swapped], ignore_index=True)
y_train_extended = pd.concat([y_train, y_train_swapped], ignore_index=True)
assert list(X_train_extended.columns) == list(X_test.columns), "train/test feature mismatch"

from sklearn.model_selection import TimeSeriesSplit

def pruning_callback(trial):
    # report per-iteration CV loss so the pruner can stop hopeless trials early
    def _callback(env):
        trial.report(env.evaluation_result_list[0][2], env.iteration)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return _callback

def objective(trial):
    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        # 'metric': 'multi_error',
        'verbosity': -1,
        'boosting_type': 'gbdt', 
    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),  # Tighter range to prevent too strong regularization
    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
    'num_leaves': trial.suggest_int('num_leaves', 20, 70),  # Default is 31, slightly increased
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # More conservative learning rate
    'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),  # Increased minimum to prevent overfitting
    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.8),
    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.8),
    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    'subsample': trial.suggest_float('subsample', 0.7, 1.0),  # Adjust for balance between speed and accuracy
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
        'num_class': 2  
    }
    # CV runs on the un-augmented training set: with the mirrored copies appended,
    # TimeSeriesSplit folds would validate on swapped duplicates of fights already
    # seen in training, leaking data into early stopping and the tuning score.
    data = lgb.Dataset(X_train, label=y_train)

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)  # Adjust the number of splits as needed

    # Training model with time series cross-validation
    cv_results = lgb.cv(
        param,
        data,
        num_boost_round=1000,
        folds=tscv,
        stratified=False,
        shuffle=False,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), pruning_callback(trial)],
    )

    scores = cv_results['valid multi_logloss-mean']
    # early stopping truncates the history at the best iteration; keep it so the
    # final refit trains that many trees instead of the sklearn default of 100
    trial.set_user_attr('best_iteration', len(scores))

    return scores[-1]

n_models = 5

sampler = optuna.samplers.TPESampler(seed=seed)
pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=150)

# ensemble the top-N distinct completed trials instead of N one-trial studies:
# same diversity mechanism (different param sets per member), but each member
# is now a tuned draw instead of a random one
completed = sorted(
    (t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
    key=lambda t: t.value,
)
top_trials, seen = [], set()
for t in completed:
    key = tuple(sorted(t.params.items()))
    if key in seen:
        continue
    seen.add(key)
    top_trials.append(t)
    if len(top_trials) == n_models:
        break

models = []
for i, t in enumerate(top_trials):
    member_params = dict(t.params)
    member_params['n_estimators'] = t.user_attrs['best_iteration']
    # distinct seed per member: with LightGBM's fixed default seed the five
    # members' bagging/feature subsamples are identical draws, so near-clone
    # params produce near-clone predictions and averaging removes no variance
    member_params['random_state'] = seed + i
    print(f"ensemble member: cv_logloss={t.value:.4f} params={member_params}")
    model = lgb.LGBMClassifier(**member_params)
    model.fit(X_train_extended, y_train_extended)
    models.append(model)

# ---- temperature calibration on pooled out-of-fold predictions ----
# logit(q) = a * logit(p): no intercept, so calibrated corner probabilities stay
# complementary under the Red/Blue swap, and a = 1 is the identity. Fit on OOF
# predictions so the calibrator never scores a fight its models trained on.
# (Full beta calibration was rejected: an intercept or a != b breaks the swap
# invariant — predictions would depend on which corner ordering was fed in.)
from scipy.optimize import minimize_scalar
import joblib

oof_splits = TimeSeriesSplit(n_splits=5)
oof_probs, oof_labels = [], []
for fold_train, fold_val in oof_splits.split(X_train):
    X_fold, y_fold = X_train.iloc[fold_train], y_train.iloc[fold_train]
    # augment within the fold exactly as the final members are trained
    X_fold_swapped = X_fold.rename(columns=swap_columns)
    for column in X_fold.columns:
        if "oppdiff" in column:
            X_fold_swapped[column] = X_fold[column] * -1
    X_fold_ext = pd.concat([X_fold, X_fold_swapped], ignore_index=True)
    y_fold_ext = pd.concat([y_fold, y_fold.apply(lambda v: 0 if v == 1 else 1)],
                           ignore_index=True)
    fold_member_probs = []
    for i, t in enumerate(top_trials):
        fold_params = dict(t.params)
        fold_params['n_estimators'] = t.user_attrs['best_iteration']
        fold_params['random_state'] = seed + i
        fold_model = lgb.LGBMClassifier(**fold_params)
        fold_model.fit(X_fold_ext, y_fold_ext)
        fold_member_probs.append(fold_model.predict_proba(X_train.iloc[fold_val])[:, 1])
    oof_probs.append(np.mean(fold_member_probs, axis=0))
    oof_labels.append(y_train.iloc[fold_val].to_numpy())

oof_probs = np.clip(np.concatenate(oof_probs), 1e-6, 1 - 1e-6)
oof_labels = np.concatenate(oof_labels)
oof_logits = np.log(oof_probs / (1 - oof_probs))

def oof_nll(a):
    q = np.clip(1.0 / (1.0 + np.exp(-a * oof_logits)), 1e-9, 1 - 1e-9)
    return -np.mean(oof_labels * np.log(q) + (1 - oof_labels) * np.log(1 - q))

temperature = float(minimize_scalar(oof_nll, bounds=(0.25, 4.0), method='bounded').x)
print(f"temperature calibrator: a={temperature:.3f} (identity=1.0), "
      f"OOF logloss {oof_nll(1.0):.4f} -> {oof_nll(temperature):.4f} on {len(oof_labels)} fights")
os.makedirs('saved_preprocessing', exist_ok=True)
joblib.dump({'a': temperature}, os.path.join('saved_preprocessing', 'calibrator.joblib'))

# SHAP feature-importance diagnostics — reporting only, and deliberately optional.
# shap is not in requirements.txt, and summary_plot() blocks on render, so a
# scheduled retrain must be able to skip this and still reach the joblib dump below.
try:
    import shap
except ImportError:
    shap = None
    print('shap not installed — skipping feature-importance diagnostics')

if shap is not None:
    shap_values_list = []

    for model in models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # older shap returns a list of per-class arrays; newer returns a single
        # (samples, features) or (samples, features, classes) array
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]
        shap_values_list.append(shap_values)

    shap_values_array = np.array(shap_values_list)  # Shape: [num_models, num_samples, num_features]

    average_shap_values = np.mean(shap_values_array, axis=0)

    # Summing SHAP values across all samples to get an overall measure of feature importance
    feature_importance = np.abs(average_shap_values).mean(axis=0)

    # Sorting features by their importance
    sorted_feature_indices = np.argsort(feature_importance)[::-1]

    # Sorted feature names
    sorted_features = np.array(X_test.columns)[sorted_feature_indices]

    # Sorted SHAP values
    sorted_shap_values = average_shap_values[:, sorted_feature_indices]

    # Plotting
    shap.summary_plot(sorted_shap_values, features=X_test[sorted_features],
                          plot_type='bar', show=False)

    threshold_percentile = 15
    threshold = np.percentile(feature_importance, threshold_percentile)

    # Get the features whose importance is below the threshold
    low_importance_features = X_test.columns[feature_importance < threshold]

    print("low importance")
    print(low_importance_features)


predicted_probabilities = [model.predict_proba(X_test) for model in models]
ensemble_predicted_probabilities = np.mean(predicted_probabilities, axis=0)

ensemble_preds = np.argmax(ensemble_predicted_probabilities, axis=1)

# Evaluate the ensemble model
accuracy = accuracy_score(y_test, ensemble_preds)
logloss = log_loss(y_test, ensemble_predicted_probabilities)
print(accuracy)
print(logloss)
# Get the fighter names and actual results for the test set
df_with_details = pd.read_csv(file_path)[
    ["Red Fighter", "Blue Fighter", "Result"]
]
df_with_details = df_with_details.iloc[split_index:]  # Align with the test data split
df_with_details.reset_index(drop=True, inplace=True)
# transform with the already-fit encoder; refitting on the test slice could change
# the class mapping and that encoder is saved for production use below
df_with_details["Result"] = label_encoder.transform(df_with_details["Result"])

# Convert the predicted and actual results back to the original labels if necessary.
predicted_labels = label_encoder.inverse_transform(ensemble_preds)
actual_labels = label_encoder.inverse_transform(df_with_details["Result"])

with open(os.path.join("data", "predicted_results.csv"), mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Red Fighter",
            "Blue Fighter",
            "Predicted Result",
            "Probability",
            "Actual Result",
        ]
    )
    for i in range(len(predicted_labels)):
        max_probability = max(ensemble_predicted_probabilities[i])

        writer.writerow(
            [
                df_with_details["Red Fighter"].iloc[i],
                df_with_details["Blue Fighter"].iloc[i],
                predicted_labels[i],
                max_probability,  # Formatting as a percentage
                actual_labels[i],
            ]
        )

model_save_dir = "saved_models"
os.makedirs(model_save_dir, exist_ok=True)

for idx, model in enumerate(models):
    model_filename = os.path.join(model_save_dir, f"lgbm_model_{idx}.joblib")
    joblib.dump(model, model_filename)

preprocessing_save_dir = "saved_preprocessing"
os.makedirs(preprocessing_save_dir, exist_ok=True)

label_encoder_filename = os.path.join(preprocessing_save_dir, "label_encoder.joblib")
joblib.dump(label_encoder, label_encoder_filename)

selected_columns_filename = os.path.join(preprocessing_save_dir, "selected_columns.json")
with open(selected_columns_filename, "w") as file:
    json.dump(selected_columns, file)

# Record genuinely unseen holdout metrics before auto_retrain's production refit.
# These hashes bind validation to this exact dataset and candidate artifact set.
from production_refit import record_candidate_evaluation
record_candidate_evaluation(file_path, model_save_dir, preprocessing_save_dir)
