"""Fit the locked logistic model on eligible training seasons."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_moneyline.constants import (
    FEATURE_COLUMNS,
    LOGISTIC_MAX_ITER,
    LOGISTIC_RANDOM_STATE,
    LOGISTIC_SOLVER,
    TEST_SEASON_START,
    TRAIN_SEASON_STARTS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = PROJECT_ROOT / "data" / "features.csv"
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "logistic_regression.joblib"
METRICS_PATH = PROJECT_ROOT / "results" / "tables" / "holdout_metrics.csv"
COEF_PATH = PROJECT_ROOT / "results" / "tables" / "coefficients.csv"


def load_model_bundle() -> dict:
    """Load the saved pipeline.

    scikit-learn 1.8 dropped ``LogisticRegression.multi_class``. A model saved
    with that version has no such attribute, and 1.7 still reads it on predict.
    """

    bundle = joblib.load(MODEL_PATH)
    logistic = bundle["pipeline"].named_steps["logistic"]
    if not hasattr(logistic, "multi_class"):
        logistic.multi_class = "auto"
    return bundle


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    solver=LOGISTIC_SOLVER,
                    max_iter=LOGISTIC_MAX_ITER,
                    random_state=LOGISTIC_RANDOM_STATE,
                ),
            ),
        ]
    )


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = frame.loc[frame["eligible"] == 1].copy()
    train = eligible.loc[eligible["season_start"].isin(TRAIN_SEASON_STARTS)]
    test = eligible.loc[eligible["season_start"] == TEST_SEASON_START]
    if train.empty or test.empty:
        raise RuntimeError("training or holdout split has no eligible games")
    return train, test


def _metrics(name: str, y_true: pd.Series, probabilities: pd.Series) -> dict[str, float | str | int]:
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "model": name,
        "games": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def fit_and_evaluate(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    if frame is None:
        frame = pd.read_csv(FEATURES_CSV)
    train, test = _split(frame)
    pipeline = make_pipeline()
    pipeline.fit(train[list(FEATURE_COLUMNS)], train["home_win"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "train_season_starts": TRAIN_SEASON_STARTS,
            "test_season_start": TEST_SEASON_START,
        },
        MODEL_PATH,
    )

    train_home_rate = float(train["home_win"].mean())
    logistic_probability = pipeline.predict_proba(test[list(FEATURE_COLUMNS)])[:, 1]
    baseline_probability = pd.Series(train_home_rate, index=test.index)
    metrics = pd.DataFrame(
        [
            _metrics("home_win_rate_baseline", test["home_win"], baseline_probability),
            _metrics("logistic", test["home_win"], pd.Series(logistic_probability, index=test.index)),
        ]
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)

    logistic = pipeline.named_steps["logistic"]
    coefficients = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "coefficient": logistic.coef_[0]}
    ).sort_values("coefficient", key=lambda column: column.abs(), ascending=False)
    coefficients.to_csv(COEF_PATH, index=False)
    print(metrics.to_string(index=False))
    print(f"wrote {MODEL_PATH}")
    return metrics
