"""Join saved model probabilities to closing moneylines."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from nba_moneyline.closing_odds import CLOSING_ODDS_CSV
from nba_moneyline.constants import FEATURE_COLUMNS
from nba_moneyline.train import FEATURES_CSV, MODEL_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def season_label(season_start: int) -> str:
    return f"{season_start}–{str(season_start + 1)[-2:]}"


def load_model_games() -> pd.DataFrame:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Fit the model first.")
    if not CLOSING_ODDS_CSV.is_file():
        raise FileNotFoundError(
            f"Missing {CLOSING_ODDS_CSV}. Run: python -m nba_moneyline closing-odds --season 2025"
        )
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    columns = list(bundle.get("feature_columns") or FEATURE_COLUMNS)
    features = pd.read_csv(FEATURES_CSV)
    eligible = features.loc[features["eligible"] == 1].copy()
    eligible["p_home_win"] = pipeline.predict_proba(eligible[columns])[:, 1]
    odds = pd.read_csv(CLOSING_ODDS_CSV)
    merged = eligible.merge(odds, on=["game_id", "season_start"], how="inner")
    if merged.empty:
        raise RuntimeError("No games matched a closing moneyline.")
    return merged.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def available_seasons(games: pd.DataFrame) -> list[int]:
    return [int(season) for season in sorted(games["season_start"].unique())]
