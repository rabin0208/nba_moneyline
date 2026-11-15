"""Join saved model probabilities to closing moneylines."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from nba_moneyline.closing_odds import CLOSING_ODDS_CSV
from nba_moneyline.constants import FEATURE_COLUMNS
from nba_moneyline.market import american_to_implied_prob, flat_bet_profit, is_valid_american_odds, remove_vig
from nba_moneyline.opening_odds import (
    AN_CLOSING_ODDS_CSV,
    OPENING_ODDS_CSV,
    overlay_closes,
    overlay_opens,
)
from nba_moneyline.train import FEATURES_CSV, MODEL_PATH, load_model_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def season_label(season_start: int) -> str:
    return f"{season_start}–{str(season_start + 1)[-2:]}"


def load_model_games() -> pd.DataFrame:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Fit the model first.")
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    columns = list(bundle.get("feature_columns") or FEATURE_COLUMNS)
    features = pd.read_csv(FEATURES_CSV)
    eligible = features.loc[features["eligible"] == 1].copy()
    eligible["p_home_win"] = pipeline.predict_proba(eligible[columns])[:, 1]
    if AN_CLOSING_ODDS_CSV.is_file():
        merged = overlay_closes(eligible, pd.read_csv(AN_CLOSING_ODDS_CSV))
        if merged.empty:
            raise RuntimeError("No games matched an Action Network consensus close.")
    else:
        if not CLOSING_ODDS_CSV.is_file():
            raise FileNotFoundError(
                f"Missing {CLOSING_ODDS_CSV}. Run: python -m nba_moneyline closing-odds"
            )
        odds = pd.read_csv(CLOSING_ODDS_CSV)
        merged = eligible.merge(odds, on=["game_id", "season_start"], how="inner")
        if merged.empty:
            raise RuntimeError("No games matched a closing moneyline.")
    if OPENING_ODDS_CSV.is_file():
        merged = overlay_opens(merged, pd.read_csv(OPENING_ODDS_CSV))
    return merged.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def available_seasons(games: pd.DataFrame) -> list[int]:
    return [int(season) for season in sorted(games["season_start"].unique())]


def _pick_price(
    home_odds: object, away_odds: object, p_home: float, home_win: int
) -> tuple[float, float, bool, str]:
    """Edge, $1 profit, and favorite/underdog role for the side the model picks."""

    if not is_valid_american_odds(home_odds) or not is_valid_american_odds(away_odds):
        return math.nan, math.nan, False, ""
    fair_home, fair_away = remove_vig(
        american_to_implied_prob(float(home_odds)),
        american_to_implied_prob(float(away_odds)),
    )
    pick_home = float(p_home) >= 0.5
    if pick_home:
        edge = float(p_home) - fair_home
        odds = float(home_odds)
        fair = fair_home
        won = int(home_win) == 1
    else:
        edge = (1.0 - float(p_home)) - fair_away
        odds = float(away_odds)
        fair = fair_away
        won = int(home_win) == 0
    if abs(fair - 0.5) <= 1e-9:
        role = "pickem"
    elif fair > 0.5:
        role = "favorite"
    else:
        role = "underdog"
    return edge, flat_bet_profit(odds, won), True, role


def attach_pick_prices(games: pd.DataFrame) -> pd.DataFrame:
    """Price the model's predicted winner at the close and at the open."""

    rows: list[dict[str, object]] = []
    for game in games.itertuples(index=False):
        close_edge, close_profit, close_valid, close_role = _pick_price(
            game.home_odds, game.away_odds, game.p_home_win, game.home_win
        )
        open_edge, open_profit, open_valid, open_role = _pick_price(
            game.home_open, game.away_open, game.p_home_win, game.home_win
        )
        rows.append(
            {
                "season_start": int(game.season_start),
                "pick_home": int(float(game.p_home_win) >= 0.5),
                "correct": int((int(game.home_win) == 1) == (float(game.p_home_win) >= 0.5)),
                "close_edge": close_edge,
                "close_profit": close_profit,
                "close_valid": close_valid,
                "close_role": close_role,
                "open_edge": open_edge,
                "open_profit": open_profit,
                "open_valid": open_valid,
                "open_role": open_role,
            }
        )
    return pd.DataFrame(rows)


def _season_row(
    name: str, group: pd.DataFrame, threshold: float, market: str, side: str
) -> dict[str, object]:
    edge = group[f"{market}_edge"]
    valid = group[f"{market}_valid"].astype(bool)
    bets = group[valid & edge.notna()]
    if threshold > 0:
        bets = bets[bets[f"{market}_edge"] >= threshold]
    if side != "all":
        bets = bets[bets[f"{market}_role"] == side]
    bet_profit = float(bets[f"{market}_profit"].sum()) if len(bets) else math.nan
    return {
        "season": name,
        "games": len(group),
        "accuracy": float(group["correct"].mean()) if len(group) else math.nan,
        "bets": len(bets),
        "hit_rate": float((bets[f"{market}_profit"] > 0).mean()) if len(bets) else math.nan,
        "profit": bet_profit,
        "roi": bet_profit / len(bets) if len(bets) else math.nan,
    }


def season_result_table(
    priced: pd.DataFrame, threshold: float, market: str, side: str = "all"
) -> pd.DataFrame:
    """One row per season plus an overall row. Accuracy is every game; ROI is the picked side."""

    rows = [
        _season_row(season_label(int(season)), group, threshold, market, side)
        for season, group in priced.groupby("season_start", sort=True)
    ]
    rows.append(_season_row("Overall", priced, threshold, market, side))
    return pd.DataFrame(rows)


def same_ticket_roi(
    priced: pd.DataFrame, threshold: float, side: str = "all"
) -> tuple[int, float, float] | None:
    """Reprice closing picks at the opening line, on games that have both."""

    both = priced[priced["close_valid"].astype(bool) & priced["open_valid"].astype(bool)].copy()
    both = both[both["close_edge"].notna() & both["open_profit"].notna()]
    if threshold > 0:
        both = both[both["close_edge"] >= threshold]
    if side != "all":
        both = both[both["close_role"] == side]
    if both.empty:
        return None
    return len(both), float(both["close_profit"].sum()) / len(both), float(both["open_profit"].sum()) / len(both)
