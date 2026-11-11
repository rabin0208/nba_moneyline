"""Moneyline prices, vig removal, and flat versus Kelly profit."""

from __future__ import annotations

import math

import pandas as pd

MARKET_SHRINK = 0.5
KELLY_MULTIPLIER = 0.25
MAX_BET_FRAC = 0.05
SLATE_CAP_FRAC = 0.10


def is_valid_american_odds(odds: float) -> bool:
    try:
        price = float(odds)
    except (TypeError, ValueError):
        return False
    if math.isnan(price):
        return False
    return price <= -100.0 or price >= 100.0


def parse_american_odds(text: object) -> float | None:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return None
    raw = str(text).strip().upper().replace(",", "")
    if raw in {"EVEN", "PK", "PICK", "PICKEM"}:
        return 100.0
    if raw.startswith("+"):
        raw = raw[1:]
    try:
        price = float(raw)
    except ValueError:
        return None
    if not is_valid_american_odds(price):
        return None
    return price


def american_to_implied_prob(odds: float) -> float:
    if not is_valid_american_odds(odds):
        return math.nan
    price = float(odds)
    if price > 0:
        return 100.0 / (price + 100.0)
    return abs(price) / (abs(price) + 100.0)


def american_to_decimal(odds: float) -> float:
    price = float(odds)
    if not is_valid_american_odds(price):
        return math.nan
    if price > 0:
        return 1.0 + price / 100.0
    return 1.0 + 100.0 / abs(price)


def remove_vig(p_home: float, p_away: float) -> tuple[float, float]:
    total = p_home + p_away
    if total <= 0:
        return math.nan, math.nan
    return p_home / total, p_away / total


def flat_bet_profit(american_odds: float, won: bool) -> float:
    if not won:
        return -1.0
    price = float(american_odds)
    if price > 0:
        return price / 100.0
    return 100.0 / abs(price)


def kelly_fraction(probability: float, american_odds: float) -> float:
    decimal = american_to_decimal(american_odds)
    if math.isnan(decimal) or decimal <= 1.0:
        return math.nan
    return (float(probability) * decimal - 1.0) / (decimal - 1.0)


def recommended_bankroll_frac(
    probability: float,
    market_probability: float,
    american_odds: float,
) -> float:
    """Quarter Kelly on a 50/50 blend of the model and the vig-free line, capped at 5%."""

    try:
        model = float(probability)
        market = float(market_probability)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(model) or math.isnan(market):
        return math.nan
    blended = (1.0 - MARKET_SHRINK) * model + MARKET_SHRINK * market
    full = kelly_fraction(blended, american_odds)
    if math.isnan(full) or full <= 0:
        return 0.0
    return float(min(KELLY_MULTIPLIER * full, MAX_BET_FRAC))


def cap_slate_stakes(stakes: pd.Series, cap: float = SLATE_CAP_FRAC) -> pd.Series:
    """Scale one slate so the recommended stakes sum to at most `cap` of bankroll."""

    values = pd.to_numeric(stakes, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(values.sum())
    if total <= 0 or total <= cap:
        return values
    return values * (cap / total)


def format_american_odds(odds: float) -> str:
    price = int(round(float(odds)))
    return f"+{price}" if price > 0 else str(price)


def add_market_probs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["p_home_mkt_raw"] = out["home_odds"].map(american_to_implied_prob)
    out["p_away_mkt_raw"] = out["away_odds"].map(american_to_implied_prob)
    fair = [
        remove_vig(home, away)
        for home, away in zip(out["p_home_mkt_raw"], out["p_away_mkt_raw"], strict=True)
    ]
    out["p_home_mkt"] = [pair[0] for pair in fair]
    out["p_away_mkt"] = [pair[1] for pair in fair]
    return out


def pick_bets(frame: pd.DataFrame, *, edge_threshold: float) -> pd.DataFrame:
    """Bet the side whose model probability beats the fair close by at least the threshold."""

    out = add_market_probs(frame)
    out["edge_home"] = out["p_home_win"] - out["p_home_mkt"]
    out["edge_away"] = (1.0 - out["p_home_win"]) - out["p_away_mkt"]
    sides: list[str | None] = []
    for home_edge, away_edge in zip(out["edge_home"], out["edge_away"], strict=True):
        if home_edge >= edge_threshold and home_edge >= away_edge:
            sides.append("home")
        elif away_edge >= edge_threshold:
            sides.append("away")
        else:
            sides.append(None)
    out["bet_side"] = sides
    out["bet_odds"] = [
        row.home_odds if side == "home" else row.away_odds if side == "away" else math.nan
        for side, row in zip(sides, out.itertuples(index=False), strict=True)
    ]
    out["bet_edge"] = [
        row.edge_home if side == "home" else row.edge_away if side == "away" else math.nan
        for side, row in zip(sides, out.itertuples(index=False), strict=True)
    ]
    out["bet_prob"] = [
        row.p_home_win if side == "home" else 1.0 - row.p_home_win if side == "away" else math.nan
        for side, row in zip(sides, out.itertuples(index=False), strict=True)
    ]
    out["bet_won"] = [
        (side == "home" and bool(won)) or (side == "away" and not bool(won))
        if side and pd.notna(won)
        else math.nan
        for side, won in zip(sides, out["home_win"], strict=True)
    ]
    out["bet_profit"] = [
        flat_bet_profit(odds, bool(won))
        if side and pd.notna(won) and is_valid_american_odds(odds)
        else math.nan
        for side, odds, won in zip(sides, out["bet_odds"], out["bet_won"], strict=True)
    ]
    kelly = []
    for probability, odds, side in zip(out["bet_prob"], out["bet_odds"], sides, strict=True):
        if side is None or not is_valid_american_odds(odds):
            kelly.append(0.0)
            continue
        fraction = kelly_fraction(probability, odds)
        kelly.append(0.0 if math.isnan(fraction) else max(0.0, min(1.0, fraction)))
    out["kelly_stake"] = kelly
    return out


def normalize_stakes_mean_one(stakes: pd.Series) -> pd.Series:
    values = pd.to_numeric(stakes, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(values.sum())
    if total <= 0 or len(values) == 0:
        return values
    return values * (len(values) / total)


def prepare_profit_bets(frame: pd.DataFrame, edge_threshold: float) -> pd.DataFrame:
    scored = pick_bets(frame, edge_threshold=edge_threshold)
    bets = scored.dropna(subset=["bet_side", "bet_odds", "bet_profit"]).copy()
    bets = bets.loc[bets["bet_odds"].map(is_valid_american_odds)]
    if bets.empty:
        return bets
    bets["kelly_stake_u"] = normalize_stakes_mean_one(bets["kelly_stake"])
    bets["kelly_profit_u"] = bets["kelly_stake_u"] * bets["bet_profit"]
    return bets.sort_values(["game_date", "game_id"]).reset_index(drop=True)
