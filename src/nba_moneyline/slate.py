"""Score one date from team state, then attach the DraftKings moneyline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from nba_moneyline.acquire import cache_path, ensure_scoreboards, games_from_scoreboard, read_games
from nba_moneyline.closing_odds import cached_summary_line, line_from_quote
from nba_moneyline.constants import FEATURE_COLUMNS, TEAMS
from nba_moneyline.features import snapshot_features
from nba_moneyline.schema import Game
from nba_moneyline.train import MODEL_PATH, load_model_bundle

TEAM_SET = set(TEAMS)


@dataclass
class SlateGame:
    event_id: str
    game_date: date
    season_start: int
    home_team: str
    away_team: str
    home_name: str
    away_name: str
    status: str
    status_detail: str
    home_score: int | None
    away_score: int | None
    home_odds: float | None
    away_odds: float | None
    provider: str
    spread: str


def _quote_line(competition: dict) -> dict[str, object] | None:
    quotes = competition.get("odds") or []
    if not quotes:
        return None
    return line_from_quote(quotes[0])


def games_on_scoreboard(payload: dict, slate_date: date) -> list[SlateGame]:
    games: list[SlateGame] = []
    for event in payload.get("events") or []:
        event_id = str(event.get("id", ""))
        season = event.get("season") or {}
        competitions = event.get("competitions") or []
        if not competitions or season.get("type") != 2:
            continue
        competition = competitions[0]
        if competition.get("neutralSite"):
            continue
        status_type = (competition.get("status") or {}).get("type") or {}
        try:
            competitors = competition.get("competitors") or []
            by_side = {row.get("homeAway"): row for row in competitors}
            home = by_side["home"]
            away = by_side["away"]
            home_team = home["team"]["abbreviation"]
            away_team = away["team"]["abbreviation"]
            if home_team not in TEAM_SET or away_team not in TEAM_SET:
                continue
            season_start = int(season["year"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        line = _quote_line(competition) or {}
        home_score = _score(home.get("score"))
        away_score = _score(away.get("score"))
        games.append(
            SlateGame(
                event_id=event_id,
                game_date=slate_date,
                season_start=season_start,
                home_team=home_team,
                away_team=away_team,
                home_name=str((home.get("team") or {}).get("displayName") or home_team),
                away_name=str((away.get("team") or {}).get("displayName") or away_team),
                status=str(status_type.get("name") or ""),
                status_detail=str(status_type.get("shortDetail") or ""),
                home_score=home_score,
                away_score=away_score,
                home_odds=_optional_float(line.get("home_odds")),
                away_odds=_optional_float(line.get("away_odds")),
                provider=str(line.get("provider") or ""),
                spread=str(line.get("spread") or ""),
            )
        )
    return games


def _score(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finals_on(day: date) -> list[Game]:
    path = cache_path(day)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    day_games, _skipped = games_from_scoreboard(payload, day)
    return day_games


def completed_games_before(target: date, *, include_same_day_finals: bool = False) -> list[Game]:
    """Finals strictly before `target`. A live date can also include finals from earlier today."""

    games = [game for game in read_games() if game.game_date < target]
    last_included = max((game.game_date for game in games), default=date(2015, 9, 30))
    day = last_included + timedelta(days=1)
    while day < target:
        games.extend(_finals_on(day))
        day += timedelta(days=1)
    if include_same_day_finals:
        games.extend(_finals_on(target))
    return games


def _placeholder(game: SlateGame) -> Game:
    return Game(
        game_id=f"espn_{game.event_id}",
        game_date=game.game_date,
        season_start=game.season_start,
        home_team=game.home_team,
        away_team=game.away_team,
        home_score=0,
        away_score=0,
    )


def score_slate(target: date, *, today: date | None = None) -> pd.DataFrame:
    """Model probabilities for one slate date, with DraftKings prices when the game is not final."""

    current = today or date.today()
    ensure_scoreboards(max(target, current))
    path = cache_path(target)
    if not path.exists():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    listed = games_on_scoreboard(payload, target)
    if target < current:
        selected = listed
        history = completed_games_before(target)
    else:
        selected = [game for game in listed if game.status != "STATUS_FINAL"]
        history = completed_games_before(target, include_same_day_finals=(target == current))

    if not selected:
        return pd.DataFrame()

    if target >= current:
        for game in selected:
            if game.home_odds is not None and game.away_odds is not None:
                continue
            line = cached_summary_line(game.event_id)
            if line is None:
                continue
            game.home_odds = _optional_float(line.get("home_odds"))
            game.away_odds = _optional_float(line.get("away_odds"))
            game.provider = str(line.get("provider") or game.provider)
            game.spread = str(line.get("spread") or game.spread)

    features = snapshot_features(history, [_placeholder(game) for game in selected])
    if features.empty:
        return features
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Fit the model first.")
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    columns = list(bundle.get("feature_columns") or FEATURE_COLUMNS)
    features = features.copy()
    features["p_home_win"] = pd.NA
    eligible = features["eligible"] == 1
    if eligible.any():
        features.loc[eligible, "p_home_win"] = pipeline.predict_proba(features.loc[eligible, list(columns)])[:, 1]
    features["p_home_win"] = pd.to_numeric(features["p_home_win"], errors="coerce")

    meta = pd.DataFrame(
        [
            {
                "game_id": f"espn_{game.event_id}",
                "away_name": game.away_name,
                "home_name": game.home_name,
                "status": game.status,
                "status_detail": game.status_detail,
                "home_odds": game.home_odds,
                "away_odds": game.away_odds,
                "provider": game.provider,
                "spread": game.spread,
                "live_home_score": game.home_score,
                "live_away_score": game.away_score,
            }
            for game in selected
        ]
    )
    merged = features.drop(columns=["home_score", "away_score", "home_win"]).merge(meta, on="game_id", how="left")
    merged["home_score"] = merged["live_home_score"]
    merged["away_score"] = merged["live_away_score"]
    if target < current:
        merged["home_win"] = (merged["home_score"] > merged["away_score"]).astype("float")
    else:
        merged["home_win"] = pd.NA
    return merged.sort_values("game_id").reset_index(drop=True)
