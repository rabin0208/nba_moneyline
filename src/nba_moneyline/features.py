"""Leakage-safe pre-game features.

History is every regular-season game with an earlier date. All games on a
date are read before any of that date's results update team state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from nba_moneyline.constants import (
    ELO_INITIAL,
    ELO_K,
    ELO_SCALE,
    FEATURE_COLUMNS,
    MIN_PRIOR_GAMES,
    RECENT_WINDOW,
    ROTATION_PRIOR_MINUTES,
    VENUE_RECENT_WINDOW,
    WIN_RATE_HALF_LIFE_GAMES,
)
from nba_moneyline.player_box import PlayerLine, RotationBook
from nba_moneyline.schema import Game


def game_weight(games_ago: int, half_life: int = WIN_RATE_HALF_LIFE_GAMES) -> float:
    """Age 0 is the most recent prior game. Age 82 has half that weight."""

    if games_ago < 0:
        raise ValueError("games_ago must be >= 0")
    return 0.5 ** (games_ago / half_life)


def decayed_win_rate(results: list[int], half_life: int = WIN_RATE_HALF_LIFE_GAMES) -> float:
    if not results:
        return 0.5
    numerator = 0.0
    denominator = 0.0
    last = len(results) - 1
    for index, won in enumerate(results):
        weight = game_weight(last - index, half_life)
        numerator += weight * won
        denominator += weight
    return numerator / denominator


def _mean_last(results: list[int], window: int) -> float:
    if not results:
        return 0.5
    sample = results[-window:]
    return sum(sample) / len(sample)


def _mean_all(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / ELO_SCALE))


def updated_ratings(home_rating: float, away_rating: float, home_won: bool) -> tuple[float, float]:
    home_expected = expected_score(home_rating, away_rating)
    home_score = 1.0 if home_won else 0.0
    new_home = home_rating + ELO_K * (home_score - home_expected)
    new_away = away_rating + ELO_K * ((1.0 - home_score) - (1.0 - home_expected))
    return new_home, new_away


def shot_quality_share(fg_value_for: float, fga_for: float, fg_value_against: float, fga_against: float) -> float | None:
    if fga_for <= 0 or fga_against <= 0:
        return None
    efg_for = fg_value_for / fga_for
    efg_against = fg_value_against / fga_against
    total = efg_for + efg_against
    if total <= 0:
        return None
    return efg_for / total


def field_goal_value(fg: int, fg3: int) -> float:
    return fg + 0.5 * fg3


@dataclass
class TeamState:
    results: list[int] = field(default_factory=list)
    points_for: list[float] = field(default_factory=list)
    points_against: list[float] = field(default_factory=list)
    home_results: list[int] = field(default_factory=list)
    away_results: list[int] = field(default_factory=list)
    season: int | None = None
    season_games: int = 0
    season_wins: int = 0
    fg_value_for: float = 0.0
    fga_for: float = 0.0
    fg_value_against: float = 0.0
    fga_against: float = 0.0
    last_game_date: date | None = None
    last_game_season: int | None = None
    elo: float = ELO_INITIAL


def _season_snapshot(state: TeamState, season_start: int) -> tuple[float, float]:
    if state.season != season_start or state.season_games == 0:
        return 0.0, 0.5
    return float(state.season_games), state.season_wins / state.season_games


def _back_to_back(state: TeamState, game_date: date, season_start: int) -> int:
    if state.last_game_date is None or state.last_game_season != season_start:
        return 0
    return int((game_date - state.last_game_date).days == 1)


def _side_features(state: TeamState, game_date: date, season_start: int, at_home: bool) -> dict[str, float | None]:
    season_games, season_win_rate = _season_snapshot(state, season_start)
    venue_results = state.home_results if at_home else state.away_results
    return {
        "win_rate": decayed_win_rate(state.results),
        "points_for_per_game": _mean_all(state.points_for),
        "points_against_per_game": _mean_all(state.points_against),
        "recent_win_rate": _mean_last(state.results, RECENT_WINDOW),
        "venue_win_rate": _mean_last(venue_results, len(venue_results) or 1) if venue_results else 0.5,
        "recent_venue_win_rate": _mean_last(venue_results, VENUE_RECENT_WINDOW),
        "season_games": season_games,
        "season_win_rate": season_win_rate,
        "shot_quality_share": shot_quality_share(
            state.fg_value_for,
            state.fga_for,
            state.fg_value_against,
            state.fga_against,
        ),
        "back_to_back": _back_to_back(state, game_date, season_start),
        "prior_games": len(state.results),
    }


def _apply_game(state: TeamState, game: Game, *, is_home: bool) -> None:
    won = game.home_win if is_home else 1 - game.home_win
    points_for = game.home_score if is_home else game.away_score
    points_against = game.away_score if is_home else game.home_score
    fg = game.home_fg if is_home else game.away_fg
    fga = game.home_fga if is_home else game.away_fga
    fg3 = game.home_fg3 if is_home else game.away_fg3
    opp_fg = game.away_fg if is_home else game.home_fg
    opp_fga = game.away_fga if is_home else game.home_fga
    opp_fg3 = game.away_fg3 if is_home else game.home_fg3

    if state.season != game.season_start:
        state.season = game.season_start
        state.season_games = 0
        state.season_wins = 0
    state.results.append(won)
    state.points_for.append(float(points_for))
    state.points_against.append(float(points_against))
    if is_home:
        state.home_results.append(won)
    else:
        state.away_results.append(won)
    state.season_games += 1
    state.season_wins += won
    if game.has_shooting:
        state.fg_value_for += field_goal_value(fg, fg3)
        state.fga_for += fga
        state.fg_value_against += field_goal_value(opp_fg, opp_fg3)
        state.fga_against += opp_fga
    state.last_game_date = game.game_date
    state.last_game_season = game.season_start


def _row(game: Game, states: dict[str, TeamState], rotations: RotationBook) -> dict[str, object]:
    home = _side_features(states[game.home_team], game.game_date, game.season_start, True)
    away = _side_features(states[game.away_team], game.game_date, game.season_start, False)
    home_share = home["shot_quality_share"]
    away_share = away["shot_quality_share"]
    eligible = int(
        home["prior_games"] >= MIN_PRIOR_GAMES
        and away["prior_games"] >= MIN_PRIOR_GAMES
        and home_share is not None
        and away_share is not None
    )
    values: dict[str, object] = {
        "game_id": game.game_id,
        "game_date": game.game_date.isoformat(),
        "season_start": game.season_start,
        "home_team": game.home_team,
        "away_team": game.away_team,
        "home_score": game.home_score,
        "away_score": game.away_score,
        "home_win": game.home_win,
        "eligible": eligible,
        "home_win_rate": home["win_rate"],
        "away_win_rate": away["win_rate"],
        "home_points_for_per_game": home["points_for_per_game"],
        "away_points_for_per_game": away["points_for_per_game"],
        "home_points_against_per_game": home["points_against_per_game"],
        "away_points_against_per_game": away["points_against_per_game"],
        "home_recent_win_rate": home["recent_win_rate"],
        "away_recent_win_rate": away["recent_win_rate"],
        "home_home_win_rate": home["venue_win_rate"],
        "away_away_win_rate": away["venue_win_rate"],
        "home_recent_home_win_rate": home["recent_venue_win_rate"],
        "away_recent_away_win_rate": away["recent_venue_win_rate"],
        "home_season_games": home["season_games"],
        "away_season_games": away["season_games"],
        "home_season_win_rate": home["season_win_rate"],
        "away_season_win_rate": away["season_win_rate"],
        "home_elo_minus_away_elo": states[game.home_team].elo - states[game.away_team].elo,
        "home_shot_quality_share": home_share,
        "away_shot_quality_share": away_share,
        "home_back_to_back": home["back_to_back"],
        "away_back_to_back": away["back_to_back"],
        "home_rotation_delta": rotations.delta(game.home_team, game.season_start),
        "away_rotation_delta": rotations.delta(game.away_team, game.season_start),
    }
    return values


def _update(game: Game, states: dict[str, TeamState]) -> None:
    home_rating = states[game.home_team].elo
    away_rating = states[game.away_team].elo
    new_home, new_away = updated_ratings(home_rating, away_rating, bool(game.home_win))
    _apply_game(states[game.home_team], game, is_home=True)
    _apply_game(states[game.away_team], game, is_home=False)
    states[game.home_team].elo = new_home
    states[game.away_team].elo = new_away


def _walk(
    games: list[Game],
    player_lines: dict[str, list[PlayerLine]] | None = None,
    prior_minutes: float = ROTATION_PRIOR_MINUTES,
) -> tuple[list[dict[str, object]], dict[str, TeamState], RotationBook]:
    ordered = sorted(games, key=lambda game: (game.game_date, game.game_id))
    lines = player_lines or {}
    states: dict[str, TeamState] = defaultdict(TeamState)
    rotations = RotationBook(prior_minutes=prior_minutes)
    rows: list[dict[str, object]] = []
    index = 0
    while index < len(ordered):
        day = ordered[index].game_date
        end = index + 1
        while end < len(ordered) and ordered[end].game_date == day:
            end += 1
        day_games = ordered[index:end]
        for game in day_games:
            rows.append(_row(game, states, rotations))
        for game in day_games:
            _update(game, states)
            game_lines = lines.get(game.game_id, [])
            rotations.apply(game.home_team, game.season_start, game_lines)
            rotations.apply(game.away_team, game.season_start, game_lines)
        index = end
    return rows, states, rotations


def _as_feature_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"feature frame missing columns: {missing}")
    return frame


def build_feature_frame(
    games: list[Game],
    player_lines: dict[str, list[PlayerLine]] | None = None,
    prior_minutes: float = ROTATION_PRIOR_MINUTES,
) -> pd.DataFrame:
    """Return one row per game, including ineligible rows."""

    rows, _states, _rotations = _walk(games, player_lines, prior_minutes)
    return _as_feature_frame(rows)


def snapshot_features(
    history: list[Game],
    slate: list[Game],
    player_lines: dict[str, list[PlayerLine]] | None = None,
    prior_minutes: float = ROTATION_PRIOR_MINUTES,
) -> pd.DataFrame:
    """Features for slate games. Slate scores and slate box scores are not applied."""

    _rows, states, rotations = _walk(history, player_lines, prior_minutes)
    ordered = sorted(slate, key=lambda game: (game.game_date, game.game_id))
    return _as_feature_frame([_row(game, states, rotations) for game in ordered])
