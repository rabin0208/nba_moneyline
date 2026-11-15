"""Leakage-safe pre-game features.

History is every regular-season game with an earlier date. All games on a
date are read before any of that date's results update team state.

Counts, rates, points, shot quality, and Elo use the current season and the
three seasons before it. The last-10 win rate, the current-season record, and
fatigue follow the hockey model. Overtime and shootout features are omitted.
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
    PRIOR_SEASON_WINDOW,
    RECENT_FORM_WINDOW,
)
from nba_moneyline.schema import Game


def seasons_in_prior_window(season_start: int, window: int = PRIOR_SEASON_WINDOW) -> frozenset[int]:
    """The current season and the seasons immediately before it."""

    return frozenset(season_start - offset for offset in range(window))


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / ELO_SCALE))


def updated_ratings(home_rating: float, away_rating: float, home_won: bool) -> tuple[float, float]:
    home_expected = expected_score(home_rating, away_rating)
    home_score = 1.0 if home_won else 0.0
    new_home = home_rating + ELO_K * (home_score - home_expected)
    new_away = away_rating + ELO_K * ((1.0 - home_score) - (1.0 - home_expected))
    return new_home, new_away


@dataclass
class SeasonBucket:
    games: int = 0
    wins: int = 0
    home_games: int = 0
    home_wins: int = 0
    away_games: int = 0
    away_wins: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    fg_value_for: float = 0.0
    fga_for: float = 0.0
    fg_value_against: float = 0.0
    fga_against: float = 0.0
    fg_made_for: float = 0.0
    fg3_made_for: float = 0.0
    fg3a_for: float = 0.0
    rebounds_for: float = 0.0
    rebound_games: int = 0
    turnovers_for: float = 0.0
    turnover_games: int = 0


@dataclass(frozen=True, slots=True)
class EloEvent:
    season_start: int
    home_team: str
    away_team: str
    home_won: bool


@dataclass
class TeamState:
    season_buckets: dict[int, SeasonBucket] = field(default_factory=dict)
    game_dates: list[date] = field(default_factory=list)
    recent_results: list[int] = field(default_factory=list)


def _window_bucket(state: TeamState, season_start: int) -> SeasonBucket:
    allowed = seasons_in_prior_window(season_start)
    total = SeasonBucket()
    for season, bucket in state.season_buckets.items():
        if season not in allowed:
            continue
        total.games += bucket.games
        total.wins += bucket.wins
        total.home_games += bucket.home_games
        total.home_wins += bucket.home_wins
        total.away_games += bucket.away_games
        total.away_wins += bucket.away_wins
        total.points_for += bucket.points_for
        total.points_against += bucket.points_against
        total.fg_value_for += bucket.fg_value_for
        total.fga_for += bucket.fga_for
        total.fg_value_against += bucket.fg_value_against
        total.fga_against += bucket.fga_against
        total.fg_made_for += bucket.fg_made_for
        total.fg3_made_for += bucket.fg3_made_for
        total.fg3a_for += bucket.fg3a_for
        total.rebounds_for += bucket.rebounds_for
        total.rebound_games += bucket.rebound_games
        total.turnovers_for += bucket.turnovers_for
        total.turnover_games += bucket.turnover_games
    return total


def _rate(wins: int, games: int) -> float:
    if games == 0:
        return 0.5
    return wins / games


def _per_game(total: float, games: int) -> float:
    if games == 0:
        return 0.0
    return total / games


def _pct(made: float, attempts: float) -> float | None:
    if attempts <= 0:
        return None
    return made / attempts


def _recent_win_rate(results: list[int]) -> float:
    if not results:
        return 0.5
    sample = results[-RECENT_FORM_WINDOW:]
    return sum(sample) / len(sample)


def shot_quality_share(fg_value_for: float, fga_for: float, fg_value_against: float, fga_against: float) -> float | None:
    """Attempt-weighted effective field-goal share, the stand-in for hockey xGoal share."""

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


def _fatigue(game_dates: list[date], game_date: date) -> tuple[float, float]:
    previous = game_dates[-1] if game_dates else None
    back_to_back = float(previous is not None and (game_date - previous).days == 1)
    prior_7d = sum(1 for day in game_dates if 1 <= (game_date - day).days <= 7)
    return back_to_back, float(prior_7d)


def ratings_from_events(events: list[EloEvent], season_start: int) -> dict[str, float]:
    """Replay Elo from games inside the four-season window. Every team starts at 1500."""

    allowed = seasons_in_prior_window(season_start)
    ratings: dict[str, float] = defaultdict(lambda: ELO_INITIAL)
    for event in events:
        if event.season_start not in allowed:
            continue
        new_home, new_away = updated_ratings(
            ratings[event.home_team],
            ratings[event.away_team],
            event.home_won,
        )
        ratings[event.home_team] = new_home
        ratings[event.away_team] = new_away
    return ratings


def _side(state: TeamState, season_start: int, *, at_home: bool) -> dict[str, float | None]:
    window = _window_bucket(state, season_start)
    current = state.season_buckets.get(season_start, SeasonBucket())
    venue_wins = window.home_wins if at_home else window.away_wins
    venue_games = window.home_games if at_home else window.away_games
    return {
        "prior_games": float(window.games),
        "win_rate": _rate(window.wins, window.games),
        "points_for_per_game": _per_game(window.points_for, window.games),
        "points_against_per_game": _per_game(window.points_against, window.games),
        "venue_win_rate": _rate(venue_wins, venue_games),
        "season_games": float(current.games),
        "season_win_rate": _rate(current.wins, current.games),
        "recent_win_rate": _recent_win_rate(state.recent_results),
        "shot_quality_share": shot_quality_share(
            window.fg_value_for,
            window.fga_for,
            window.fg_value_against,
            window.fga_against,
        ),
        "field_goal_pct": _pct(window.fg_made_for, window.fga_for),
        "three_point_pct": _pct(window.fg3_made_for, window.fg3a_for),
        "turnovers_per_game": _pct(window.turnovers_for, window.turnover_games),
        "rebounds_per_game": _pct(window.rebounds_for, window.rebound_games),
    }


def _apply(state: TeamState, game: Game, *, is_home: bool) -> None:
    won = game.home_win if is_home else 1 - game.home_win
    points_for = game.home_score if is_home else game.away_score
    points_against = game.away_score if is_home else game.home_score
    bucket = state.season_buckets.setdefault(game.season_start, SeasonBucket())
    bucket.games += 1
    bucket.wins += won
    bucket.points_for += float(points_for)
    bucket.points_against += float(points_against)
    if is_home:
        bucket.home_games += 1
        bucket.home_wins += won
    else:
        bucket.away_games += 1
        bucket.away_wins += won
    if game.has_shooting:
        fg = game.home_fg if is_home else game.away_fg
        fga = game.home_fga if is_home else game.away_fga
        fg3 = game.home_fg3 if is_home else game.away_fg3
        opp_fg = game.away_fg if is_home else game.home_fg
        opp_fga = game.away_fga if is_home else game.home_fga
        opp_fg3 = game.away_fg3 if is_home else game.home_fg3
        bucket.fg_value_for += field_goal_value(fg, fg3)
        bucket.fga_for += fga
        bucket.fg_made_for += fg
        bucket.fg_value_against += field_goal_value(opp_fg, opp_fg3)
        bucket.fga_against += opp_fga
    fg3 = game.home_fg3 if is_home else game.away_fg3
    fg3a = game.home_fg3a if is_home else game.away_fg3a
    if fg3 is not None and fg3a is not None:
        bucket.fg3_made_for += fg3
        bucket.fg3a_for += fg3a
    rebounds = game.home_rebounds if is_home else game.away_rebounds
    if rebounds is not None:
        bucket.rebounds_for += rebounds
        bucket.rebound_games += 1
    turnovers = game.home_turnovers if is_home else game.away_turnovers
    if turnovers is not None:
        bucket.turnovers_for += turnovers
        bucket.turnover_games += 1
    state.recent_results.append(won)
    state.game_dates.append(game.game_date)


def _row(game: Game, states: dict[str, TeamState], ratings: dict[str, float]) -> dict[str, object]:
    home = _side(states[game.home_team], game.season_start, at_home=True)
    away = _side(states[game.away_team], game.season_start, at_home=False)
    home_b2b, home_7d = _fatigue(states[game.home_team].game_dates, game.game_date)
    away_b2b, away_7d = _fatigue(states[game.away_team].game_dates, game.game_date)
    home_share = home["shot_quality_share"]
    away_share = away["shot_quality_share"]
    def _ready(side: dict[str, float]) -> bool:
        return all(
            side[name] is not None
            for name in (
                "shot_quality_share",
                "field_goal_pct",
                "three_point_pct",
                "turnovers_per_game",
                "rebounds_per_game",
            )
        )

    eligible = int(
        home["prior_games"] >= MIN_PRIOR_GAMES
        and away["prior_games"] >= MIN_PRIOR_GAMES
        and _ready(home)
        and _ready(away)
    )
    return {
        "game_id": game.game_id,
        "game_date": game.game_date.isoformat(),
        "season_start": game.season_start,
        "home_team": game.home_team,
        "away_team": game.away_team,
        "home_score": game.home_score,
        "away_score": game.away_score,
        "home_win": game.home_win,
        "eligible": eligible,
        "home_prior_games": home["prior_games"],
        "away_prior_games": away["prior_games"],
        "home_win_rate": home["win_rate"],
        "away_win_rate": away["win_rate"],
        "home_points_for_per_game": home["points_for_per_game"],
        "away_points_for_per_game": away["points_for_per_game"],
        "home_points_against_per_game": home["points_against_per_game"],
        "away_points_against_per_game": away["points_against_per_game"],
        "home_field_goal_pct": home["field_goal_pct"],
        "away_field_goal_pct": away["field_goal_pct"],
        "home_three_point_pct": home["three_point_pct"],
        "away_three_point_pct": away["three_point_pct"],
        "home_turnovers_per_game": home["turnovers_per_game"],
        "away_turnovers_per_game": away["turnovers_per_game"],
        "home_rebounds_per_game": home["rebounds_per_game"],
        "away_rebounds_per_game": away["rebounds_per_game"],
        "home_home_win_rate": home["venue_win_rate"],
        "away_away_win_rate": away["venue_win_rate"],
        "home_season_games": home["season_games"],
        "away_season_games": away["season_games"],
        "home_season_win_rate": home["season_win_rate"],
        "away_season_win_rate": away["season_win_rate"],
        "home_recent_win_rate": home["recent_win_rate"],
        "away_recent_win_rate": away["recent_win_rate"],
        "home_elo_minus_away_elo": ratings[game.home_team] - ratings[game.away_team],
        "home_shot_quality_share": home_share,
        "away_shot_quality_share": away_share,
        "home_back_to_back": home_b2b,
        "away_back_to_back": away_b2b,
        "home_games_prior_7d": home_7d,
        "away_games_prior_7d": away_7d,
    }


def _walk(games: list[Game]) -> tuple[list[dict[str, object]], dict[str, TeamState], list[EloEvent]]:
    ordered = sorted(games, key=lambda game: (game.game_date, game.game_id))
    states: dict[str, TeamState] = defaultdict(TeamState)
    events: list[EloEvent] = []
    rows: list[dict[str, object]] = []
    index = 0
    while index < len(ordered):
        day = ordered[index].game_date
        end = index + 1
        while end < len(ordered) and ordered[end].game_date == day:
            end += 1
        day_games = ordered[index:end]
        ratings_by_season = {
            season: ratings_from_events(events, season)
            for season in {game.season_start for game in day_games}
        }
        for game in day_games:
            rows.append(_row(game, states, ratings_by_season[game.season_start]))
        for game in day_games:
            _apply(states[game.home_team], game, is_home=True)
            _apply(states[game.away_team], game, is_home=False)
            events.append(
                EloEvent(
                    season_start=game.season_start,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    home_won=bool(game.home_win),
                )
            )
        index = end
    return rows, states, events


def _as_feature_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"feature frame missing columns: {missing}")
    return frame


def build_feature_frame(games: list[Game]) -> pd.DataFrame:
    """Return one row per game, including ineligible rows."""

    rows, _states, _events = _walk(games)
    return _as_feature_frame(rows)


def snapshot_features(history: list[Game], slate: list[Game]) -> pd.DataFrame:
    """Features for slate games. Slate scores are not applied."""

    _rows, states, events = _walk(history)
    ordered = sorted(slate, key=lambda game: (game.game_date, game.game_id))
    ratings_by_season = {
        season: ratings_from_events(events, season) for season in {game.season_start for game in ordered}
    }
    return _as_feature_frame(
        [_row(game, states, ratings_by_season[game.season_start]) for game in ordered]
    )
