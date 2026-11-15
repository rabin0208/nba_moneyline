"""Leakage-safe pre-game features.

History is every regular-season game with an earlier date. All games on a
date are read before any of that date's results update team state.

Strength rates use the previous regular season and the current season shrunk
toward it. Elo still replays the current season and the three before it.
Scoring, turnovers, and rebounds are per possession. The last-10 window keeps
the win rate, an Elo residual, and opponent-adjusted offensive and defensive
ratings. Fatigue follows the hockey model. Overtime and shootout features are
omitted.
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
    SHRINK_PSEUDO_GAMES,
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
    fg_value_for: float = 0.0
    fga_for: float = 0.0
    rated_points_for: float = 0.0
    rated_points_against: float = 0.0
    poss_for: float = 0.0
    poss_against: float = 0.0
    turnover_numer: float = 0.0
    turnover_denom: float = 0.0
    rebounds_for: float = 0.0
    rebounds_against: float = 0.0
    rated_games: int = 0
    shooting_games: int = 0
    rebound_games: int = 0


@dataclass
class LeagueSeason:
    """Pooled results for one season. Used as the prior when a team has no season of its own."""

    games: int = 0
    home_wins: int = 0
    points: float = 0.0
    possessions: float = 0.0
    fg_value: float = 0.0
    fga: float = 0.0
    turnover_numer: float = 0.0
    turnover_denom: float = 0.0


@dataclass(frozen=True, slots=True)
class EloEvent:
    season_start: int
    home_team: str
    away_team: str
    home_won: bool


@dataclass(frozen=True, slots=True)
class RecentGame:
    won: int
    residual: float
    offensive_adj: float | None
    defensive_adj: float | None


@dataclass
class TeamState:
    season_buckets: dict[int, SeasonBucket] = field(default_factory=dict)
    game_dates: list[date] = field(default_factory=list)
    recent_games: list[RecentGame] = field(default_factory=list)


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
        total.fg_value_for += bucket.fg_value_for
        total.fga_for += bucket.fga_for
        total.rated_points_for += bucket.rated_points_for
        total.rated_points_against += bucket.rated_points_against
        total.poss_for += bucket.poss_for
        total.poss_against += bucket.poss_against
        total.turnover_numer += bucket.turnover_numer
        total.turnover_denom += bucket.turnover_denom
        total.rebounds_for += bucket.rebounds_for
        total.rebounds_against += bucket.rebounds_against
        total.rated_games += bucket.rated_games
        total.shooting_games += bucket.shooting_games
        total.rebound_games += bucket.rebound_games
    return total


# Free throws are worth about 0.44 of a possession in the standard estimate.
FTA_POSSESSION_WEIGHT = 0.44


def possessions(fga: float, offensive_rebounds: float, turnovers: float, fta: float) -> float:
    """Estimated possessions: field-goal attempts, offensive rebounds, turnovers, free throws."""

    return fga - offensive_rebounds + turnovers + FTA_POSSESSION_WEIGHT * fta


def _per_hundred(points: float, poss: float) -> float | None:
    if poss <= 0:
        return None
    return 100.0 * points / poss


def _pct(made: float, attempts: float) -> float | None:
    if attempts <= 0:
        return None
    return made / attempts


def _diff(home: float | None, away: float | None) -> float | None:
    """Home minus away. Missing on either side stays missing."""

    if home is None or away is None:
        return None
    return home - away


def _observed_rate(wins: int, games: int) -> float | None:
    if games == 0:
        return None
    return wins / games


def _shrink(
    current: float | None,
    current_games: int,
    previous: float | None,
    prior: float | None,
) -> float | None:
    """Pull a current-season rate toward the previous season by a fixed number of games.

    With no current games the value is the previous season. A team with no previous
    season uses the league prior. With neither, the current rate stands on its own.
    """

    anchor = previous if previous is not None else prior
    if current_games <= 0 or current is None:
        return anchor
    if anchor is None:
        return current
    return (current_games * current + SHRINK_PSEUDO_GAMES * anchor) / (
        current_games + SHRINK_PSEUDO_GAMES
    )


def _bucket_rates(bucket: SeasonBucket) -> dict[str, float | None]:
    venue_home = _observed_rate(bucket.home_wins, bucket.home_games)
    venue_away = _observed_rate(bucket.away_wins, bucket.away_games)
    if bucket.games > 0 and bucket.home_games == 0:
        venue_home = 0.5
    if bucket.games > 0 and bucket.away_games == 0:
        venue_away = 0.5
    return {
        "games": float(bucket.games),
        "win_rate": _observed_rate(bucket.wins, bucket.games),
        "offensive_rating": _per_hundred(bucket.rated_points_for, bucket.poss_for),
        "defensive_rating": _per_hundred(bucket.rated_points_against, bucket.poss_against),
        "efg_pct": _pct(bucket.fg_value_for, bucket.fga_for),
        "turnover_pct": _pct(bucket.turnover_numer, bucket.turnover_denom),
        "rebound_pct": _pct(bucket.rebounds_for, bucket.rebounds_for + bucket.rebounds_against),
        "rated_games": bucket.rated_games,
        "shooting_games": bucket.shooting_games,
        "rebound_games": bucket.rebound_games,
        "home_games": bucket.home_games,
        "away_games": bucket.away_games,
        "home_win_rate": venue_home,
        "away_win_rate": venue_away,
    }


def _league_prior(bucket: LeagueSeason | None) -> dict[str, float | None]:
    if bucket is None or bucket.games == 0:
        return {
            "win_rate": 0.5,
            "home_win_rate": None,
            "away_win_rate": None,
            "offensive_rating": None,
            "defensive_rating": None,
            "efg_pct": None,
            "turnover_pct": None,
            "rebound_pct": 0.5,
        }
    home_win_rate = bucket.home_wins / bucket.games
    rating = _per_hundred(bucket.points, bucket.possessions)
    return {
        "win_rate": 0.5,
        "home_win_rate": home_win_rate,
        "away_win_rate": 1.0 - home_win_rate,
        "offensive_rating": rating,
        "defensive_rating": rating,
        "efg_pct": _pct(bucket.fg_value, bucket.fga),
        "turnover_pct": _pct(bucket.turnover_numer, bucket.turnover_denom),
        "rebound_pct": 0.5,
    }


def _recent_sample(games: list[RecentGame]) -> list[RecentGame]:
    return games[-RECENT_FORM_WINDOW:]


def _recent_win_rate(games: list[RecentGame]) -> float:
    sample = _recent_sample(games)
    if not sample:
        return 0.5
    return sum(game.won for game in sample) / len(sample)


def _recent_residual(games: list[RecentGame]) -> float:
    """Mean of win minus pregame Elo win probability. No games is 0."""

    sample = _recent_sample(games)
    if not sample:
        return 0.0
    return sum(game.residual for game in sample) / len(sample)


def _recent_adjusted(games: list[RecentGame], *, offensive: bool) -> float | None:
    """Mean opponent-adjusted rating inside the last 10 games.

    A game with no box line, or an opponent who had no pregame rating, is left
    out. The mean is missing when none of the last 10 games qualify.
    """

    sample = _recent_sample(games)
    values = [game.offensive_adj if offensive else game.defensive_adj for game in sample]
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _game_ratings(game: Game, is_home: bool) -> tuple[float | None, float | None]:
    """This game's offensive and defensive rating. Missing when the box is incomplete."""

    points_for = game.home_score if is_home else game.away_score
    points_against = game.away_score if is_home else game.home_score
    own_fga, own_orb, own_tov, own_fta, _own_rebounds = _box_line(game, is_home)
    opp_fga, opp_orb, opp_tov, opp_fta, _opp_rebounds = _box_line(game, not is_home)
    if None in (own_fga, own_orb, own_tov, own_fta, opp_fga, opp_orb, opp_tov, opp_fta):
        return None, None
    offensive = _per_hundred(float(points_for), possessions(own_fga, own_orb, own_tov, own_fta))
    defensive = _per_hundred(float(points_against), possessions(opp_fga, opp_orb, opp_tov, opp_fta))
    return offensive, defensive


def _adjusted(game_rating: float | None, opponent_rating: float | None) -> float | None:
    if game_rating is None or opponent_rating is None:
        return None
    return game_rating - opponent_rating


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


def _window_ratings(state: TeamState, season_start: int) -> tuple[float | None, float | None]:
    """Four-season offensive and defensive rating, for the last-10 opponent adjustment."""

    window = _window_bucket(state, season_start)
    return (
        _per_hundred(window.rated_points_for, window.poss_for),
        _per_hundred(window.rated_points_against, window.poss_against),
    )


def _profile(
    state: TeamState,
    season_start: int,
    *,
    at_home: bool,
    prior: dict[str, float | None],
) -> dict[str, float | None]:
    previous = _bucket_rates(state.season_buckets.get(season_start - 1, SeasonBucket()))
    current = _bucket_rates(state.season_buckets.get(season_start, SeasonBucket()))
    venue_key = "home_win_rate" if at_home else "away_win_rate"
    venue_games_key = "home_games" if at_home else "away_games"
    venue_prior_key = "home_win_rate" if at_home else "away_win_rate"

    def shrunk(name: str, games_key: str, prior_key: str) -> float | None:
        return _shrink(
            current[name],
            int(current[games_key]),
            previous[name],
            prior[prior_key],
        )

    return {
        "prev_games": previous["games"],
        "prev_win_rate": previous["win_rate"],
        "prev_offensive_rating": previous["offensive_rating"],
        "prev_defensive_rating": previous["defensive_rating"],
        "prev_efg_pct": previous["efg_pct"],
        "prev_turnover_pct": previous["turnover_pct"],
        "prev_rebound_pct": previous["rebound_pct"],
        "prev_venue_win_rate": previous[venue_key],
        "season_games": current["games"],
        "season_win_rate": shrunk("win_rate", "games", "win_rate"),
        "season_offensive_rating": shrunk("offensive_rating", "rated_games", "offensive_rating"),
        "season_defensive_rating": shrunk("defensive_rating", "rated_games", "defensive_rating"),
        "season_efg_pct": shrunk("efg_pct", "shooting_games", "efg_pct"),
        "season_turnover_pct": shrunk("turnover_pct", "rated_games", "turnover_pct"),
        "season_rebound_pct": shrunk("rebound_pct", "rebound_games", "rebound_pct"),
        "season_venue_win_rate": shrunk(venue_key, venue_games_key, venue_prior_key),
        "recent_win_rate": _recent_win_rate(state.recent_games),
        "recent_residual": _recent_residual(state.recent_games),
        "recent_offensive_rating": _recent_adjusted(state.recent_games, offensive=True),
        "recent_defensive_rating": _recent_adjusted(state.recent_games, offensive=False),
    }


def _box_line(game: Game, is_home: bool) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    if is_home:
        return game.home_fga, game.home_orb, game.home_turnovers, game.home_fta, game.home_rebounds
    return game.away_fga, game.away_orb, game.away_turnovers, game.away_fta, game.away_rebounds


def _apply(
    state: TeamState,
    game: Game,
    *,
    is_home: bool,
    own_elo: float,
    opponent_elo: float,
    opponent_offensive_rating: float | None,
    opponent_defensive_rating: float | None,
) -> None:
    won = game.home_win if is_home else 1 - game.home_win
    points_for = game.home_score if is_home else game.away_score
    points_against = game.away_score if is_home else game.home_score
    bucket = state.season_buckets.setdefault(game.season_start, SeasonBucket())
    bucket.games += 1
    bucket.wins += won
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
        bucket.fg_value_for += field_goal_value(fg, fg3)
        bucket.fga_for += fga
        bucket.shooting_games += 1
    own_fga, own_orb, own_tov, own_fta, own_reb = _box_line(game, is_home)
    opp_fga, opp_orb, opp_tov, opp_fta, opp_reb = _box_line(game, not is_home)
    if None not in (own_fga, own_orb, own_tov, own_fta, opp_fga, opp_orb, opp_tov, opp_fta):
        bucket.poss_for += possessions(own_fga, own_orb, own_tov, own_fta)
        bucket.poss_against += possessions(opp_fga, opp_orb, opp_tov, opp_fta)
        bucket.rated_points_for += float(points_for)
        bucket.rated_points_against += float(points_against)
        bucket.turnover_numer += own_tov
        bucket.turnover_denom += own_fga + FTA_POSSESSION_WEIGHT * own_fta + own_tov
        bucket.rated_games += 1
    if own_reb is not None and opp_reb is not None:
        bucket.rebounds_for += own_reb
        bucket.rebounds_against += opp_reb
        bucket.rebound_games += 1
    offensive, defensive = _game_ratings(game, is_home)
    state.recent_games.append(
        RecentGame(
            won=won,
            residual=won - expected_score(own_elo, opponent_elo),
            offensive_adj=_adjusted(offensive, opponent_defensive_rating),
            defensive_adj=_adjusted(defensive, opponent_offensive_rating),
        )
    )
    state.game_dates.append(game.game_date)


def _accumulate_league(bucket: LeagueSeason, game: Game) -> None:
    bucket.games += 1
    bucket.home_wins += game.home_win
    if game.has_shooting:
        bucket.fg_value += field_goal_value(game.home_fg, game.home_fg3) + field_goal_value(
            game.away_fg, game.away_fg3
        )
        bucket.fga += game.home_fga + game.away_fga
    home_line = _box_line(game, True)
    away_line = _box_line(game, False)
    if None not in home_line[:4] and None not in away_line[:4]:
        home_poss = possessions(*home_line[:4])
        away_poss = possessions(*away_line[:4])
        bucket.points += float(game.home_score + game.away_score)
        bucket.possessions += home_poss + away_poss
        bucket.turnover_numer += home_line[2] + away_line[2]
        bucket.turnover_denom += (
            home_line[0] + FTA_POSSESSION_WEIGHT * home_line[3] + home_line[2]
            + away_line[0] + FTA_POSSESSION_WEIGHT * away_line[3] + away_line[2]
        )


def _row(
    game: Game,
    states: dict[str, TeamState],
    ratings: dict[str, float],
    prior: dict[str, float | None],
) -> dict[str, object]:
    home = _profile(states[game.home_team], game.season_start, at_home=True, prior=prior)
    away = _profile(states[game.away_team], game.season_start, at_home=False, prior=prior)
    home_b2b, home_7d = _fatigue(states[game.home_team].game_dates, game.game_date)
    away_b2b, away_7d = _fatigue(states[game.away_team].game_dates, game.game_date)

    def _ready(side: dict[str, float | None]) -> bool:
        return all(
            side[name] is not None
            for name in (
                "prev_offensive_rating",
                "prev_defensive_rating",
                "prev_efg_pct",
                "prev_turnover_pct",
                "prev_rebound_pct",
                "recent_offensive_rating",
                "recent_defensive_rating",
            )
        )

    eligible = int(
        home["prev_games"] >= MIN_PRIOR_GAMES
        and away["prev_games"] >= MIN_PRIOR_GAMES
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
        "home_prev_games": home["prev_games"],
        "away_prev_games": away["prev_games"],
        "prev_win_rate_diff": _diff(home["prev_win_rate"], away["prev_win_rate"]),
        "prev_offensive_rating_diff": _diff(home["prev_offensive_rating"], away["prev_offensive_rating"]),
        "prev_defensive_rating_diff": _diff(home["prev_defensive_rating"], away["prev_defensive_rating"]),
        "prev_efg_pct_diff": _diff(home["prev_efg_pct"], away["prev_efg_pct"]),
        "prev_turnover_pct_diff": _diff(home["prev_turnover_pct"], away["prev_turnover_pct"]),
        "prev_rebound_pct_diff": _diff(home["prev_rebound_pct"], away["prev_rebound_pct"]),
        "prev_home_home_win_rate": home["prev_venue_win_rate"],
        "prev_away_away_win_rate": away["prev_venue_win_rate"],
        "home_season_games": home["season_games"],
        "away_season_games": away["season_games"],
        "season_win_rate_diff": _diff(home["season_win_rate"], away["season_win_rate"]),
        "season_offensive_rating_diff": _diff(home["season_offensive_rating"], away["season_offensive_rating"]),
        "season_defensive_rating_diff": _diff(home["season_defensive_rating"], away["season_defensive_rating"]),
        "season_efg_pct_diff": _diff(home["season_efg_pct"], away["season_efg_pct"]),
        "season_turnover_pct_diff": _diff(home["season_turnover_pct"], away["season_turnover_pct"]),
        "season_rebound_pct_diff": _diff(home["season_rebound_pct"], away["season_rebound_pct"]),
        "season_home_home_win_rate": home["season_venue_win_rate"],
        "season_away_away_win_rate": away["season_venue_win_rate"],
        "recent_win_rate_diff": _diff(home["recent_win_rate"], away["recent_win_rate"]),
        "recent_result_residual_diff": _diff(home["recent_residual"], away["recent_residual"]),
        "recent_offensive_rating_diff": _diff(
            home["recent_offensive_rating"], away["recent_offensive_rating"]
        ),
        "recent_defensive_rating_diff": _diff(
            home["recent_defensive_rating"], away["recent_defensive_rating"]
        ),
        "home_elo_minus_away_elo": ratings[game.home_team] - ratings[game.away_team],
        "home_back_to_back": home_b2b,
        "away_back_to_back": away_b2b,
        "home_games_prior_7d": home_7d,
        "away_games_prior_7d": away_7d,
    }


def _walk(
    games: list[Game],
) -> tuple[list[dict[str, object]], dict[str, TeamState], list[EloEvent], dict[int, LeagueSeason]]:
    ordered = sorted(games, key=lambda game: (game.game_date, game.game_id))
    states: dict[str, TeamState] = defaultdict(TeamState)
    events: list[EloEvent] = []
    league: dict[int, LeagueSeason] = defaultdict(LeagueSeason)
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
            rows.append(
                _row(
                    game,
                    states,
                    ratings_by_season[game.season_start],
                    _league_prior(league.get(game.season_start - 1)),
                )
            )
        for game in day_games:
            ratings = ratings_by_season[game.season_start]
            home_state = states[game.home_team]
            away_state = states[game.away_team]
            home_off, home_def = _window_ratings(home_state, game.season_start)
            away_off, away_def = _window_ratings(away_state, game.season_start)
            _apply(
                home_state,
                game,
                is_home=True,
                own_elo=ratings[game.home_team],
                opponent_elo=ratings[game.away_team],
                opponent_offensive_rating=away_off,
                opponent_defensive_rating=away_def,
            )
            _apply(
                away_state,
                game,
                is_home=False,
                own_elo=ratings[game.away_team],
                opponent_elo=ratings[game.home_team],
                opponent_offensive_rating=home_off,
                opponent_defensive_rating=home_def,
            )
            _accumulate_league(league[game.season_start], game)
            events.append(
                EloEvent(
                    season_start=game.season_start,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    home_won=bool(game.home_win),
                )
            )
        index = end
    return rows, states, events, league


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

    rows, _states, _events, _league = _walk(games)
    return _as_feature_frame(rows)


def snapshot_features(history: list[Game], slate: list[Game]) -> pd.DataFrame:
    """Features for slate games. Slate scores are not applied."""

    _rows, states, events, league = _walk(history)
    ordered = sorted(slate, key=lambda game: (game.game_date, game.game_id))
    ratings_by_season = {
        season: ratings_from_events(events, season) for season in {game.season_start for game in ordered}
    }
    return _as_feature_frame(
        [
            _row(
                game,
                states,
                ratings_by_season[game.season_start],
                _league_prior(league.get(game.season_start - 1)),
            )
            for game in ordered
        ]
    )
