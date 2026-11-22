"""Leakage-safe pre-game features.

History is every regular-season game with an earlier date. All games on a
date are read before any of that date's results update team state.

Counts, win rates, and Elo use the current season and the two seasons before
it. Shooting, turnovers, rebounding, and free throws are the four factors,
each minus the opponent's pregame rate. The last-10 window keeps the win
rate plus opponent-adjusted offensive and defensive ratings. Fatigue is a back-to-back flag and games in the prior seven days.
Overtime and shootout features are omitted.
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
    WIN_RATE_PRIOR_GAMES,
    WIN_RATE_PRIOR_WINS,
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


def margin_multiplier(home_rating: float, away_rating: float, home_points: int, away_points: int) -> float:
    """A blowout moves Elo more than a one-point win. A favorite's blowout moves it less."""

    margin = abs(home_points - away_points)
    winner_rating = home_rating if home_points > away_points else away_rating
    loser_rating = away_rating if home_points > away_points else home_rating
    return (margin + 3) ** 0.8 / (7.5 + 0.006 * (winner_rating - loser_rating))


def updated_ratings(
    home_rating: float,
    away_rating: float,
    home_won: bool,
    home_points: int,
    away_points: int,
) -> tuple[float, float]:
    home_expected = expected_score(home_rating, away_rating)
    home_score = 1.0 if home_won else 0.0
    step = ELO_K * margin_multiplier(home_rating, away_rating, home_points, away_points)
    new_home = home_rating + step * (home_score - home_expected)
    new_away = away_rating + step * ((1.0 - home_score) - (1.0 - home_expected))
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
    rated_points_for: float = 0.0
    rated_points_against: float = 0.0
    poss_for: float = 0.0
    poss_against: float = 0.0
    tov_for: float = 0.0
    tov_against: float = 0.0
    orb_for: float = 0.0
    opp_drb: float = 0.0
    orb_against: float = 0.0
    own_drb: float = 0.0
    fta_for: float = 0.0
    fga_ft_for: float = 0.0
    fta_against: float = 0.0
    fga_ft_against: float = 0.0


@dataclass(frozen=True, slots=True)
class RecentGame:
    won: int
    residual: float
    offensive_adj: float | None
    defensive_adj: float | None


@dataclass(frozen=True, slots=True)
class FactorGame:
    """One game's four factors, each minus the opponent's pregame rate."""

    season_start: int
    efg_adj: float | None
    efg_allowed_adj: float | None
    tov_rate_adj: float | None
    tov_forced_adj: float | None
    orb_rate_adj: float | None
    orb_allowed_adj: float | None
    ft_rate_adj: float | None
    ft_allowed_adj: float | None


FACTOR_FIELDS = (
    "efg_adj",
    "efg_allowed_adj",
    "tov_rate_adj",
    "tov_forced_adj",
    "orb_rate_adj",
    "orb_allowed_adj",
    "ft_rate_adj",
    "ft_allowed_adj",
)


@dataclass(frozen=True, slots=True)
class EloEvent:
    season_start: int
    home_team: str
    away_team: str
    home_won: bool
    home_points: int
    away_points: int


@dataclass
class TeamState:
    season_buckets: dict[int, SeasonBucket] = field(default_factory=dict)
    game_dates: list[date] = field(default_factory=list)
    recent_games: list[RecentGame] = field(default_factory=list)
    factor_games: list[FactorGame] = field(default_factory=list)


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
        total.rated_points_for += bucket.rated_points_for
        total.rated_points_against += bucket.rated_points_against
        total.poss_for += bucket.poss_for
        total.poss_against += bucket.poss_against
        total.tov_for += bucket.tov_for
        total.tov_against += bucket.tov_against
        total.orb_for += bucket.orb_for
        total.opp_drb += bucket.opp_drb
        total.orb_against += bucket.orb_against
        total.own_drb += bucket.own_drb
        total.fta_for += bucket.fta_for
        total.fga_ft_for += bucket.fga_ft_for
        total.fta_against += bucket.fta_against
        total.fga_ft_against += bucket.fga_ft_against
    return total


def _rate(wins: int, games: int) -> float:
    """Wins over games, pulled toward 0.5 by five fake wins and five fake losses."""

    return (wins + WIN_RATE_PRIOR_WINS) / (games + WIN_RATE_PRIOR_GAMES)


def _pct(made: float, attempts: float) -> float | None:
    if attempts <= 0:
        return None
    return made / attempts


# Free throws are worth about 0.44 of a possession in the standard estimate.
FTA_POSSESSION_WEIGHT = 0.44


def possessions(fga: float, offensive_rebounds: float, turnovers: float, fta: float) -> float:
    """Estimated possessions: field-goal attempts, offensive rebounds, turnovers, free throws."""

    return fga - offensive_rebounds + turnovers + FTA_POSSESSION_WEIGHT * fta


def _per_hundred(points: float, poss: float) -> float | None:
    if poss <= 0:
        return None
    return 100.0 * points / poss


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


def _box_line(game: Game, is_home: bool) -> tuple[int | None, int | None, int | None, int | None]:
    if is_home:
        return game.home_fga, game.home_orb, game.home_turnovers, game.home_fta
    return game.away_fga, game.away_orb, game.away_turnovers, game.away_fta


def _game_ratings(game: Game, is_home: bool) -> tuple[float | None, float | None]:
    """This game's offensive and defensive rating. Missing when the box is incomplete."""

    points_for = game.home_score if is_home else game.away_score
    points_against = game.away_score if is_home else game.home_score
    own_fga, own_orb, own_tov, own_fta = _box_line(game, is_home)
    opp_fga, opp_orb, opp_tov, opp_fta = _box_line(game, not is_home)
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


def _defensive_rebounds(rebounds: int | None, offensive_rebounds: int | None) -> int | None:
    if rebounds is None or offensive_rebounds is None or rebounds < offensive_rebounds:
        return None
    return rebounds - offensive_rebounds


def _fatigue(game_dates: list[date], game_date: date) -> tuple[float, float]:
    previous = game_dates[-1] if game_dates else None
    back_to_back = float(previous is not None and (game_date - previous).days == 1)
    prior_7d = sum(1 for day in game_dates if 1 <= (game_date - day).days <= 7)
    return back_to_back, float(prior_7d)


def _factor_levels(bucket: SeasonBucket) -> dict[str, float | None]:
    """Three-season four factors. Offense and what the team allows, summed then divided."""

    return {
        "efg": _pct(bucket.fg_value_for, bucket.fga_for),
        "efg_allowed": _pct(bucket.fg_value_against, bucket.fga_against),
        "tov_rate": _pct(bucket.tov_for, bucket.poss_for),
        "tov_forced": _pct(bucket.tov_against, bucket.poss_against),
        "orb_rate": _pct(bucket.orb_for, bucket.orb_for + bucket.opp_drb),
        "orb_allowed": _pct(bucket.orb_against, bucket.orb_against + bucket.own_drb),
        "ft_rate": _pct(bucket.fta_for, bucket.fga_ft_for),
        "ft_allowed": _pct(bucket.fta_against, bucket.fga_ft_against),
    }


def _game_factors(game: Game, is_home: bool) -> dict[str, float | None]:
    """This game's four factors. A missing line leaves that factor empty."""

    if is_home:
        fg, fga, fg3 = game.home_fg, game.home_fga, game.home_fg3
        opp_fg, opp_fga, opp_fg3 = game.away_fg, game.away_fga, game.away_fg3
        orb, rebounds, turnovers, fta = game.home_orb, game.home_rebounds, game.home_turnovers, game.home_fta
        opp_orb, opp_rebounds, opp_turnovers, opp_fta = (
            game.away_orb,
            game.away_rebounds,
            game.away_turnovers,
            game.away_fta,
        )
    else:
        fg, fga, fg3 = game.away_fg, game.away_fga, game.away_fg3
        opp_fg, opp_fga, opp_fg3 = game.home_fg, game.home_fga, game.home_fg3
        orb, rebounds, turnovers, fta = game.away_orb, game.away_rebounds, game.away_turnovers, game.away_fta
        opp_orb, opp_rebounds, opp_turnovers, opp_fta = (
            game.home_orb,
            game.home_rebounds,
            game.home_turnovers,
            game.home_fta,
        )
    efg = efg_allowed = None
    if game.has_shooting:
        efg = field_goal_value(fg, fg3) / fga
        efg_allowed = field_goal_value(opp_fg, opp_fg3) / opp_fga
    tov_rate = tov_forced = None
    if None not in (fga, orb, turnovers, fta, opp_fga, opp_orb, opp_turnovers, opp_fta):
        tov_rate = _pct(float(turnovers), possessions(fga, orb, turnovers, fta))
        tov_forced = _pct(float(opp_turnovers), possessions(opp_fga, opp_orb, opp_turnovers, opp_fta))
    own_drb = _defensive_rebounds(rebounds, orb)
    opp_drb = _defensive_rebounds(opp_rebounds, opp_orb)
    orb_rate = _pct(float(orb), float(orb) + float(opp_drb)) if orb is not None and opp_drb is not None else None
    orb_allowed = (
        _pct(float(opp_orb), float(opp_orb) + float(own_drb)) if opp_orb is not None and own_drb is not None else None
    )
    ft_rate = _pct(float(fta), float(fga)) if fta is not None and fga is not None else None
    ft_allowed = _pct(float(opp_fta), float(opp_fga)) if opp_fta is not None and opp_fga is not None else None
    return {
        "efg": efg,
        "efg_allowed": efg_allowed,
        "tov_rate": tov_rate,
        "tov_forced": tov_forced,
        "orb_rate": orb_rate,
        "orb_allowed": orb_allowed,
        "ft_rate": ft_rate,
        "ft_allowed": ft_allowed,
    }


def _factor_mean(games: list[FactorGame], season_start: int, name: str) -> float | None:
    """Equal-weight mean of one adjusted factor inside the three-season window."""

    allowed = seasons_in_prior_window(season_start)
    values = [
        getattr(game, name)
        for game in games
        if game.season_start in allowed and getattr(game, name) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def ratings_from_events(events: list[EloEvent], season_start: int) -> dict[str, float]:
    """Replay Elo from games inside the three-season window. Every team starts at 1500."""

    allowed = seasons_in_prior_window(season_start)
    ratings: dict[str, float] = defaultdict(lambda: ELO_INITIAL)
    for event in events:
        if event.season_start not in allowed:
            continue
        new_home, new_away = updated_ratings(
            ratings[event.home_team],
            ratings[event.away_team],
            event.home_won,
            event.home_points,
            event.away_points,
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
        "venue_win_rate": _rate(venue_wins, venue_games),
        "season_games": float(current.games),
        "season_win_rate": _rate(current.wins, current.games),
        "recent_win_rate": _recent_win_rate(state.recent_games),
        "recent_residual": _recent_residual(state.recent_games),
        "recent_offensive_rating": _recent_adjusted(state.recent_games, offensive=True),
        "recent_defensive_rating": _recent_adjusted(state.recent_games, offensive=False),
        **{name: _factor_mean(state.factor_games, season_start, name) for name in FACTOR_FIELDS},
    }


def _window_ratings(state: TeamState, season_start: int) -> tuple[float | None, float | None]:
    """Three-season offensive and defensive rating, for the last-10 opponent adjustment."""

    window = _window_bucket(state, season_start)
    return (
        _per_hundred(window.rated_points_for, window.poss_for),
        _per_hundred(window.rated_points_against, window.poss_against),
    )


def _apply(
    state: TeamState,
    game: Game,
    *,
    is_home: bool,
    own_elo: float,
    opponent_elo: float,
    opponent_offensive_rating: float | None,
    opponent_defensive_rating: float | None,
    opponent_factors: dict[str, float | None],
) -> None:
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
    own_fga, own_orb, own_tov, own_fta = _box_line(game, is_home)
    opp_fga, opp_orb, opp_tov, opp_fta = _box_line(game, not is_home)
    if None not in (own_fga, own_orb, own_tov, own_fta, opp_fga, opp_orb, opp_tov, opp_fta):
        bucket.poss_for += possessions(own_fga, own_orb, own_tov, own_fta)
        bucket.poss_against += possessions(opp_fga, opp_orb, opp_tov, opp_fta)
        bucket.tov_for += float(own_tov)
        bucket.tov_against += float(opp_tov)
        bucket.rated_points_for += float(points_for)
        bucket.rated_points_against += float(points_against)
    factors = _game_factors(game, is_home)
    if factors["orb_rate"] is not None:
        opp_rebounds = game.away_rebounds if is_home else game.home_rebounds
        bucket.orb_for += float(own_orb)
        bucket.opp_drb += float(_defensive_rebounds(opp_rebounds, opp_orb))
    if factors["orb_allowed"] is not None:
        own_rebounds = game.home_rebounds if is_home else game.away_rebounds
        bucket.orb_against += float(opp_orb)
        bucket.own_drb += float(_defensive_rebounds(own_rebounds, own_orb))
    if factors["ft_rate"] is not None:
        bucket.fta_for += float(own_fta)
        bucket.fga_ft_for += float(own_fga)
    if factors["ft_allowed"] is not None:
        bucket.fta_against += float(opp_fta)
        bucket.fga_ft_against += float(opp_fga)
    offensive, defensive = _game_ratings(game, is_home)
    state.factor_games.append(
        FactorGame(
            season_start=game.season_start,
            efg_adj=_adjusted(factors["efg"], opponent_factors["efg_allowed"]),
            efg_allowed_adj=_adjusted(factors["efg_allowed"], opponent_factors["efg"]),
            tov_rate_adj=_adjusted(factors["tov_rate"], opponent_factors["tov_forced"]),
            tov_forced_adj=_adjusted(factors["tov_forced"], opponent_factors["tov_rate"]),
            orb_rate_adj=_adjusted(factors["orb_rate"], opponent_factors["orb_allowed"]),
            orb_allowed_adj=_adjusted(factors["orb_allowed"], opponent_factors["orb_rate"]),
            ft_rate_adj=_adjusted(factors["ft_rate"], opponent_factors["ft_allowed"]),
            ft_allowed_adj=_adjusted(factors["ft_allowed"], opponent_factors["ft_rate"]),
        )
    )
    state.recent_games.append(
        RecentGame(
            won=won,
            residual=won - expected_score(own_elo, opponent_elo),
            offensive_adj=_adjusted(offensive, opponent_defensive_rating),
            defensive_adj=_adjusted(defensive, opponent_offensive_rating),
        )
    )
    state.game_dates.append(game.game_date)


def _row(game: Game, states: dict[str, TeamState], ratings: dict[str, float]) -> dict[str, object]:
    home = _side(states[game.home_team], game.season_start, at_home=True)
    away = _side(states[game.away_team], game.season_start, at_home=False)
    home_b2b, home_7d = _fatigue(states[game.home_team].game_dates, game.game_date)
    away_b2b, away_7d = _fatigue(states[game.away_team].game_dates, game.game_date)

    def _ready(side: dict[str, float | None]) -> bool:
        return all(
            side[name] is not None
            for name in (*FACTOR_FIELDS, "recent_offensive_rating", "recent_defensive_rating")
        )

    eligible = int(
        home["prior_games"] >= MIN_PRIOR_GAMES
        and away["prior_games"] >= MIN_PRIOR_GAMES
        and _ready(home)
        and _ready(away)
    )
    row: dict[str, object] = {
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
        "home_home_win_rate": home["venue_win_rate"],
        "away_away_win_rate": away["venue_win_rate"],
        "home_season_win_rate": home["season_win_rate"],
        "away_season_win_rate": away["season_win_rate"],
        "home_recent_win_rate": home["recent_win_rate"],
        "away_recent_win_rate": away["recent_win_rate"],
        "home_recent_offensive_rating": home["recent_offensive_rating"],
        "away_recent_offensive_rating": away["recent_offensive_rating"],
        "home_recent_defensive_rating": home["recent_defensive_rating"],
        "away_recent_defensive_rating": away["recent_defensive_rating"],
        "home_elo_minus_away_elo": ratings[game.home_team] - ratings[game.away_team],
        "home_back_to_back": home_b2b,
        "away_back_to_back": away_b2b,
        "home_games_prior_7d": home_7d,
        "away_games_prior_7d": away_7d,
    }
    for name in FACTOR_FIELDS:
        if f"home_{name}" not in FEATURE_COLUMNS:
            continue
        row[f"home_{name}"] = home[name]
        row[f"away_{name}"] = away[name]
    return row


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
            ratings = ratings_by_season[game.season_start]
            home_state = states[game.home_team]
            away_state = states[game.away_team]
            home_off, home_def = _window_ratings(home_state, game.season_start)
            away_off, away_def = _window_ratings(away_state, game.season_start)
            home_factors = _factor_levels(_window_bucket(home_state, game.season_start))
            away_factors = _factor_levels(_window_bucket(away_state, game.season_start))
            _apply(
                home_state,
                game,
                is_home=True,
                own_elo=ratings[game.home_team],
                opponent_elo=ratings[game.away_team],
                opponent_offensive_rating=away_off,
                opponent_defensive_rating=away_def,
                opponent_factors=away_factors,
            )
            _apply(
                away_state,
                game,
                is_home=False,
                own_elo=ratings[game.away_team],
                opponent_elo=ratings[game.home_team],
                opponent_offensive_rating=home_off,
                opponent_defensive_rating=home_def,
                opponent_factors=home_factors,
            )
            events.append(
                EloEvent(
                    season_start=game.season_start,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    home_won=bool(game.home_win),
                    home_points=game.home_score,
                    away_points=game.away_score,
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
