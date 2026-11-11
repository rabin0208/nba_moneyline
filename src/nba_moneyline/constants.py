"""Locked feature contract. Definitions live in docs/feature-selection.md."""

from __future__ import annotations

FEATURE_COLUMNS = (
    "home_win_rate",
    "away_win_rate",
    "home_points_for_per_game",
    "away_points_for_per_game",
    "home_points_against_per_game",
    "away_points_against_per_game",
    "home_recent_win_rate",
    "away_recent_win_rate",
    "home_home_win_rate",
    "away_away_win_rate",
    "home_recent_home_win_rate",
    "away_recent_away_win_rate",
    "home_season_games",
    "away_season_games",
    "home_season_win_rate",
    "away_season_win_rate",
    "home_elo_minus_away_elo",
    "home_shot_quality_share",
    "away_shot_quality_share",
    "home_back_to_back",
    "away_back_to_back",
)

# ESPN scoreboard abbreviations.
TEAMS = (
    "ATL",
    "BOS",
    "BKN",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GS",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NO",
    "NY",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SAC",
    "SA",
    "TOR",
    "UTAH",
    "WSH",
)

URL_YEARS = range(2016, 2027)
BURN_IN_SEASON_STARTS = (2015, 2016)
TRAIN_SEASON_STARTS = tuple(range(2017, 2025))
TEST_SEASON_START = 2025

MIN_PRIOR_GAMES = 5
WIN_RATE_HALF_LIFE_GAMES = 82
RECENT_WINDOW = 10
VENUE_RECENT_WINDOW = 3

ELO_INITIAL = 1500.0
ELO_SCALE = 400.0
ELO_K = 20.0

# Minutes of team-average box score mixed into a player's own rate.
ROTATION_PRIOR_MINUTES = 400.0

LOGISTIC_SOLVER = "lbfgs"
LOGISTIC_MAX_ITER = 1000
LOGISTIC_RANDOM_STATE = 42
