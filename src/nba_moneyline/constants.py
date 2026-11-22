"""Locked feature contract. Definitions live in docs/feature-selection.md."""

from __future__ import annotations

# Five fake wins and five fake losses. A short win rate is pulled toward 0.5.
WIN_RATE_PRIOR_WINS = 5
WIN_RATE_PRIOR_GAMES = 10

FEATURE_COLUMNS = (
    "home_win_rate",
    "away_win_rate",
    "home_efg_adj",
    "away_efg_adj",
    "home_efg_allowed_adj",
    "away_efg_allowed_adj",
    "home_tov_rate_adj",
    "away_tov_rate_adj",
    "home_tov_forced_adj",
    "away_tov_forced_adj",
    "home_orb_rate_adj",
    "away_orb_rate_adj",
    "home_orb_allowed_adj",
    "away_orb_allowed_adj",
    "home_home_win_rate",
    "away_away_win_rate",
    "home_season_win_rate",
    "away_season_win_rate",
    "home_recent_win_rate",
    "away_recent_win_rate",
    "home_recent_offensive_rating",
    "away_recent_offensive_rating",
    "home_recent_defensive_rating",
    "away_recent_defensive_rating",
    "home_elo_minus_away_elo",
    "home_back_to_back",
    "away_back_to_back",
    "home_games_prior_7d",
    "away_games_prior_7d",
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
# Current season plus the two before it.
PRIOR_SEASON_WINDOW = 3
RECENT_FORM_WINDOW = 10

ELO_INITIAL = 1500.0
ELO_SCALE = 400.0
ELO_K = 20.0

# Minutes of team-average box score mixed into a player's own rate.
ROTATION_PRIOR_MINUTES = 400.0

LOGISTIC_SOLVER = "lbfgs"
LOGISTIC_MAX_ITER = 1000
LOGISTIC_RANDOM_STATE = 42
