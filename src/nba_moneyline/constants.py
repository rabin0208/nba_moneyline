"""Locked feature contract. Definitions live in docs/feature-selection.md."""

from __future__ import annotations

FEATURE_COLUMNS = (
    "home_prev_games",
    "away_prev_games",
    "prev_win_rate_diff",
    "prev_offensive_rating_diff",
    "prev_defensive_rating_diff",
    "prev_efg_pct_diff",
    "prev_turnover_pct_diff",
    "prev_rebound_pct_diff",
    "prev_home_home_win_rate",
    "prev_away_away_win_rate",
    "home_season_games",
    "away_season_games",
    "season_win_rate_diff",
    "season_offensive_rating_diff",
    "season_defensive_rating_diff",
    "season_efg_pct_diff",
    "season_turnover_pct_diff",
    "season_rebound_pct_diff",
    "season_home_home_win_rate",
    "season_away_away_win_rate",
    "recent_win_rate_diff",
    "recent_result_residual_diff",
    "recent_offensive_rating_diff",
    "recent_defensive_rating_diff",
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
# Current season plus the three before it. Elo still replays this window.
PRIOR_SEASON_WINDOW = 4
RECENT_FORM_WINDOW = 10
# Current-season rates lean on the previous season by this many pseudo-games.
SHRINK_PSEUDO_GAMES = 10

ELO_INITIAL = 1500.0
ELO_SCALE = 400.0
ELO_K = 20.0

# Minutes of team-average box score mixed into a player's own rate.
ROTATION_PRIOR_MINUTES = 400.0

LOGISTIC_SOLVER = "lbfgs"
LOGISTIC_MAX_ITER = 1000
LOGISTIC_RANDOM_STATE = 42
