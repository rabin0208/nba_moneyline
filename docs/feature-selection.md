# Feature selection

Pre-game model for whether the home team wins an NBA regular-season game, overtime included. The columns match the hockey model's frozen set except the four overtime and shootout features, plus field-goal percentage, three-point percentage, turnovers per game, and rebounds per game. Shot quality stands in for expected-goal share. Points per game is `points_for_per_game`. Odds are not features.

The executable column list is `FEATURE_COLUMNS` in `src/nba_moneyline/constants.py`.

## Target and rows

`home_win` is 1 when the home team's final score is greater than the away team's. The final score includes overtime.

A row is one regular-season game. Preseason, play-in, and playoffs are not rows, and those games do not update team state. A tied score is dropped. A neutral-site game is dropped, because the home and away roles are the target.

The game date is the ESPN slate date, not the UTC tip time. Rows are ordered by that date, then game id. Every game on a date is scored before any result from that date updates team state. A feature may use only games with an earlier date.

A row is eligible for fitting and evaluation when both teams have at least 5 completed regular-season games inside the three-season window before that date, and both shot-quality shares, field-goal percentages, three-point percentages, turnover rates, rebound rates, and last-10 opponent-adjusted offensive and defensive ratings are finite. No feature is imputed. An empty win rate is 0.5. An empty Elo residual is 0. The first game of a season uses 0 games and 0.5 for the season win rate.

## Sample

ESPN daily scoreboards for regular-season games. Each completed game supplies the final score. Most games also supply both teams' field goals, attempts, and threes, which shot quality uses. A game with no shooting line still updates wins, points, fatigue, and Elo. A game with no turnover or rebound line still updates those, and it is left out of the turnover and rebound rates. Play-in and playoff scoreboards are ignored. Neutral-site games are ignored. Team codes are the ESPN abbreviations (`NY`, `GS`, `SA`, `NO`, `UTAH`, `WSH`, `BKN`, `CHA`, `PHX`).

| Season | Role |
| --- | --- |
| 2015–16 and 2016–17 | State only. Elo and the decayed win rate need history before the first fitted game. |
| 2017–18 through 2024–25 | Training. |
| 2025–26 | Holdout. Not used to choose features or the model. |

The 2019–20 and 2020–21 seasons stay in the file as played. There is no special case for the bubble or the short season.

## Model

`StandardScaler` followed by `LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)`.

The training home-win rate is the comparison baseline. It is a constant probability, not a feature.

## Features

37 columns. The first 23 match the hockey model except the four overtime and shootout features. Eight more are the team box-score rates below, and six are the last-10 opponent-adjusted form. The three-season window is the current season plus the two before it. Games already played this season count in both the window and the season columns.

### Last three seasons

Equal-weight counts inside the window. A season older than that is dropped, not decayed.

- `home_prior_games`, `away_prior_games`
- `home_win_rate`, `away_win_rate`: wins divided by games, home and road together. Empty history is 0.5.
- `home_points_for_per_game`, `away_points_for_per_game`: points per game. This is the PPG feature.
- `home_points_against_per_game`, `away_points_against_per_game`
- `home_field_goal_pct`, `away_field_goal_pct`: field goals made divided by field goals attempted. Makes and attempts are summed inside the window, then divided.
- `home_three_point_pct`, `away_three_point_pct`: threes made divided by threes attempted, summed the same way.
- `home_turnovers_per_game`, `away_turnovers_per_game`: turnovers divided by the games that posted a turnover total. A game with no turnover line is left out of this rate.
- `home_rebounds_per_game`, `away_rebounds_per_game`: total rebounds divided by the games that posted a rebound total.
- `home_home_win_rate`: the home team's win rate in home games inside the window.
- `away_away_win_rate`: the away team's win rate in road games inside the window. Empty venue history is 0.5.
- `home_shot_quality_share`, `away_shot_quality_share`: attempt-weighted effective field-goal percentage scored, divided by that plus the effective field-goal percentage allowed. Counting stats are summed inside the window. This is the stand-in for hockey `prior_xgoal_share`.

```text
efg = (FG + 0.5 * FG3) / FGA
shot_quality_share = efg_scored / (efg_scored + efg_allowed)
```

### This season

Reset when the season changes. October through June is one season, labeled by the starting year (2025 is the 2025–26 season).

- `home_season_games`, `away_season_games`
- `home_season_win_rate`, `away_season_win_rate`

### Last 10 games

The window is the last 10 games, home and road together, and it can cross a season. With fewer than 10, the mean uses the games played. A game that is missing free-throw attempts, offensive rebounds, or turnovers, or whose opponent had no pregame rating, is left out of the rating means and does not pull in an older game.

```text
possessions = FGA − offensive rebounds + turnovers + 0.44 × free-throw attempts
offensive rating = 100 × points scored / possessions
defensive rating = 100 × points allowed / opponent possessions
```

- `home_recent_win_rate`, `away_recent_win_rate`. Equal-weight win rate. With no games, the value is 0.5.
- `home_recent_result_residual`, `away_recent_result_residual`. For each game, the win minus that team's pregame Elo win probability, then the mean. A win over a weak opponent shrinks, and a win over a strong opponent stays large. With no games, the value is 0.
- `home_recent_offensive_rating`, `away_recent_offensive_rating`. For each game, that game's offensive rating minus the opponent's pregame defensive rating, then the mean. The opponent rating is the three-season rating entering that game. Positive means the team scored more per 100 possessions than the opponent usually allows.
- `home_recent_defensive_rating`, `away_recent_defensive_rating`. For each game, that game's defensive rating minus the opponent's pregame offensive rating, then the mean. A lower number is better.

### Elo

`home_elo_minus_away_elo`.

One difference. Ratings are rebuilt from games inside the same three-season window, and every team starts at 1500 for that replay. Scale 400. K is 20. The update uses the win and the loss only: no home-court bump, no margin of victory, and no smaller step for overtime. Ratings for a date are read before that date's games are applied.

### Fatigue

The previous game can fall in the prior season. There is no season reset.

- `home_back_to_back`, `away_back_to_back`: 1 when that team's previous regular-season game was exactly one calendar day earlier.
- `home_games_prior_7d`, `away_games_prior_7d`: how many of that team's games have a date from 1 to 7 days before this game.

## Considered and left out

- Overtime and extra-time rate and win counts. Those four columns are the only hockey features left out.
- A decayed win rate, goal differential, and a separate column for each of the last 10 results.
- Moneylines, spreads, and totals.
- A last-10 net rating. It equals the recent offensive adjustment minus the recent defensive adjustment.
- A raw last-10 offensive or defensive rating. The last-10 ratings are the opponent-adjusted versions.
