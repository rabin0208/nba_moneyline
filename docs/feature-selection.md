# Feature selection

Pre-game model for whether the home team wins an NBA regular-season game, overtime included. The columns match the hockey model's frozen set except the four overtime and shootout features. The overlapping box-score rates are replaced by the four factors: effective field-goal percentage, turnovers per possession, offensive-rebound rate, and free-throw rate, each opponent-adjusted. Odds are not features.

The executable column list is `FEATURE_COLUMNS` in `src/nba_moneyline/constants.py`.

## Target and rows

`home_win` is 1 when the home team's final score is greater than the away team's. The final score includes overtime.

A row is one regular-season game. Preseason, play-in, and playoffs are not rows, and those games do not update team state. A tied score is dropped. A neutral-site game is dropped, because the home and away roles are the target.

The game date is the ESPN slate date, not the UTC tip time. Rows are ordered by that date, then game id. Every game on a date is scored before any result from that date updates team state. A feature may use only games with an earlier date.

A row is eligible for fitting and evaluation when both teams have at least 5 completed regular-season games inside the three-season window before that date, and both teams' eight adjusted four factors and last-10 opponent-adjusted offensive and defensive ratings are finite. No feature is imputed. An empty win rate is 0.5. The first game of a season uses 0 games and 0.5 for the season win rate.

## Sample

ESPN daily scoreboards for regular-season games. Each completed game supplies the final score. Most games also supply both teams' field goals, attempts, threes, free-throw attempts, rebounds, offensive rebounds, and turnovers, which the four factors use. A game with no shooting line still updates wins, fatigue, and Elo, and it is left out of the effective-field-goal and free-throw factors. A game with no turnover or rebound line is left out of that factor. Play-in and playoff scoreboards are ignored. Neutral-site games are ignored. Team codes are the ESPN abbreviations (`NY`, `GS`, `SA`, `NO`, `UTAH`, `WSH`, `BKN`, `CHA`, `PHX`).

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

29 columns. Twelve of them are effective field-goal percentage, turnover rate, and offensive-rebound rate, home and away, offense and defense. The three-season window is the current season plus the two before it. Games already played this season count in both the window and the season columns.

Win rates use the same sample-size adjustment. Five fake wins and five fake losses are added before dividing, so a short record is pulled toward 0.5 and a long record stays close to the raw rate. No games yet is 0.5.

### Last three seasons

Equal-weight counts inside the window. A season older than that is dropped, not decayed.

- `home_win_rate`, `away_win_rate`: wins over games inside the window, home and road together, with the five-and-five adjustment. The game count is the adjustment, not its own column.
- `home_home_win_rate`: the home team's home-game win rate inside the window, with the same adjustment on the home-game count.
- `away_away_win_rate`: the away team's road-game win rate inside the window, with the same adjustment on the road-game count.

### Four factors

Dean Oliver's four factors, for the offense and for what the team allows. Counting stats for the opponent's pregame rate are summed inside the three-season window, then divided. The feature itself is the equal-weight mean of the per-game adjustments. A game that is missing that factor's line, or whose opponent had no pregame rate, is left out of that mean.

```text
efg = (FG + 0.5 × threes) / FGA
tov_rate = turnovers / possessions
orb_rate = offensive rebounds / (offensive rebounds + opponent defensive rebounds)
ft_rate = free-throw attempts / FGA
```

Defensive rebounds are total rebounds minus offensive rebounds. Possessions are the same estimate as the last-10 ratings.

For each prior game, the offensive adjustment is that game's rate minus the opponent's pregame allowed rate. The defensive adjustment is that game's allowed rate minus the opponent's pregame offensive rate. The opponent rate is the three-season rate entering that game. Positive means the team did more of that thing than the opponent usually allows, or allowed more than the opponent usually does.

- `home_efg_adj`, `away_efg_adj` and `home_efg_allowed_adj`, `away_efg_allowed_adj`
- `home_tov_rate_adj`, `away_tov_rate_adj` and `home_tov_forced_adj`, `away_tov_forced_adj`. A lower turnover-rate adjustment is better. A higher forced-turnover adjustment is better.
- `home_orb_rate_adj`, `away_orb_rate_adj` and `home_orb_allowed_adj`, `away_orb_allowed_adj`. A lower allowed-rebound adjustment is better.
### This season

Reset when the season changes. October through June is one season, labeled by the starting year (2025 is the 2025–26 season).

- `home_season_win_rate`, `away_season_win_rate`. Wins and games reset with the season, then the five-and-five adjustment is applied. The games-played count is that adjustment, not its own column.

### Last 10 games

The window is the last 10 games, home and road together, and it can cross a season. With fewer than 10, the mean uses the games played. A game that is missing free-throw attempts, offensive rebounds, or turnovers, or whose opponent had no pregame rating, is left out of the rating means and does not pull in an older game.

```text
possessions = FGA − offensive rebounds + turnovers + 0.44 × free-throw attempts
offensive rating = 100 × points scored / possessions
defensive rating = 100 × points allowed / opponent possessions
```

- `home_recent_win_rate`, `away_recent_win_rate`. Equal-weight win rate. With no games, the value is 0.5.
- `home_recent_offensive_rating`, `away_recent_offensive_rating`. For each game, that game's offensive rating minus the opponent's pregame defensive rating, then the mean. The opponent rating is the three-season rating entering that game. Positive means the team scored more per 100 possessions than the opponent usually allows.
- `home_recent_defensive_rating`, `away_recent_defensive_rating`. For each game, that game's defensive rating minus the opponent's pregame offensive rating, then the mean. A lower number is better.

### Elo

`home_elo_minus_away_elo`.

One difference. Ratings are rebuilt from games inside the same three-season window, and every team starts at 1500 for that replay. Scale 400. K is 20. There is no home-court bump and no smaller step for overtime. The step is multiplied by `(margin + 3) ^ 0.8 / (7.5 + 0.006 × winner Elo minus loser Elo)`, so a blowout moves the rating more than a one-point win, and a favorite's blowout moves it less than an underdog's. Ratings for a date are read before that date's games are applied.

### Fatigue

The previous game can fall in the prior season. There is no season reset.

- `home_back_to_back`, `away_back_to_back`: 1 when that team's previous regular-season game was exactly one calendar day earlier.
- `home_games_prior_7d`, `away_games_prior_7d`: how many of that team's games have a date from 1 to 7 days before this game.

## Considered and left out

- Games played this season, and games played inside the three-season window. Both counts are inside the win-rate adjustment above.
- The last-10 result residual and free-throw rate. Their weights were near zero once the other columns were in the model.
- Points per game, field-goal percentage, three-point percentage, shot-quality share, turnovers per game, and total rebounds per game. The four factors replace them.
- Assists, steals, and blocks. They are on the raw scoreboard extract, not on the game table the features are built from.
- Overtime and extra-time rate and win counts. Those four columns are the only hockey features left out.
- A decayed win rate, goal differential, and a separate column for each of the last 10 results.
- Moneylines, spreads, and totals.
- A last-10 net rating. It equals the recent offensive adjustment minus the recent defensive adjustment.
- A raw last-10 offensive or defensive rating. The last-10 ratings are the opponent-adjusted versions.
