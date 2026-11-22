# Feature selection

Pre-game model for whether the home team wins an NBA regular-season game, overtime included. Strength comparisons enter as home minus away. Scoring, turnovers, and rebounds are per possession. Sample size, venue role, and fatigue stay as separate home and away columns. Odds are not features.

The executable column list is `FEATURE_COLUMNS` in `src/nba_moneyline/constants.py`.

## Target and rows

`home_win` is 1 when the home team's final score is greater than the away team's. The final score includes overtime.

A row is one regular-season game. Preseason, play-in, and playoffs are not rows, and those games do not update team state. A tied score is dropped. A neutral-site game is dropped, because the home and away roles are the target.

The game date is the ESPN slate date, not the UTC tip time. Rows are ordered by that date, then game id. Every game on a date is scored before any result from that date updates team state. A feature may use only games with an earlier date.

A row is eligible for fitting and evaluation when both teams have at least 5 games in the previous regular season, and both previous-season offensive ratings, defensive ratings, effective field-goal percentages, turnover percentages, rebound percentages, and last-10 opponent-adjusted offensive and defensive ratings are finite. No feature is imputed. An empty recent win rate is 0.5. An empty Elo residual is 0. With no games yet this season, a shrunk rate equals the previous season. A team with no previous season uses 0.5 for a win rate and the league average from the completed previous season for a rating.

## Sample

ESPN daily scoreboards for regular-season games, plus the game summary for offensive rebounds and turnovers. Each completed game supplies the final score. A game with no shooting line still updates wins, fatigue, and Elo, and it is left out of effective field-goal percentage. A game missing free-throw attempts, offensive rebounds, or turnovers is left out of offensive rating, defensive rating, and turnover percentage. A game missing either rebound total is left out of rebound percentage. Play-in and playoff scoreboards are ignored. Neutral-site games are ignored. Team codes are the ESPN abbreviations (`NY`, `GS`, `SA`, `NO`, `UTAH`, `WSH`, `BKN`, `CHA`, `PHX`).

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

29 columns. A strength stat is home minus away, so logistic regression gets one coefficient for the gap. Strength rates use the previous regular season and the current season. Elo still uses the four-season window, which is the current season plus the three before it.

Possession-adjusted rates sum the counting stats inside the season, then divide. A game that is missing one of the inputs is left out of that rate. The current-season sample size for a rate is the number of games that entered it.

```text
possessions = FGA − offensive rebounds + turnovers + 0.44 × free-throw attempts
offensive rating = 100 × points scored / possessions
defensive rating = 100 × points allowed / opponent possessions
effective field-goal percentage = (FG + 0.5 × threes) / FGA
turnover percentage = turnovers / (FGA + 0.44 × free-throw attempts + turnovers)
rebound percentage = rebounds / (rebounds + opponent rebounds)
```

### Previous season

The regular season before this one. Seasons older than that are not included.

- `home_prev_games`, `away_prev_games`: games in that season. Kept separate because they are sample sizes.
- `prev_win_rate_diff`: home previous win rate minus away previous win rate. Each rate is wins divided by games, home and road together.
- `prev_offensive_rating_diff`: home previous offensive rating minus away previous offensive rating. Points scored per 100 estimated possessions.
- `prev_defensive_rating_diff`: home previous defensive rating minus away previous defensive rating. Points allowed per 100 opponent possessions. A lower defensive rating is better, so a negative difference favors the home team.
- `prev_efg_pct_diff`: home previous effective field-goal percentage minus away previous effective field-goal percentage.
- `prev_turnover_pct_diff`: home previous turnover percentage minus away previous turnover percentage.
- `prev_rebound_pct_diff`: home previous rebound percentage minus away previous rebound percentage.
- `prev_home_home_win_rate`: the home team's win rate in home games that season.
- `prev_away_away_win_rate`: the away team's win rate in road games that season. A team that played the season but had no games in that venue uses 0.5.

### This season

Reset when the season changes. October through June is one season, labeled by the starting year (2025 is the 2025–26 season). Each rate is shrunk toward the previous season with 10 pseudo-games:

```text
(games × current + 10 × previous) / (games + 10)
```

After 5 games the current season is about a third of the value. After 40 games it is about 80%. With no games yet, the value is the previous season. `games` is the number of games that entered that rate. Venue rates use games in that venue.

- `home_season_games`, `away_season_games`: games played this season. Kept separate because they are sample sizes.
- `season_win_rate_diff`, `season_offensive_rating_diff`, `season_defensive_rating_diff`, `season_efg_pct_diff`, `season_turnover_pct_diff`, `season_rebound_pct_diff`: the shrunk home rate minus the shrunk away rate. The sign of the defensive gap matches the previous-season defensive gap.
- `season_home_home_win_rate`: the home team's shrunk home win rate.
- `season_away_away_win_rate`: the away team's shrunk road win rate.

### Last 10 games

The window is the last 10 games, home and road together, and it can cross a season. With fewer than 10, the mean uses the games played. A game that is missing a box line, or whose opponent had no pregame rating, is left out of the rating means and does not pull in an older game.

- `recent_win_rate_diff`. Each side is the equal-weight win rate. With no games, the value is 0.5. The feature is home minus away.
- `recent_result_residual_diff`. For each game, the win minus that team's pregame Elo win probability, then the mean. A win over a weak opponent shrinks, and a win over a strong opponent stays large. With no games, the value is 0.
- `recent_offensive_rating_diff`. For each game, that game's offensive rating minus the opponent's pregame defensive rating, then the mean. The opponent rating is the four-season rating entering that game. Positive means the team scored more per 100 possessions than the opponent usually allows.
- `recent_defensive_rating_diff`. For each game, that game's defensive rating minus the opponent's pregame offensive rating, then the mean. The opponent rating is the four-season rating entering that game. A lower number is better, so a negative difference favors the home team.

### Elo

`home_elo_minus_away_elo`.

One difference. Ratings are rebuilt from games inside the same four-season window, and every team starts at 1500 for that replay. Scale 400. K is 20. The update uses the win and the loss only: no home-court bump, no margin of victory, and no smaller step for overtime. Ratings for a date are read before that date's games are applied.

### Fatigue

The previous game can fall in the prior season. There is no season reset.

- `home_back_to_back`, `away_back_to_back`: 1 when that team's previous regular-season game was exactly one calendar day earlier.
- `home_games_prior_7d`, `away_games_prior_7d`: how many of that team's games have a date from 1 to 7 days before this game.

## Considered and left out

- Overtime and extra-time rate and win counts. Those four columns are the only hockey features left out.
- An equal-weight blend of the last four seasons. Last season and this season are separate, and this season is shrunk toward last season.
- A raw current-season rate beside the shrunk rate.
- A separate exponentially weighted average. Elo and the last-10 window already carry recent results.
- A decayed win rate, goal differential, and a separate column for each of the last 10 results.
- Moneylines, spreads, and totals.
- Separate home and away copies of a strength stat. The model uses the difference. Both teams being tired is kept, through the separate fatigue columns.
- Net rating, including a last-10 net rating. It equals offensive rating minus defensive rating, so the two rating differences already contain it.
- A raw last-10 offensive or defensive rating. The last-10 ratings are the opponent-adjusted versions.
- Points per game, field-goal percentage, three-point percentage, turnovers per game, and rebounds per game. The possession-adjusted rates replace them.
- Shot-quality share. Offensive and defensive rating cover that comparison.
- An equal-weighted starting five, a raw count of players listed Out, and the minutes of a regular who is newly missing. Long absences are already in the team ratings. The fresh-absence columns made the probabilities sharper and the high-edge closing bets worse, so they were removed.
