# Feature selection

Pre-game model for whether the home team wins an NBA regular-season game, overtime included. The feature set follows the NHL moneyline model: rates and one Elo difference, not a column per past game. Odds are not features. They stay reserved for a later comparison with the market.

The executable column list is `FEATURE_COLUMNS` in `src/nba_moneyline/constants.py`.

## Target and rows

`home_win` is 1 when the home team's final score is greater than the away team's. The final score includes overtime.

A row is one regular-season game. Preseason, play-in, and playoffs are not rows, and those games do not update team state. A tied score is dropped. A neutral-site game is dropped, because the home and away roles are the target.

The game date is the ESPN slate date, not the UTC tip time. Rows are ordered by that date, then game id. Every game on a date is scored before any result from that date updates team state. A feature may use only games with an earlier date.

A row is eligible for fitting and evaluation when both teams have at least 5 completed regular-season games before that date, and both shot-quality shares are finite. No feature is imputed. Empty venue history uses 0.5 for a win rate. The first game of a season uses 0.5 for the season win rate.

## Sample

ESPN daily scoreboards for regular-season games. Each completed game supplies the final score. Most games also supply both teams' field goals, field-goal attempts, and three-pointers. On parts of 2015–16 through 2017–18 the scoreboard has the score and no shooting line. Those games still update wins, points, back-to-backs, and Elo. They do not update shot quality, and no number is filled in for the missing attempts. Play-in and playoff scoreboards are ignored. Neutral-site games are ignored. Team codes are the ESPN abbreviations (`NY`, `GS`, `SA`, `NO`, `UTAH`, `WSH`, `BKN`, `CHA`, `PHX`).

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

21 columns.

### Decayed win rate

`home_win_rate`, `away_win_rate`.

Every prior regular-season game, home and road, across seasons. The most recent prior game has age 0 and weight 1. A game `k` games before that has weight `0.5 ** (k / 82)`. A win from 82 games ago counts half as much as the last game. Older seasons fade and are never cut off.

```text
weight(k) = 0.5 ** (k / 82)
win_rate = sum(weight * win) / sum(weight)
```

The half-life is 82 games, one NBA season. A 41-game half-life was the faster alternative and was not used. The last-10 mean below is not decayed. Points, venue rates, and the season win rate are not decayed.

`home_prior_games` and `away_prior_games` are not features. The 5-game minimum already drops the shortest samples, and a raw count is not a strength signal once the win rates and Elo are present.

### Scoring level

Means over every prior regular-season game, carried across seasons. These two cannot be rebuilt from each other.

- `home_points_for_per_game`, `away_points_for_per_game`
- `home_points_against_per_game`, `away_points_against_per_game`

Point differential per game is points scored minus points allowed. It is omitted. After scaling it is still an exact copy of those two columns.

### Recent results

Equal-weight means. They are not decayed.

- `home_recent_win_rate`, `away_recent_win_rate`: last 10 games, home and road together. With 5 to 9 prior games, the mean uses those games.
- `home_home_win_rate`: home team's win rate in all prior home games.
- `away_away_win_rate`: away team's win rate in all prior road games.
- `home_recent_home_win_rate`: mean of the home team's last 3 home games.
- `away_recent_away_win_rate`: mean of the away team's last 3 road games.

With fewer than 3 games in that building, the venue mean uses the games played there. With none, the value is 0.5.

### This season

Reset when the season changes. October through June is one season, labeled by the starting year (2025 is the 2025–26 season).

- `home_season_games`, `away_season_games`
- `home_season_win_rate`, `away_season_win_rate`

### Elo

`home_elo_minus_away_elo`.

One difference. Both teams start at 1500. Scale 400. K is 20. The update uses the win and the loss only: no home-court bump, no margin of victory, no reset and no pull back toward 1500 between seasons. Ratings for a date are read before that date's games are applied.

### Shot quality

`home_shot_quality_share`, `away_shot_quality_share`.

Attempt-weighted effective field-goal percentage scored, divided by that plus the effective field-goal percentage allowed. Counting stats are summed over every prior game, then turned into the two percentages.

```text
efg = (FG + 0.5 * FG3) / FGA
shot_quality_share = efg_scored / (efg_scored + efg_allowed)
```

This is the stand-in for the NHL model's expected-goal share: shot quality, not the final score. Offensive rating, defensive rating, pace, and the four factors are not added beside it.

### Schedule

- `home_back_to_back`, `away_back_to_back`: 1 when that team's previous regular-season game in the same season was exactly one calendar day earlier. Otherwise 0. The first game of a season is 0 even if the previous season ended the day before.

Games in the prior 7 days were part of the NHL schedule pair and were removed here. Most NBA weeks are three games, so that count barely moves.

## Considered and left out

- A separate column for each of the last 10 game results. The NHL test of that shape was worse on every validation fold. The last 3 home or road games are one mean per team, not three columns.
- Reweighting only the last 10 results. That NHL test did not help. Decay is applied to the full history instead.
- A conference or division flag. Elo and the scoring rates already carry how good the teams are. The conference line does not measure travel.
- Head-to-head results. Pairs meet too few times for a stable streak.
- Calendar week or days since opening night. Games played this season already records how far into the year a team is. An ISO week also breaks when the season crosses the new year.
- Injuries, starting lineups, in-game win probability, and play-by-play. The public injury feed is the current list, not the list from that morning.
- Moneylines, spreads, and totals.
- Continuous rest days. The back-to-back flag is the schedule effect that is large enough to keep.
- Last game's rotation versus the season's usual mix. It describes who played last night, not who is out tonight, and it did not improve the holdout.
