"""Definitions from docs/feature-selection.md."""

from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date, timedelta

from nba_moneyline.acquire import turnovers_from_summary
from nba_moneyline.constants import FEATURE_COLUMNS, PRIOR_SEASON_WINDOW, SHRINK_PSEUDO_GAMES
from nba_moneyline.features import (
    FTA_POSSESSION_WEIGHT,
    build_feature_frame,
    expected_score,
    possessions,
    seasons_in_prior_window,
    snapshot_features,
    updated_ratings,
)
from nba_moneyline.player_box import box_score, player_lines_from_summary
from nba_moneyline.schema import Game


def _game(
    days: int,
    home: str,
    away: str,
    home_score: int = 110,
    away_score: int = 100,
    season_start: int = 2024,
    *,
    home_fg: int = 40,
    home_fga: int = 80,
    home_fg3: int = 10,
    away_fg: int = 36,
    away_fga: int = 84,
    away_fg3: int = 12,
    home_fg3a: int = 30,
    away_fg3a: int = 36,
    home_rebounds: int = 44,
    away_rebounds: int = 41,
    home_orb: int = 10,
    away_orb: int = 8,
    home_fta: int = 20,
    away_fta: int = 18,
    home_turnovers: int = 12,
    away_turnovers: int = 15,
) -> Game:
    game_date = date(2024, 10, 22) + timedelta(days=days)
    return Game(
        game_id=f"{game_date.isoformat()}_{away}_at_{home}",
        game_date=game_date,
        season_start=season_start,
        home_team=home,
        away_team=away,
        home_score=home_score,
        away_score=away_score,
        home_fg=home_fg,
        home_fga=home_fga,
        home_fg3=home_fg3,
        away_fg=away_fg,
        away_fga=away_fga,
        away_fg3=away_fg3,
        home_fg3a=home_fg3a,
        away_fg3a=away_fg3a,
        home_rebounds=home_rebounds,
        away_rebounds=away_rebounds,
        home_orb=home_orb,
        away_orb=away_orb,
        home_fta=home_fta,
        away_fta=away_fta,
        home_turnovers=home_turnovers,
        away_turnovers=away_turnovers,
    )


class FeatureTests(unittest.TestCase):
    def test_column_contract(self) -> None:
        self.assertEqual(len(FEATURE_COLUMNS), 29)
        self.assertIn("prev_offensive_rating_diff", FEATURE_COLUMNS)
        self.assertIn("prev_defensive_rating_diff", FEATURE_COLUMNS)
        self.assertIn("season_offensive_rating_diff", FEATURE_COLUMNS)
        self.assertIn("season_efg_pct_diff", FEATURE_COLUMNS)
        self.assertIn("season_turnover_pct_diff", FEATURE_COLUMNS)
        self.assertIn("season_rebound_pct_diff", FEATURE_COLUMNS)
        self.assertNotIn("offensive_rating_diff", FEATURE_COLUMNS)
        self.assertNotIn("win_rate_diff", FEATURE_COLUMNS)
        self.assertNotIn("home_prior_games", FEATURE_COLUMNS)
        self.assertNotIn("points_for_diff", FEATURE_COLUMNS)
        self.assertNotIn("shot_quality_share_diff", FEATURE_COLUMNS)
        self.assertNotIn("net_rating_diff", FEATURE_COLUMNS)
        self.assertEqual(SHRINK_PSEUDO_GAMES, 10)
        self.assertEqual(PRIOR_SEASON_WINDOW, 4)
        self.assertEqual(
            seasons_in_prior_window(2025),
            frozenset({2025, 2024, 2023, 2022}),
        )
        self.assertIn("recent_win_rate_diff", FEATURE_COLUMNS)
        self.assertIn("recent_result_residual_diff", FEATURE_COLUMNS)
        self.assertIn("recent_offensive_rating_diff", FEATURE_COLUMNS)
        self.assertIn("recent_defensive_rating_diff", FEATURE_COLUMNS)
        self.assertNotIn("recent_net_rating_diff", FEATURE_COLUMNS)
        self.assertIn("prev_home_home_win_rate", FEATURE_COLUMNS)
        self.assertIn("season_home_home_win_rate", FEATURE_COLUMNS)
        self.assertNotIn("home_home_win_rate", FEATURE_COLUMNS)
        self.assertIn("home_back_to_back", FEATURE_COLUMNS)
        self.assertNotIn("home_recent_extra_time_rate", FEATURE_COLUMNS)
        self.assertNotIn("home_recent_extra_time_wins", FEATURE_COLUMNS)

    def test_elo_update_has_no_home_or_margin_term(self) -> None:
        home, away = updated_ratings(1500, 1500, True)
        self.assertAlmostEqual(home, 1510)
        self.assertAlmostEqual(away, 1490)
        blowout_home, _blowout_away = updated_ratings(1500, 1500, True)
        self.assertAlmostEqual(home, blowout_home)

    def test_same_day_results_are_invisible(self) -> None:
        first = _game(0, "BOS", "NYK", 120, 100)
        second = _game(0, "LAL", "GSW", 90, 110)
        later = _game(2, "BOS", "LAL", 100, 99)
        frame = build_feature_frame([second, first, later])
        day = frame.loc[frame["game_date"] == first.game_date.isoformat()]
        self.assertTrue((day["home_elo_minus_away_elo"] == 0).all())
        bos = frame.loc[frame["game_id"] == later.game_id].iloc[0]
        self.assertGreater(bos["home_elo_minus_away_elo"], 0)
        self.assertGreater(bos["season_win_rate_diff"], 0)

    def test_back_to_back_resets_at_the_season_boundary(self) -> None:
        opener = Game(
            game_id="2024-10-22_NYK_at_BOS",
            game_date=date(2024, 10, 22),
            season_start=2024,
            home_team="BOS",
            away_team="NYK",
            home_score=110,
            away_score=100,
            home_fg=40,
            home_fga=80,
            home_fg3=10,
            away_fg=36,
            away_fga=84,
            away_fg3=12,
        )
        next_day = Game(
            game_id="2024-10-23_GSW_at_BOS",
            game_date=date(2024, 10, 23),
            season_start=2024,
            home_team="BOS",
            away_team="GSW",
            home_score=101,
            away_score=99,
            home_fg=40,
            home_fga=80,
            home_fg3=10,
            away_fg=36,
            away_fga=84,
            away_fg3=12,
        )
        # Six games in the prior season, on consecutive days ending the day before the opener.
        prior = []
        for index in range(6):
            prior.append(
                Game(
                    game_id=f"prior-{index}",
                    game_date=date(2024, 4, 10) + timedelta(days=index),
                    season_start=2023,
                    home_team="BOS",
                    away_team="NYK",
                    home_score=110,
                    away_score=100,
                    home_fg=40,
                    home_fga=80,
                    home_fg3=10,
                    away_fg=36,
                    away_fga=84,
                    away_fg3=12,
                )
            )
        frame = build_feature_frame(prior + [opener, next_day])
        opening_row = frame.loc[frame["game_id"] == opener.game_id].iloc[0]
        following_row = frame.loc[frame["game_id"] == next_day.game_id].iloc[0]
        self.assertEqual(opening_row["home_back_to_back"], 0)
        self.assertEqual(opening_row["home_season_games"], 0)
        self.assertEqual(opening_row["season_win_rate_diff"], 1.0)
        self.assertEqual(opening_row["prev_win_rate_diff"], 1.0)
        self.assertEqual(following_row["home_back_to_back"], 1)
        self.assertEqual(following_row["home_season_games"], 1)
        self.assertEqual(opening_row["home_prev_games"], 6)
        self.assertEqual(opening_row["prev_home_home_win_rate"], 1.0)
        self.assertEqual(opening_row["season_home_home_win_rate"], 1.0)

    def test_four_season_window_drops_older_results_and_counts_recent_load(self) -> None:
        old = [
            _game(0, "BOS", "NYK", 120, 90, season_start=2019)
            for _ in range(8)
        ]
        for index, game in enumerate(old):
            game_date = date(2019, 11, 1) + timedelta(days=index)
            old[index] = Game(
                game_id=f"old-{index}",
                game_date=game_date,
                season_start=2019,
                home_team="BOS",
                away_team="NYK",
                home_score=120,
                away_score=90,
                home_fg=40,
                home_fga=80,
                home_fg3=10,
                away_fg=36,
                away_fga=84,
                away_fg3=12,
            )
        recent = []
        for index in range(3):
            recent.append(
                Game(
                    game_id=f"recent-{index}",
                    game_date=date(2024, 10, 20) + timedelta(days=index),
                    season_start=2023,
                    home_team="NYK",
                    away_team="BOS",
                    home_score=110,
                    away_score=100,
                    home_fg=40,
                    home_fga=80,
                    home_fg3=10,
                    away_fg=36,
                    away_fga=84,
                    away_fg3=12,
                )
            )
        opener = Game(
            game_id="opener",
            game_date=date(2024, 10, 23),
            season_start=2024,
            home_team="BOS",
            away_team="NYK",
            home_score=100,
            away_score=99,
            home_fg=40,
            home_fga=80,
            home_fg3=10,
            away_fg=36,
            away_fga=84,
            away_fg3=12,
        )
        row = build_feature_frame(old + recent + [opener]).iloc[-1]
        self.assertEqual(row["home_prev_games"], 3)
        self.assertEqual(row["prev_win_rate_diff"], -1.0)
        self.assertEqual(row["season_win_rate_diff"], -1.0)
        self.assertAlmostEqual(row["recent_win_rate_diff"], 0.4)
        self.assertEqual(row["home_games_prior_7d"], 3)
        self.assertEqual(row["away_games_prior_7d"], 3)
        self.assertEqual(row["home_back_to_back"], 1)
        self.assertLess(row["home_elo_minus_away_elo"], 0)

    def test_future_score_does_not_change_an_earlier_row(self) -> None:
        games = [_game(index, "BOS", "NYK", 110, 100) for index in range(8)]
        games.append(_game(8, "BOS", "LAL", 130, 90))
        original = build_feature_frame(games)
        changed = list(games)
        changed[-1] = _game(8, "BOS", "LAL", 90, 130)
        rebuilt = build_feature_frame(changed)
        earlier = original.iloc[:-1][list(FEATURE_COLUMNS)].reset_index(drop=True)
        rebuilt_earlier = rebuilt.iloc[:-1][list(FEATURE_COLUMNS)].reset_index(drop=True)
        self.assertTrue(earlier.equals(rebuilt_earlier))

    def test_snapshot_ignores_the_slate_score(self) -> None:
        history = [_game(0, "BOS", "NYK", 110, 100)]
        home_win = snapshot_features(history, [_game(1, "BOS", "NYK", 140, 80)])
        home_loss = snapshot_features(history, [_game(1, "BOS", "NYK", 80, 140)])
        self.assertTrue(
            home_win.iloc[0][list(FEATURE_COLUMNS)].equals(home_loss.iloc[0][list(FEATURE_COLUMNS)])
        )
        walked = build_feature_frame(history + [_game(1, "BOS", "NYK", 80, 140)])
        self.assertAlmostEqual(
            home_loss.iloc[0]["home_elo_minus_away_elo"],
            walked.iloc[1]["home_elo_minus_away_elo"],
        )

    def test_box_score_subtracts_turnovers_and_misses(self) -> None:
        self.assertAlmostEqual(
            box_score(
                points=20,
                rebounds=5,
                assists=6,
                steals=2,
                blocks=0,
                turnovers=4,
                field_goals_made=5,
                field_goals_attempted=15,
                free_throws_made=6,
                free_throws_attempted=6,
            ),
            19,
        )

    def test_summary_box_score_skips_a_player_who_did_not_play(self) -> None:
        payload = {
            "boxscore": {
                "players": [
                    {
                        "team": {"abbreviation": "BOS"},
                        "statistics": [
                            {
                                "keys": [
                                    "minutes",
                                    "points",
                                    "fieldGoalsMade-fieldGoalsAttempted",
                                    "freeThrowsMade-freeThrowsAttempted",
                                    "rebounds",
                                    "assists",
                                    "turnovers",
                                    "steals",
                                    "blocks",
                                ],
                                "athletes": [
                                    {
                                        "didNotPlay": True,
                                        "athlete": {"id": "1"},
                                        "stats": ["0", "0", "0-0", "0-0", "0", "0", "0", "0", "0"],
                                    },
                                    {
                                        "didNotPlay": False,
                                        "athlete": {"id": "2"},
                                        "stats": ["30", "20", "8-16", "4-4", "5", "4", "2", "1", "0"],
                                    },
                                ],
                            }
                        ],
                    }
                ]
            }
        }
        lines = player_lines_from_summary(payload)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].player_id, "2")
        self.assertEqual(lines[0].team, "BOS")
        self.assertAlmostEqual(lines[0].minutes, 30)

    def test_box_rates_ignore_the_current_game_and_a_missing_line(self) -> None:
        history = [_game(index, "BOS", "NYK") for index in range(3)]
        history.append(replace(_game(3, "BOS", "NYK"), home_turnovers=None, away_turnovers=None))
        later = replace(
            _game(5, "BOS", "NYK"),
            home_score=140,
            home_rebounds=80,
            home_fg=60,
            home_fga=80,
            home_fg3=20,
            home_fg3a=30,
            home_turnovers=30,
        )
        row = build_feature_frame(history + [later]).iloc[-1]
        home_poss = possessions(80, 10, 12, 20)
        away_poss = possessions(84, 8, 15, 18)
        self.assertAlmostEqual(row["season_offensive_rating_diff"], 100 * 110 / home_poss - 100 * 100 / away_poss)
        self.assertAlmostEqual(row["season_defensive_rating_diff"], 100 * 100 / away_poss - 100 * 110 / home_poss)
        self.assertAlmostEqual(row["season_efg_pct_diff"], (40 + 0.5 * 10) / 80 - (36 + 0.5 * 12) / 84)
        home_tov = 12 / (80 + FTA_POSSESSION_WEIGHT * 20 + 12)
        away_tov = 15 / (84 + FTA_POSSESSION_WEIGHT * 18 + 15)
        self.assertAlmostEqual(row["season_turnover_pct_diff"], home_tov - away_tov)
        home_rebound = (4 * (44 / 85) + SHRINK_PSEUDO_GAMES * 0.5) / (4 + SHRINK_PSEUDO_GAMES)
        away_rebound = (4 * (41 / 85) + SHRINK_PSEUDO_GAMES * 0.5) / (4 + SHRINK_PSEUDO_GAMES)
        self.assertAlmostEqual(row["season_rebound_pct_diff"], home_rebound - away_rebound)

    def test_turnovers_come_from_the_summary_team_box(self) -> None:
        home, away = turnovers_from_summary(
            {
                "boxscore": {
                    "teams": [
                        {"homeAway": "home", "statistics": [{"name": "turnovers", "displayValue": "11"}]},
                        {"homeAway": "away", "statistics": [{"name": "turnovers", "displayValue": "9"}]},
                    ]
                }
            }
        )
        self.assertEqual((home, away), (11, 9))

    def test_recent_form_uses_pregame_opponent_strength(self) -> None:
        seed = _game(0, "NYK", "GSW")
        lakers = _game(1, "LAL", "GSW")
        celtics = _game(2, "BOS", "NYK")
        missing = replace(_game(3, "BOS", "NYK"), home_turnovers=None, away_turnovers=None)
        row = build_feature_frame([seed, lakers, celtics, missing, _game(5, "BOS", "LAL")]).iloc[-1]

        home_poss = possessions(80, 10, 12, 20)
        away_poss = possessions(84, 8, 15, 18)
        nyk_ortg = 100 * 110 / home_poss
        nyk_drtg = 100 * 100 / away_poss
        bos_off = 100 * 110 / home_poss - nyk_drtg
        bos_def = 100 * 100 / away_poss - nyk_ortg
        self.assertAlmostEqual(row["recent_offensive_rating_diff"], bos_off)
        self.assertAlmostEqual(row["recent_defensive_rating_diff"], bos_def)

        _nyk_after_seed, gsw_after_seed = updated_ratings(1500, 1500, True)
        lal_residual = 1 - expected_score(1500, gsw_after_seed)
        bos_first = 1 - expected_score(1500, 1510)
        bos_elo, nyk_elo = updated_ratings(1500, 1510, True)
        bos_second = 1 - expected_score(bos_elo, nyk_elo)
        self.assertAlmostEqual(
            row["recent_result_residual_diff"],
            (bos_first + bos_second) / 2 - lal_residual,
        )

    def test_current_rates_shrink_toward_the_previous_season(self) -> None:
        previous = []
        for index in range(10):
            previous.append(
                replace(
                    _game(index, "BOS", "NYK", season_start=2023),
                    game_date=date(2024, 3, 1) + timedelta(days=index),
                    game_id=f"prev-{index}",
                )
            )
        current = []
        for index in range(5):
            current.append(
                replace(
                    _game(index, "BOS", "NYK", 100, 110, season_start=2024),
                    game_date=date(2024, 10, 22) + timedelta(days=index),
                    game_id=f"current-{index}",
                )
            )
        opener = replace(
            _game(0, "BOS", "NYK", season_start=2024),
            game_date=date(2024, 10, 21),
            game_id="opener",
        )
        later = replace(
            _game(0, "BOS", "NYK", season_start=2024),
            game_date=date(2024, 11, 1),
            game_id="later",
        )
        opening_row = build_feature_frame(previous + [opener]).iloc[-1]
        later_row = build_feature_frame(previous + current + [later]).iloc[-1]
        self.assertEqual(opening_row["home_season_games"], 0)
        self.assertEqual(opening_row["season_win_rate_diff"], 1.0)
        self.assertEqual(opening_row["prev_win_rate_diff"], 1.0)

        def shrunk(current_rate: float, games: int, previous_rate: float) -> float:
            return (games * current_rate + SHRINK_PSEUDO_GAMES * previous_rate) / (
                games + SHRINK_PSEUDO_GAMES
            )

        self.assertAlmostEqual(later_row["season_win_rate_diff"], shrunk(0.0, 5, 1.0) - shrunk(1.0, 5, 0.0))
        home_poss = possessions(80, 10, 12, 20)
        away_poss = possessions(84, 8, 15, 18)
        bos_prev_off = 100 * 110 / home_poss
        bos_current_off = 100 * 100 / home_poss
        nyk_prev_off = 100 * 100 / away_poss
        nyk_current_off = 100 * 110 / away_poss
        self.assertAlmostEqual(
            later_row["season_offensive_rating_diff"],
            shrunk(bos_current_off, 5, bos_prev_off) - shrunk(nyk_current_off, 5, nyk_prev_off),
        )
        self.assertAlmostEqual(later_row["season_home_home_win_rate"], shrunk(0.0, 5, 1.0))
        self.assertAlmostEqual(later_row["season_away_away_win_rate"], shrunk(1.0, 5, 0.0))

    def test_first_games_are_ineligible(self) -> None:
        games = [_game(index * 2, "BOS", "NYK") for index in range(4)]
        frame = build_feature_frame(games)
        self.assertTrue((frame["eligible"] == 0).all())


if __name__ == "__main__":
    unittest.main()
