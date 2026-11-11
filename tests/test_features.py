"""Definitions from docs/feature-selection.md."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

from nba_moneyline.constants import FEATURE_COLUMNS, WIN_RATE_HALF_LIFE_GAMES
from nba_moneyline.features import (
    build_feature_frame,
    decayed_win_rate,
    game_weight,
    shot_quality_share,
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
    )


class FeatureTests(unittest.TestCase):
    def test_column_contract(self) -> None:
        self.assertEqual(len(FEATURE_COLUMNS), 21)
        self.assertNotIn("home_rotation_delta", FEATURE_COLUMNS)
        self.assertNotIn("away_rotation_delta", FEATURE_COLUMNS)
        self.assertNotIn("home_prior_games", FEATURE_COLUMNS)
        self.assertNotIn("away_prior_games", FEATURE_COLUMNS)
        self.assertNotIn("home_points_diff_per_game", FEATURE_COLUMNS)
        self.assertNotIn("home_games_prior_7d", FEATURE_COLUMNS)
        self.assertNotIn("away_games_prior_7d", FEATURE_COLUMNS)

    def test_half_life_is_one_season(self) -> None:
        self.assertEqual(WIN_RATE_HALF_LIFE_GAMES, 82)
        self.assertAlmostEqual(game_weight(0), 1.0)
        self.assertAlmostEqual(game_weight(82), 0.5)

    def test_decay_weights_the_latest_game_more_than_a_flat_mean(self) -> None:
        results = [1, 0]
        self.assertLess(decayed_win_rate(results), 0.5)

    def test_shot_quality_uses_attempt_weighted_percentages(self) -> None:
        share = shot_quality_share(fg_value_for=12, fga_for=20, fg_value_against=9, fga_against=20)
        self.assertAlmostEqual(share, 0.6 / (0.6 + 0.45))

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
        self.assertLess(bos["away_win_rate"], 0.5)

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
        self.assertEqual(opening_row["home_season_win_rate"], 0.5)
        self.assertGreater(opening_row["home_win_rate"], 0.5)
        self.assertEqual(following_row["home_back_to_back"], 1)
        self.assertEqual(following_row["home_season_games"], 1)

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

    def test_first_games_are_ineligible(self) -> None:
        games = [_game(index * 2, "BOS", "NYK") for index in range(4)]
        frame = build_feature_frame(games)
        self.assertTrue((frame["eligible"] == 0).all())


if __name__ == "__main__":
    unittest.main()
