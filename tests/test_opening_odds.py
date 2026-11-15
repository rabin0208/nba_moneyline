"""Action Network opening-line parsing and alignment."""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from nba_moneyline.opening_odds import (
    AN_CLOSE_BOOK,
    align_opening_odds,
    overlay_closes,
    overlay_opens,
    quotes_from_scoreboard,
)


def _game(an_id: int, home: str, away: str, home_odds: int, away_odds: int) -> dict:
    return {
        "id": an_id,
        "type": "reg",
        "start_time": "2024-10-23T23:30:00.000Z",
        "home_team_id": 1,
        "away_team_id": 2,
        "teams": [
            {"id": 1, "abbr": home},
            {"id": 2, "abbr": away},
        ],
        "markets": {
            "30": {
                "event": {
                    "moneyline": [
                        {"side": "home", "odds": home_odds, "period": "event", "is_live": False},
                        {"side": "away", "odds": away_odds, "period": "event", "is_live": False},
                        {"side": "home", "odds": -500, "period": "event", "is_live": True},
                    ]
                }
            }
        },
    }


class OpeningOddsTests(unittest.TestCase):
    def test_maps_action_network_abbreviations_and_skips_live(self) -> None:
        quotes = quotes_from_scoreboard({"games": [_game(10, "GSW", "WAS", -130, 110)]})
        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0]["home"], "GS")
        self.assertEqual(quotes[0]["away"], "WSH")
        self.assertEqual(quotes[0]["open_ml_home"], -130)
        self.assertEqual(quotes[0]["open_ml_away"], 110)

    def test_ignores_preseason_and_a_missing_book(self) -> None:
        preseason = _game(11, "BOS", "NYK", -150, 130)
        preseason["type"] = "pre"
        bare = _game(12, "BOS", "NYK", -150, 130)
        bare["markets"] = {"15": bare["markets"]["30"]}
        self.assertEqual(quotes_from_scoreboard({"games": [preseason, bare]}), [])

    def test_aligns_to_the_espn_game_within_one_day(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "game_id": "espn_1",
                    "game_date": date(2024, 10, 23),
                    "home_key": "GS",
                    "away_key": "WSH",
                },
                {
                    "game_id": "espn_2",
                    "game_date": date(2024, 12, 1),
                    "home_key": "GS",
                    "away_key": "WSH",
                },
            ]
        )
        quotes = pd.DataFrame(quotes_from_scoreboard({"games": [_game(10, "GSW", "WAS", -130, 110)]}))
        aligned = align_opening_odds(games, quotes)
        self.assertEqual(aligned.loc[0, "game_id"], "espn_1")
        self.assertEqual(aligned.loc[0, "home_open"], -130)
        self.assertEqual(aligned.loc[0, "open_book"], "Open")

    def test_consensus_book_is_the_hockey_close(self) -> None:
        self.assertEqual(AN_CLOSE_BOOK, "15")
        game = _game(20, "BOS", "NYK", -150, 130)
        game["markets"] = {"15": game["markets"].pop("30")}
        quotes = quotes_from_scoreboard({"games": [game]}, book="15")
        self.assertEqual(quotes[0]["open_ml_home"], -150)
        self.assertEqual(quotes[0]["open_ml_away"], 130)

    def test_overlay_closes_keeps_only_consensus_games(self) -> None:
        games = pd.DataFrame(
            [
                {"game_id": "espn_1", "home_odds": -148, "provider": "Draft Kings"},
                {"game_id": "espn_2", "home_odds": -110, "provider": "ESPN BET"},
            ]
        )
        closes = pd.DataFrame([{"game_id": "espn_1", "home_odds": -155, "away_odds": 135}])
        merged = overlay_closes(games, closes)
        self.assertEqual(list(merged["game_id"]), ["espn_1"])
        self.assertEqual(merged.iloc[0]["home_odds"], -155)
        self.assertEqual(merged.iloc[0]["provider"], "Action Network Consensus")

    def test_overlay_replaces_the_espn_open(self) -> None:
        games = pd.DataFrame(
            [
                {"game_id": "espn_1", "home_odds": -148, "home_open": -130, "away_open": 110},
                {"game_id": "espn_2", "home_odds": -110, "home_open": -105, "away_open": -105},
            ]
        )
        opens = pd.DataFrame([{"game_id": "espn_1", "home_open": -125, "away_open": 105}])
        merged = overlay_opens(games, opens)
        self.assertEqual(merged.loc[merged["game_id"] == "espn_1", "home_open"].iloc[0], -125)
        self.assertTrue(pd.isna(merged.loc[merged["game_id"] == "espn_2", "home_open"].iloc[0]))
        self.assertEqual(merged.loc[merged["game_id"] == "espn_2", "home_odds"].iloc[0], -110)


if __name__ == "__main__":
    unittest.main()
