"""Closing-line parsing and flat-bet profit."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from nba_moneyline.closing_odds import close_from_core_odds, closing_line_from_summary, line_from_quote
from nba_moneyline.market import flat_bet_profit, parse_american_odds, pick_bets
from nba_moneyline.profits import attach_pick_prices, season_result_table


class MarketTests(unittest.TestCase):
    def test_parses_close_from_pickcenter(self) -> None:
        payload = {
            "pickcenter": [
                {
                    "provider": {"name": "Draft Kings"},
                    "moneyline": {
                        "home": {"close": {"odds": "-148"}, "open": {"odds": "-130"}},
                        "away": {"close": {"odds": "+124"}, "open": {"odds": "+110"}},
                    },
                }
            ]
        }
        line = closing_line_from_summary(payload)
        self.assertEqual(line["home_odds"], -148)
        self.assertEqual(line["away_odds"], 124)
        self.assertEqual(line["provider"], "Draft Kings")
        self.assertEqual(line["home_open"], -130)
        self.assertEqual(line["away_open"], 110)

    def test_core_odds_prefer_draftkings_over_consensus_and_skip_live(self) -> None:
        payload = {
            "items": [
                {
                    "provider": {"name": "ESPN Bet - Live Odds"},
                    "homeTeamOdds": {"moneyLine": -5000},
                    "awayTeamOdds": {"moneyLine": 1400},
                },
                {
                    "provider": {"name": "consensus"},
                    "homeTeamOdds": {"moneyLine": -150},
                    "awayTeamOdds": {"moneyLine": 130},
                },
                {
                    "provider": {"name": "DraftKings"},
                    "details": "BOS -3.5",
                    "homeTeamOdds": {"moneyLine": -155},
                    "awayTeamOdds": {"moneyLine": 135},
                },
            ]
        }
        line = close_from_core_odds(payload)
        self.assertEqual(line["provider"], "DraftKings")
        self.assertEqual(line["home_odds"], -155)
        self.assertEqual(line["away_odds"], 135)
        self.assertIsNone(line["home_open"])

    def test_summary_moneyline_without_close_object(self) -> None:
        line = line_from_quote(
            {
                "provider": {"name": "Draft Kings"},
                "details": "GS -1.5",
                "awayTeamOdds": {"moneyLine": 110},
                "homeTeamOdds": {"moneyLine": -130},
            }
        )
        self.assertEqual(line["home_odds"], -130)
        self.assertEqual(line["away_odds"], 110)
        self.assertIsNone(line["home_open"])
        self.assertIsNone(line["away_open"])
        self.assertEqual(line["spread"], "GS -1.5")

    def test_scoreboard_quote_keeps_the_opening_price(self) -> None:
        line = line_from_quote(
            {
                "provider": {"name": "Draft Kings"},
                "moneyline": {
                    "home": {"close": {"odds": "-148"}, "open": {"odds": "-130"}},
                    "away": {"close": {"odds": "+124"}, "open": {"odds": "+110"}},
                },
            }
        )
        self.assertEqual(line["home_odds"], -148)
        self.assertEqual(line["away_odds"], 124)
        self.assertEqual(line["home_open"], -130)
        self.assertEqual(line["away_open"], 110)

    def test_even_money_and_profit(self) -> None:
        self.assertEqual(parse_american_odds("EVEN"), 100)
        self.assertAlmostEqual(flat_bet_profit(-110, True), 100 / 110)
        self.assertEqual(flat_bet_profit(-110, False), -1)

    def test_pick_bets_uses_the_larger_edge(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-01-01",
                    "p_home_win": 0.70,
                    "home_odds": -120,
                    "away_odds": 100,
                    "home_win": 1,
                }
            ]
        )
        bets = pick_bets(frame, edge_threshold=0.02)
        self.assertEqual(bets.loc[0, "bet_side"], "home")
        self.assertGreater(bets.loc[0, "bet_profit"], 0)
        self.assertTrue(math.isfinite(bets.loc[0, "bet_edge"]))

    def test_away_bet_loss_is_not_scored_as_a_home_win(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "game_id": "g2",
                    "game_date": "2026-01-02",
                    "p_home_win": 0.20,
                    "home_odds": 150,
                    "away_odds": -180,
                    "home_win": 1,
                }
            ]
        )
        bets = pick_bets(frame, edge_threshold=0.02)
        self.assertEqual(bets.loc[0, "bet_side"], "away")
        self.assertEqual(bets.loc[0, "bet_profit"], -1.0)

    def test_results_table_bets_the_predicted_winner(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "season_start": 2025,
                    "p_home_win": 0.70,
                    "home_odds": -150,
                    "away_odds": 130,
                    "home_open": -140,
                    "away_open": 120,
                    "home_win": 1,
                },
                {
                    "season_start": 2025,
                    "p_home_win": 0.40,
                    "home_odds": 150,
                    "away_odds": -170,
                    "home_open": 140,
                    "away_open": -160,
                    "home_win": 1,
                },
            ]
        )
        priced = attach_pick_prices(games)
        table = season_result_table(priced, 0.0, "close")
        overall = table.loc[table["season"] == "Overall"].iloc[0]
        self.assertEqual(overall["games"], 2)
        self.assertEqual(overall["bets"], 2)
        self.assertAlmostEqual(overall["accuracy"], 0.5)
        self.assertGreater(priced.loc[0, "close_profit"], 0)
        self.assertEqual(priced.loc[1, "close_profit"], -1.0)
        filtered = season_result_table(priced, 0.5, "close")
        self.assertEqual(filtered.loc[filtered["season"] == "Overall", "bets"].iloc[0], 0)

    def test_results_table_can_keep_underdogs(self) -> None:
        games = pd.DataFrame(
            [
                {
                    "season_start": 2025,
                    "p_home_win": 0.70,
                    "home_odds": -150,
                    "away_odds": 130,
                    "home_open": -140,
                    "away_open": 120,
                    "home_win": 1,
                },
                {
                    "season_start": 2025,
                    "p_home_win": 0.55,
                    "home_odds": 140,
                    "away_odds": -160,
                    "home_open": 130,
                    "away_open": -150,
                    "home_win": 1,
                },
            ]
        )
        priced = attach_pick_prices(games)
        self.assertEqual(priced.loc[0, "close_role"], "favorite")
        self.assertEqual(priced.loc[1, "close_role"], "underdog")
        dogs = season_result_table(priced, 0.0, "close", "underdog")
        favorites = season_result_table(priced, 0.0, "close", "favorite")
        self.assertEqual(dogs.loc[dogs["season"] == "Overall", "bets"].iloc[0], 1)
        self.assertEqual(favorites.loc[favorites["season"] == "Overall", "bets"].iloc[0], 1)
        self.assertGreater(dogs.loc[dogs["season"] == "Overall", "profit"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
