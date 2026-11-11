"""Closing-line parsing and flat-bet profit."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from nba_moneyline.closing_odds import closing_line_from_summary, line_from_quote
from nba_moneyline.market import flat_bet_profit, parse_american_odds, pick_bets


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
        self.assertEqual(line["spread"], "GS -1.5")

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


if __name__ == "__main__":
    unittest.main()
