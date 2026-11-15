"""Build the feature table and fit the logistic model."""

from __future__ import annotations

import argparse

from nba_moneyline.acquire import GAMES_CSV, download_scoreboards, read_games, write_games
from nba_moneyline.closing_odds import download_closing_odds, download_historical_closes
from nba_moneyline.features import build_feature_frame
from nba_moneyline.opening_odds import build_closing_odds, build_opening_odds
from nba_moneyline.player_box import download_player_summaries, write_player_lines
from nba_moneyline.train import FEATURES_CSV, fit_and_evaluate


def build_features() -> None:
    games = read_games()
    frame = build_feature_frame(games)
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FEATURES_CSV, index=False)
    eligible = int(frame["eligible"].sum())
    print(f"wrote {len(frame)} rows ({eligible} eligible) to {FEATURES_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA home-win model")
    parser.add_argument(
        "command",
        choices=("acquire", "player-boxes", "features", "train", "closing-odds", "opening-odds", "all"),
        help="acquire logs, fetch player box scores, build features, fit the model, fetch closes, or run the data steps",
    )
    parser.add_argument("--season", type=int, default=2025, help="Season start year for closing-odds")
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Fetch 2017-18 through 2024-25 closes from the ESPN core odds feed",
    )
    parser.add_argument("--refresh", action="store_true", help="Refetch Action Network opening lines")
    args = parser.parse_args()
    if args.command == "opening-odds":
        build_opening_odds(refresh=args.refresh)
        return
    if args.command == "closing-odds":
        if args.historical:
            download_historical_closes()
        else:
            build_closing_odds(refresh=args.refresh)
        return
    if args.command == "player-boxes":
        games = read_games() if GAMES_CSV.exists() else write_games()
        download_player_summaries(games)
        write_player_lines(games)
        return
    if args.command in {"acquire", "all"}:
        download_scoreboards()
        write_games()
    if args.command in {"features", "all"}:
        if args.command == "features" and not GAMES_CSV.exists():
            write_games()
        build_features()
    if args.command in {"train", "all"}:
        fit_and_evaluate()


if __name__ == "__main__":
    main()
