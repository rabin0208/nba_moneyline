"""Action Network book 30 ("Open") moneylines, the same book the hockey model uses.

Hockey reads that book from the v1 scoreboard. That route no longer returns
games, so this uses the v2 scoreboard, which still serves book 30. For the NBA
the Open line starts with the 2024–25 season; earlier slates have no book 30.

Usage:
  python -m nba_moneyline opening-odds
  python -m nba_moneyline opening-odds --refresh
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from nba_moneyline.acquire import GAMES_CSV
from nba_moneyline.market import is_valid_american_odds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUOTES_CSV = PROJECT_ROOT / "data" / "an_opening_quotes.csv"
FETCHED_DATES_CSV = PROJECT_ROOT / "data" / "an_opening_fetched_dates.csv"
OPENING_ODDS_CSV = PROJECT_ROOT / "data" / "opening_odds.csv"
CLOSING_QUOTES_CSV = PROJECT_ROOT / "data" / "an_closing_quotes.csv"
CLOSING_FETCHED_DATES_CSV = PROJECT_ROOT / "data" / "an_closing_fetched_dates.csv"
AN_CLOSING_ODDS_CSV = PROJECT_ROOT / "data" / "an_closing_odds.csv"
SCOREBOARD_URL = "https://api.actionnetwork.com/web/v2/scoreboard/nba"
AN_OPEN_BOOK = "30"
AN_CLOSE_BOOK = "15"
MIN_DATE = date(2024, 10, 1)
CLOSE_MIN_DATE = date(2023, 10, 1)
PAUSE_SECONDS = 0.05
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# Action Network abbreviations that differ from the ESPN scoreboard.
ABBR_TO_ESPN = {
    "GSW": "GS",
    "NOP": "NO",
    "NYK": "NY",
    "SAS": "SA",
    "UTA": "UTAH",
    "WAS": "WSH",
}


def espn_abbr(abbr: object) -> str:
    text = str(abbr).upper()
    return ABBR_TO_ESPN.get(text, text)


def load_games(min_date: date = MIN_DATE) -> pd.DataFrame:
    games = pd.read_csv(GAMES_CSV, usecols=["game_id", "game_date", "home_team", "away_team"])
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    games = games.loc[games["game_date"] >= min_date].copy()
    games["home_key"] = games["home_team"].map(espn_abbr)
    games["away_key"] = games["away_team"].map(espn_abbr)
    return games


def candidate_dates(games: pd.DataFrame, min_date: date = MIN_DATE) -> list[str]:
    dates: set[date] = set()
    for game_date in games["game_date"]:
        dates.add(game_date)
        dates.add(game_date + timedelta(days=1))
        dates.add(game_date - timedelta(days=1))
    return [day.strftime("%Y%m%d") for day in sorted(dates) if day >= min_date]


def _fetch_scoreboard(url_date: str, book: str = AN_OPEN_BOOK) -> dict:
    query = urllib.parse.urlencode({"date": url_date, "bookIds": book})
    request = urllib.request.Request(
        f"{SCOREBOARD_URL}?{query}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 503}:
                raise
            time.sleep(2 ** attempt)
        except urllib.error.URLError as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{url_date}: {last_error}") from last_error


def _book_moneyline(game: dict, book_id: str) -> tuple[int, int] | None:
    markets = game.get("markets") or {}
    book = markets.get(book_id) or markets.get(int(book_id))
    if not isinstance(book, dict):
        return None
    home = away = None
    for row in (book.get("event") or {}).get("moneyline") or []:
        if row.get("period") not in (None, "event") or row.get("is_live"):
            continue
        price = row.get("odds")
        if not is_valid_american_odds(price):
            continue
        if row.get("side") == "home":
            home = int(price)
        elif row.get("side") == "away":
            away = int(price)
    if home is None or away is None:
        return None
    return home, away


def quotes_from_scoreboard(payload: dict, book: str = AN_OPEN_BOOK) -> list[dict]:
    quotes: list[dict] = []
    for game in payload.get("games") or []:
        if game.get("type") not in (None, "reg"):
            continue
        teams = {team.get("id"): team for team in (game.get("teams") or [])}
        home = teams.get(game.get("home_team_id"))
        away = teams.get(game.get("away_team_id"))
        if not home or not away:
            continue
        prices = _book_moneyline(game, book)
        if prices is None:
            continue
        quotes.append(
            {
                "an_id": game.get("id"),
                "kickoff_utc": game.get("start_time"),
                "home": espn_abbr(home.get("abbr")),
                "away": espn_abbr(away.get("abbr")),
                "open_ml_home": prices[0],
                "open_ml_away": prices[1],
            }
        )
    return quotes


def fetch_opening_quotes(
    dates: list[str], *, book: str = AN_OPEN_BOOK, pause: float = PAUSE_SECONDS
) -> tuple[list[dict], list[str]]:
    quotes: list[dict] = []
    seen: set[object] = set()
    fetched: list[str] = []
    for index, day in enumerate(dates, start=1):
        try:
            payload = _fetch_scoreboard(day, book)
        except Exception as exc:
            print(f"  skip {day}: {exc}", flush=True)
            continue
        fetched.append(day)
        for quote in quotes_from_scoreboard(payload, book):
            if quote["an_id"] in seen:
                continue
            seen.add(quote["an_id"])
            quotes.append(quote)
        if index % 25 == 0 or index == len(dates):
            print(f"  {index}/{len(dates)} dates  quotes={len(quotes)}", flush=True)
        if pause:
            time.sleep(pause)
    return quotes, fetched


def align_opening_odds(games: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    by_teams: dict[tuple[str, str], list[tuple[date, str]]] = {}
    for row in games.itertuples(index=False):
        by_teams.setdefault((row.home_key, row.away_key), []).append((row.game_date, str(row.game_id)))

    rows = []
    unmatched = 0
    for quote in quotes.itertuples(index=False):
        kick = pd.to_datetime(quote.kickoff_utc, utc=True, errors="coerce")
        if pd.isna(kick):
            unmatched += 1
            continue
        et_date = kick.tz_convert("America/New_York").date()
        candidates = by_teams.get((espn_abbr(quote.home), espn_abbr(quote.away)), [])
        hits = [(abs((game_date - et_date).days), game_id) for game_date, game_id in candidates]
        hits = [hit for hit in hits if hit[0] <= 1]
        if not hits:
            unmatched += 1
            continue
        hits.sort()
        rows.append(
            {
                "game_id": hits[0][1],
                "home_open": int(quote.open_ml_home),
                "away_open": int(quote.open_ml_away),
                "open_source": "actionnetwork",
                "open_book": "Open",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates("game_id", keep="last")
    print(f"  matched {len(out)}  unmatched quotes {unmatched}", flush=True)
    return out


def overlay_closes(games: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Keep games that have an Action Network consensus close and use that price."""

    base = games.drop(columns=[column for column in ("home_odds", "away_odds") if column in games.columns])
    keep = closes.loc[:, ["game_id", "home_odds", "away_odds"]].drop_duplicates("game_id", keep="last")
    merged = base.merge(keep, on="game_id", how="inner")
    merged["provider"] = "Action Network Consensus"
    return merged


def overlay_opens(games: pd.DataFrame, opens: pd.DataFrame) -> pd.DataFrame:
    """Replace any stored open with the Action Network line. Games without one stay blank."""

    base = games.drop(columns=[column for column in ("home_open", "away_open") if column in games.columns])
    keep = opens.loc[:, ["game_id", "home_open", "away_open"]].drop_duplicates("game_id", keep="last")
    return base.merge(keep, on="game_id", how="left")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_opening_odds(*, refresh: bool = False) -> pd.DataFrame:
    games = load_games()
    cached_quotes = pd.DataFrame() if refresh else _load_csv(QUOTES_CSV)
    fetched = set() if refresh or not FETCHED_DATES_CSV.is_file() else set(
        pd.read_csv(FETCHED_DATES_CSV)["date"].astype(str)
    )
    needed = [day for day in candidate_dates(games) if day not in fetched]
    fresh_quotes = pd.DataFrame()
    if needed:
        print(f"Fetching Action Network Open moneylines for {len(needed)} dates...", flush=True)
        quotes, fetched_now = fetch_opening_quotes(needed)
        fresh_quotes = pd.DataFrame(quotes)
        fetched.update(fetched_now)
    else:
        print("Opening quote cache is complete. Realigning to games.", flush=True)

    if cached_quotes.empty:
        quotes = fresh_quotes
    elif fresh_quotes.empty:
        quotes = cached_quotes
    else:
        quotes = pd.concat([cached_quotes, fresh_quotes], ignore_index=True)
    if not quotes.empty:
        quotes = quotes.drop_duplicates("an_id", keep="last")

    OPENING_ODDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not quotes.empty:
        quotes.to_csv(QUOTES_CSV, index=False)
    pd.DataFrame({"date": sorted(fetched)}).to_csv(FETCHED_DATES_CSV, index=False)

    aligned = align_opening_odds(games, quotes) if not quotes.empty else pd.DataFrame()
    if not aligned.empty:
        aligned = aligned.sort_values("game_id")
        aligned.to_csv(OPENING_ODDS_CSV, index=False)
    print(f"Wrote {OPENING_ODDS_CSV} ({len(aligned)} games)", flush=True)
    covered = games["game_id"].astype(str).isin(set(aligned["game_id"].astype(str)) if not aligned.empty else set())
    print(f"  Coverage since {MIN_DATE}: {int(covered.sum())}/{len(games)} games", flush=True)
    return aligned


def build_closing_odds(*, refresh: bool = False) -> pd.DataFrame:
    """Action Network book 15, the consensus close the hockey model uses.

    On the NBA scoreboard that book starts with the 2023–24 season.
    """

    games = load_games(CLOSE_MIN_DATE)
    cached_quotes = pd.DataFrame() if refresh else _load_csv(CLOSING_QUOTES_CSV)
    fetched = set() if refresh or not CLOSING_FETCHED_DATES_CSV.is_file() else set(
        pd.read_csv(CLOSING_FETCHED_DATES_CSV)["date"].astype(str)
    )
    needed = [day for day in candidate_dates(games, CLOSE_MIN_DATE) if day not in fetched]
    fresh_quotes = pd.DataFrame()
    if needed:
        print(f"Fetching Action Network consensus closes for {len(needed)} dates...", flush=True)
        quotes, fetched_now = fetch_opening_quotes(needed, book=AN_CLOSE_BOOK)
        fresh_quotes = pd.DataFrame(quotes)
        fetched.update(fetched_now)
    else:
        print("Closing quote cache is complete. Realigning to games.", flush=True)

    if cached_quotes.empty:
        quotes = fresh_quotes
    elif fresh_quotes.empty:
        quotes = cached_quotes
    else:
        quotes = pd.concat([cached_quotes, fresh_quotes], ignore_index=True)
    if not quotes.empty:
        quotes = quotes.drop_duplicates("an_id", keep="last")

    AN_CLOSING_ODDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not quotes.empty:
        quotes.to_csv(CLOSING_QUOTES_CSV, index=False)
    pd.DataFrame({"date": sorted(fetched)}).to_csv(CLOSING_FETCHED_DATES_CSV, index=False)

    aligned = align_opening_odds(games, quotes) if not quotes.empty else pd.DataFrame()
    if not aligned.empty:
        aligned = aligned.rename(columns={"home_open": "home_odds", "away_open": "away_odds"})
        aligned["close_source"] = "actionnetwork"
        aligned["close_book"] = "Consensus"
        aligned = aligned.drop(columns=["open_source", "open_book"], errors="ignore")
        aligned = aligned.sort_values("game_id")
        aligned.to_csv(AN_CLOSING_ODDS_CSV, index=False)
    print(f"Wrote {AN_CLOSING_ODDS_CSV} ({len(aligned)} games)", flush=True)
    covered = games["game_id"].astype(str).isin(set(aligned["game_id"].astype(str)) if not aligned.empty else set())
    print(f"  Coverage since {CLOSE_MIN_DATE}: {int(covered.sum())}/{len(games)} games", flush=True)
    return aligned
