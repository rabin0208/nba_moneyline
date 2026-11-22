"""DraftKings closing moneylines from the ESPN game summary."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from nba_moneyline.market import is_valid_american_odds, parse_american_odds
from nba_moneyline.train import FEATURES_CSV

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = PROJECT_ROOT / "data" / "raw" / "espn_summary"
CORE_ODDS_DIR = PROJECT_ROOT / "data" / "raw" / "espn_core_odds"
CLOSING_ODDS_CSV = PROJECT_ROOT / "data" / "closing_odds.csv"
HISTORICAL_SEASON_STARTS = tuple(range(2017, 2025))
PAUSE_SECONDS = 0.05
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def summary_url(event_id: str) -> str:
    return f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"


def cache_path(event_id: str) -> Path:
    return SUMMARY_DIR / f"{event_id}.json"


def ensure_summary(event_id: str) -> Path:
    """Cache the ESPN summary for one event. Does not pause."""

    path = cache_path(event_id)
    if path.exists():
        return path
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    body = _fetch(summary_url(event_id))
    json.loads(body)
    path.write_text(body, encoding="utf-8")
    return path


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 503}:
                raise
            time.sleep(20 * (attempt + 1))
        except urllib.error.URLError as error:
            last_error = error
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}") from last_error


def _close_price(side: dict | None) -> float | None:
    if not side:
        return None
    close = side.get("close") or {}
    parsed = parse_american_odds(close.get("odds"))
    if parsed is not None:
        return parsed
    return parse_american_odds(side.get("odds"))


def _team_moneyline(side: dict | None) -> float | None:
    if not isinstance(side, dict):
        return None
    if side.get("moneyLine") is not None:
        parsed = parse_american_odds(side.get("moneyLine"))
        if parsed is not None:
            return parsed
    return _close_price(side)


def line_from_quote(quote: dict) -> dict[str, object] | None:
    """DraftKings moneyline from a scoreboard odds object or a summary pick."""

    moneyline = quote.get("moneyline") or {}
    home_odds = _close_price(moneyline.get("home"))
    away_odds = _close_price(moneyline.get("away"))
    if home_odds is None:
        home_odds = _team_moneyline(quote.get("homeTeamOdds"))
    if away_odds is None:
        away_odds = _team_moneyline(quote.get("awayTeamOdds"))
    if home_odds is None or away_odds is None:
        return None
    provider = (quote.get("provider") or {}).get("name") or ""
    return {
        "home_odds": home_odds,
        "away_odds": away_odds,
        "home_open": _recorded_price(moneyline.get("home"), "open"),
        "away_open": _recorded_price(moneyline.get("away"), "open"),
        "provider": provider,
        "spread": str(quote.get("details") or ""),
    }


def cached_summary_line(event_id: str, *, refresh: bool = False) -> dict[str, object] | None:
    path = cache_path(event_id)
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_fetch(summary_url(event_id)), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    line = closing_line_from_summary(payload)
    if line is None and not refresh:
        return cached_summary_line(event_id, refresh=True)
    return line


def _recorded_price(side: dict | None, kind: str) -> float | None:
    if not isinstance(side, dict):
        return None
    quote = side.get(kind) or {}
    return parse_american_odds(quote.get("odds"))


def closing_line_from_summary(payload: dict) -> dict[str, object] | None:
    """Return DraftKings close and open when the summary has a moneyline."""

    picks = payload.get("pickcenter") or []
    if not picks:
        return None
    quote = picks[0]
    line = line_from_quote(quote)
    if line is None:
        return None
    moneyline = quote.get("moneyline") or {}
    line["home_open"] = _recorded_price(moneyline.get("home"), "open")
    line["away_open"] = _recorded_price(moneyline.get("away"), "open")
    return line


def core_odds_url(event_id: str) -> str:
    return (
        "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
        f"events/{event_id}/competitions/{event_id}/odds?limit=50"
    )


def _book_rank(name: str) -> int | None:
    """Lower is better. Live prices are ignored. DraftKings, then consensus, then ESPN BET."""

    folded = name.casefold()
    if "live" in folded:
        return None
    compact = folded.replace(" ", "")
    if "draftkings" in compact:
        return 0
    if folded == "consensus":
        return 1
    if compact == "espnbet":
        return 2
    if compact == "caesarssportsbook":
        return 3
    return None


def close_from_core_odds(payload: dict) -> dict[str, object] | None:
    """Pick one pregame moneyline from an ESPN core odds list."""

    best: dict[str, object] | None = None
    best_rank = 99
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        name = str((item.get("provider") or {}).get("name") or "")
        rank = _book_rank(name)
        if rank is None or rank >= best_rank:
            continue
        home = (item.get("homeTeamOdds") or {}).get("moneyLine")
        away = (item.get("awayTeamOdds") or {}).get("moneyLine")
        if not is_valid_american_odds(home) or not is_valid_american_odds(away):
            continue
        best_rank = rank
        best = {
            "home_odds": float(home),
            "away_odds": float(away),
            "provider": name,
            "spread": str(item.get("details") or ""),
            "home_open": None,
            "away_open": None,
        }
    return best


def _load_core_odds(event_id: str) -> dict:
    path = CORE_ODDS_DIR / f"{event_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    CORE_ODDS_DIR.mkdir(parents=True, exist_ok=True)
    body = _fetch(core_odds_url(event_id))
    json.loads(body)
    path.write_text(body, encoding="utf-8")
    return json.loads(body)


def download_historical_closes(
    season_starts: tuple[int, ...] = HISTORICAL_SEASON_STARTS,
    pause_seconds: float = PAUSE_SECONDS,
) -> pd.DataFrame:
    """Closing moneylines for seasons the game summary does not carry.

    DraftKings when that book is posted, otherwise the ESPN consensus, otherwise ESPN BET,
    otherwise Caesars.
    """

    features = pd.read_csv(FEATURES_CSV)
    season = features.loc[
        features["season_start"].isin(season_starts) & (features["eligible"] == 1),
        ["game_id", "season_start"],
    ]
    if season.empty:
        raise RuntimeError(f"No eligible games for seasons {season_starts}")
    def one(game: object) -> tuple[bool, dict[str, object] | None]:
        event_id = str(game.game_id).removeprefix("espn_")
        cached = (CORE_ODDS_DIR / f"{event_id}.json").exists()
        payload = _load_core_odds(event_id)
        if not cached and pause_seconds:
            time.sleep(pause_seconds)
        parsed = close_from_core_odds(payload)
        if parsed is None:
            return cached, None
        return cached, {"game_id": game.game_id, "season_start": int(game.season_start), **parsed}

    rows: list[dict[str, object]] = []
    downloaded = 0
    games = list(season.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(one, game) for game in games]
        for index, future in enumerate(as_completed(futures), start=1):
            cached, row = future.result()
            downloaded += int(not cached)
            if row is not None:
                rows.append(row)
            if index % 200 == 0 or index == len(games):
                print(
                    f"historical closes {index}/{len(games)}  lines={len(rows)}  downloaded={downloaded}",
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    if CLOSING_ODDS_CSV.exists():
        existing = pd.read_csv(CLOSING_ODDS_CSV)
        existing = existing.loc[~existing["season_start"].isin(season_starts)]
        merged = pd.concat([existing, frame], ignore_index=True)
    else:
        merged = frame
    merged = merged.drop_duplicates(subset=["game_id"], keep="first")
    CLOSING_ODDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(CLOSING_ODDS_CSV, index=False)
    print(f"wrote {len(merged)} closing lines to {CLOSING_ODDS_CSV}", flush=True)
    return merged


def download_closing_odds(season_start: int, pause_seconds: float = PAUSE_SECONDS) -> pd.DataFrame:
    features = pd.read_csv(FEATURES_CSV)
    season = features.loc[
        (features["season_start"] == season_start) & (features["eligible"] == 1),
        ["game_id"],
    ]
    if season.empty:
        raise RuntimeError(f"No eligible games for season_start={season_start}")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    event_ids = [str(game_id).removeprefix("espn_") for game_id in season["game_id"]]
    downloaded = 0
    for index, event_id in enumerate(event_ids, start=1):
        path = cache_path(event_id)
        if path.exists():
            continue
        body = _fetch(summary_url(event_id))
        json.loads(body)
        path.write_text(body, encoding="utf-8")
        downloaded += 1
        if downloaded % 100 == 0:
            print(f"fetched {downloaded} summaries (through {index}/{len(event_ids)})", flush=True)
        time.sleep(pause_seconds)
    print(f"summaries ready for {season_start} ({downloaded} downloaded)", flush=True)
    return write_closing_odds(season_start)


def write_closing_odds(season_start: int | None = None) -> pd.DataFrame:
    features = pd.read_csv(FEATURES_CSV)
    if season_start is not None:
        features = features.loc[features["season_start"] == season_start]
    rows: list[dict[str, object]] = []
    for game_id, season in zip(features["game_id"], features["season_start"], strict=True):
        event_id = str(game_id).removeprefix("espn_")
        path = cache_path(event_id)
        if not path.exists():
            continue
        parsed = closing_line_from_summary(json.loads(path.read_text(encoding="utf-8")))
        if parsed is None:
            continue
        rows.append({"game_id": game_id, "season_start": int(season), **parsed})
    frame = pd.DataFrame(rows)
    if season_start is None or not CLOSING_ODDS_CSV.exists():
        merged = frame
    else:
        existing = pd.read_csv(CLOSING_ODDS_CSV)
        existing = existing.loc[existing["season_start"] != season_start]
        merged = pd.concat([existing, frame], ignore_index=True)
    merged = merged.drop_duplicates(subset=["game_id"], keep="last")
    CLOSING_ODDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(CLOSING_ODDS_CSV, index=False)
    print(f"wrote {len(merged)} closing lines to {CLOSING_ODDS_CSV}", flush=True)
    return merged
