"""Download ESPN regular-season scoreboards and build one row per game.

Basketball-Reference has the same shooting line, and it answered 429 after a
short burst. The daily ESPN scoreboard returns final scores plus made field
goals, attempts, and threes for both teams, which is everything the feature
table uses.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from nba_moneyline.constants import TEAMS
from nba_moneyline.schema import Game

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "espn"
SUMMARY_DIR = PROJECT_ROOT / "data" / "raw" / "espn_summary"
GAMES_CSV = PROJECT_ROOT / "data" / "games.csv"
SKIPPED_CSV = PROJECT_ROOT / "data" / "skipped_games.csv"

FIRST_DATE = date(2015, 10, 1)
LAST_DATE = date(2026, 6, 30)
REQUEST_PAUSE_SECONDS = 0.05
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
TEAM_SET = set(TEAMS)


def scoreboard_url(game_date: date) -> str:
    return (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        f"?dates={game_date:%Y%m%d}&limit=50"
    )


def cache_path(game_date: date) -> Path:
    return RAW_DIR / f"{game_date:%Y%m%d}.json"


def iter_dates() -> list[date]:
    days = (LAST_DATE - FIRST_DATE).days + 1
    return [FIRST_DATE + timedelta(days=offset) for offset in range(days)]


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 503}:
                raise
            print(f"  {error.code} on {url}; waiting {30 * (attempt + 1)}s", flush=True)
            time.sleep(30 * (attempt + 1))
        except urllib.error.URLError as error:
            last_error = error
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}") from last_error


def ensure_scoreboards(
    end: date,
    start: date = FIRST_DATE,
    pause_seconds: float = REQUEST_PAUSE_SECONDS,
) -> None:
    """Cache any missing daily scoreboards from start through end, inclusive."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetched = 0
    day = start
    while day <= end:
        path = cache_path(day)
        if not path.exists():
            body = _fetch(scoreboard_url(day))
            json.loads(body)
            path.write_text(body, encoding="utf-8")
            fetched += 1
            if fetched % 25 == 0:
                print(f"fetched {fetched} new days (through {day.isoformat()})", flush=True)
            time.sleep(pause_seconds)
        day += timedelta(days=1)
    if fetched:
        print(f"scoreboard cache extended through {end.isoformat()} ({fetched} downloaded)", flush=True)


def download_scoreboards(pause_seconds: float = REQUEST_PAUSE_SECONDS) -> None:
    ensure_scoreboards(LAST_DATE, pause_seconds=pause_seconds)
    print("scoreboard cache ready", flush=True)


def _stat(competitor: dict, name: str) -> int:
    for stat in competitor.get("statistics") or []:
        if stat.get("name") == name:
            return int(float(stat["displayValue"]))
    raise KeyError(name)


def _optional_stat(competitor: dict, name: str) -> int | None:
    try:
        return _stat(competitor, name)
    except (KeyError, TypeError, ValueError):
        return None


def team_stat_from_summary(payload: dict, name: str) -> tuple[int | None, int | None]:
    """Home and away values for one team box-score stat."""

    found: dict[str, int] = {}
    for team_block in (payload.get("boxscore") or {}).get("teams") or []:
        side = team_block.get("homeAway")
        if side not in {"home", "away"}:
            continue
        for stat in team_block.get("statistics") or []:
            if stat.get("name") != name:
                continue
            try:
                found[side] = int(float(stat.get("displayValue")))
            except (TypeError, ValueError):
                continue
    return found.get("home"), found.get("away")


def turnovers_from_summary(payload: dict) -> tuple[int | None, int | None]:
    """Home and away turnover totals from an ESPN game summary."""

    return team_stat_from_summary(payload, "turnovers")


def _summary_box(event_id: str) -> tuple[tuple[int | None, int | None], tuple[int | None, int | None]]:
    path = SUMMARY_DIR / f"{event_id}.json"
    if not path.is_file():
        return (None, None), (None, None)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return turnovers_from_summary(payload), team_stat_from_summary(payload, "offensiveRebounds")


def games_from_scoreboard(payload: dict, slate_date: date) -> tuple[list[Game], list[dict[str, str]]]:
    games: list[Game] = []
    skipped: list[dict[str, str]] = []
    for event in payload.get("events") or []:
        event_id = str(event.get("id", ""))
        season = event.get("season") or {}
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        status = ((competition.get("status") or {}).get("type") or {}).get("name")
        if season.get("type") != 2 or status != "STATUS_FINAL":
            continue
        if competition.get("neutralSite"):
            skipped.append({"game_id": event_id, "reason": "neutral site"})
            continue
        try:
            season_start = int(season["year"]) - 1
            competitors = competition.get("competitors") or []
            by_side = {row.get("homeAway"): row for row in competitors}
            home = by_side["home"]
            away = by_side["away"]
            home_team = home["team"]["abbreviation"]
            away_team = away["team"]["abbreviation"]
            if home_team not in TEAM_SET or away_team not in TEAM_SET:
                raise KeyError("team outside the 30-team set")
            home_score = int(home["score"])
            away_score = int(away["score"])
            try:
                shooting = {
                    "home_fg": _stat(home, "fieldGoalsMade"),
                    "home_fga": _stat(home, "fieldGoalsAttempted"),
                    "home_fg3": _stat(home, "threePointFieldGoalsMade"),
                    "away_fg": _stat(away, "fieldGoalsMade"),
                    "away_fga": _stat(away, "fieldGoalsAttempted"),
                    "away_fg3": _stat(away, "threePointFieldGoalsMade"),
                }
            except KeyError:
                shooting = {}
            (home_turnovers, away_turnovers), (home_orb, away_orb) = _summary_box(event_id)
            record = Game(
                game_id=f"espn_{event_id}",
                game_date=slate_date,
                season_start=season_start,
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                home_fg3a=_optional_stat(home, "threePointFieldGoalsAttempted"),
                away_fg3a=_optional_stat(away, "threePointFieldGoalsAttempted"),
                home_rebounds=_optional_stat(home, "rebounds"),
                away_rebounds=_optional_stat(away, "rebounds"),
                home_orb=home_orb,
                away_orb=away_orb,
                home_fta=_optional_stat(home, "freeThrowsAttempted"),
                away_fta=_optional_stat(away, "freeThrowsAttempted"),
                home_turnovers=home_turnovers,
                away_turnovers=away_turnovers,
                **shooting,
            )
        except (KeyError, TypeError, ValueError) as error:
            skipped.append({"game_id": event_id, "reason": str(error)})
            continue
        if record.home_score == record.away_score:
            skipped.append({"game_id": event_id, "reason": "tied score"})
            continue
        if record.has_shooting and (record.home_fga <= 0 or record.away_fga <= 0):
            skipped.append({"game_id": event_id, "reason": "missing attempts"})
            continue
        games.append(record)
    return games, skipped


def write_games() -> list[Game]:
    games: list[Game] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in sorted(RAW_DIR.glob("*.json")):
        slate_date = datetime.strptime(path.stem, "%Y%m%d").date()
        payload = json.loads(path.read_text(encoding="utf-8"))
        day_games, day_skipped = games_from_scoreboard(payload, slate_date)
        for game in day_games:
            if game.game_id in seen:
                skipped.append({"game_id": game.game_id, "reason": "duplicate id"})
                continue
            seen.add(game.game_id)
            games.append(game)
        skipped.extend(day_skipped)

    GAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "game_id": game.game_id,
                "game_date": game.game_date.isoformat(),
                "season_start": game.season_start,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "home_score": game.home_score,
                "away_score": game.away_score,
                "home_fg": game.home_fg,
                "home_fga": game.home_fga,
                "home_fg3": game.home_fg3,
                "away_fg": game.away_fg,
                "away_fga": game.away_fga,
                "away_fg3": game.away_fg3,
                "home_fg3a": game.home_fg3a,
                "away_fg3a": game.away_fg3a,
                "home_rebounds": game.home_rebounds,
                "away_rebounds": game.away_rebounds,
                "home_orb": game.home_orb,
                "away_orb": game.away_orb,
                "home_fta": game.home_fta,
                "away_fta": game.away_fta,
                "home_turnovers": game.home_turnovers,
                "away_turnovers": game.away_turnovers,
                "home_win": game.home_win,
            }
            for game in games
        ]
    ).to_csv(GAMES_CSV, index=False)
    pd.DataFrame(skipped).to_csv(SKIPPED_CSV, index=False)
    print(f"wrote {len(games)} games to {GAMES_CSV}", flush=True)
    print(f"skipped {len(skipped)} games", flush=True)
    return games


def _optional_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    return int(value)


def read_games() -> list[Game]:
    frame = pd.read_csv(GAMES_CSV)
    games: list[Game] = []
    for row in frame.itertuples(index=False):
        games.append(
            Game(
                game_id=str(row.game_id),
                game_date=datetime.strptime(str(row.game_date), "%Y-%m-%d").date(),
                season_start=int(row.season_start),
                home_team=str(row.home_team),
                away_team=str(row.away_team),
                home_score=int(row.home_score),
                away_score=int(row.away_score),
                home_fg=_optional_int(row.home_fg),
                home_fga=_optional_int(row.home_fga),
                home_fg3=_optional_int(row.home_fg3),
                away_fg=_optional_int(row.away_fg),
                away_fga=_optional_int(row.away_fga),
                away_fg3=_optional_int(row.away_fg3),
                home_fg3a=_optional_int(getattr(row, "home_fg3a", None)),
                away_fg3a=_optional_int(getattr(row, "away_fg3a", None)),
                home_rebounds=_optional_int(getattr(row, "home_rebounds", None)),
                away_rebounds=_optional_int(getattr(row, "away_rebounds", None)),
                home_orb=_optional_int(getattr(row, "home_orb", None)),
                away_orb=_optional_int(getattr(row, "away_orb", None)),
                home_fta=_optional_int(getattr(row, "home_fta", None)),
                away_fta=_optional_int(getattr(row, "away_fta", None)),
                home_turnovers=_optional_int(getattr(row, "home_turnovers", None)),
                away_turnovers=_optional_int(getattr(row, "away_turnovers", None)),
            )
        )
    return games
