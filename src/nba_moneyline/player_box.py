"""Player box scores and the rotation gap versus a team's usual mix.

A player's rate is his box score per minute, shrunk toward the team's own
per-minute average. Tonight's rotation is whoever played the team's previous
game, weighted by those minutes. The usual mix is everyone who has played for
that team this season, weighted by season minutes. The feature is the
difference. The first game of a season is 0, because there is no season mix yet.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from nba_moneyline.closing_odds import SUMMARY_DIR, ensure_summary
from nba_moneyline.constants import ROTATION_PRIOR_MINUTES, TEAMS
from nba_moneyline.schema import Game

PLAYER_LINES_CSV = SUMMARY_DIR.parents[1] / "player_lines.csv"
TEAM_SET = set(TEAMS)


@dataclass(frozen=True, slots=True)
class PlayerLine:
    player_id: str
    team: str
    minutes: float
    box: float


@dataclass
class _PlayerTotals:
    minutes: float = 0.0
    box: float = 0.0


@dataclass
class RotationBook:
    """Prior-game player state. Update it only after a date has been featurized."""

    prior_minutes: float = ROTATION_PRIOR_MINUTES
    players: dict[str, _PlayerTotals] = field(default_factory=lambda: defaultdict(_PlayerTotals))
    team_minutes: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    team_box: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    team_season: dict[str, int] = field(default_factory=dict)
    season_minutes: dict[tuple[str, str], float] = field(default_factory=dict)
    last_rotation: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def delta(self, team: str, season: int) -> float:
        if self.team_season.get(team) != season:
            return 0.0
        last = self.last_rotation.get(team) or []
        usual = [
            (player_id, minutes)
            for (player_id, player_team), minutes in self.season_minutes.items()
            if player_team == team and minutes > 0
        ]
        if not last or not usual or self.team_minutes[team] <= 0:
            return 0.0
        team_average = self.team_box[team] / self.team_minutes[team]
        last_strength = self._mix(last, team_average)
        usual_strength = self._mix(usual, team_average)
        return last_strength - usual_strength

    def apply(self, team: str, season: int, lines: list[PlayerLine]) -> None:
        played = [line for line in lines if line.team == team and line.minutes > 0]
        if not played:
            return
        if self.team_season.get(team) != season:
            self.team_season[team] = season
            self.season_minutes = {
                key: minutes for key, minutes in self.season_minutes.items() if key[1] != team
            }
        self.last_rotation[team] = [(line.player_id, line.minutes) for line in played]
        for line in played:
            player = self.players[line.player_id]
            player.minutes += line.minutes
            player.box += line.box
            self.team_minutes[team] += line.minutes
            self.team_box[team] += line.box
            key = (line.player_id, team)
            self.season_minutes[key] = self.season_minutes.get(key, 0.0) + line.minutes

    def _rate(self, player_id: str, team_average: float) -> float:
        player = self.players[player_id]
        weight = player.minutes + self.prior_minutes
        if weight <= 0:
            return team_average
        return (player.box + self.prior_minutes * team_average) / weight

    def _mix(self, weights: list[tuple[str, float]], team_average: float) -> float:
        total = sum(minutes for _player_id, minutes in weights)
        return sum(minutes * self._rate(player_id, team_average) for player_id, minutes in weights) / total


def box_score(
    points: float,
    rebounds: float,
    assists: float,
    steals: float,
    blocks: float,
    turnovers: float,
    field_goals_made: float,
    field_goals_attempted: float,
    free_throws_made: float,
    free_throws_attempted: float,
) -> float:
    """Points, rebounds, assists, steals, and blocks, minus turnovers and misses."""

    missed = (field_goals_attempted - field_goals_made) + (free_throws_attempted - free_throws_made)
    return points + rebounds + assists + steals + blocks - turnovers - missed


def parse_minutes(text: object) -> float:
    raw = str(text or "").strip()
    if raw in {"", "--", "DNP"}:
        return 0.0
    if ":" in raw:
        minutes, seconds = raw.split(":", 1)
        return float(minutes) + float(seconds) / 60.0
    return float(raw)


def parse_made_attempted(text: object) -> tuple[float, float]:
    raw = str(text or "").strip()
    if raw in {"", "--"} or "-" not in raw:
        return 0.0, 0.0
    made, attempted = raw.split("-", 1)
    return float(made), float(attempted)


def player_lines_from_summary(payload: dict) -> list[PlayerLine]:
    """Read both teams' box scores. Players who did not play are omitted."""

    lines: list[PlayerLine] = []
    for team_block in (payload.get("boxscore") or {}).get("players") or []:
        team = str((team_block.get("team") or {}).get("abbreviation") or "")
        if team not in TEAM_SET:
            continue
        for table in team_block.get("statistics") or []:
            keys = [str(key) for key in table.get("keys") or []]
            for athlete in table.get("athletes") or []:
                line = _line_from_athlete(athlete, keys, team)
                if line is not None:
                    lines.append(line)
    return lines


def _line_from_athlete(athlete: dict, keys: list[str], team: str) -> PlayerLine | None:
    if athlete.get("didNotPlay"):
        return None
    player_id = str((athlete.get("athlete") or {}).get("id") or "")
    stats = athlete.get("stats") or []
    if not player_id or len(stats) != len(keys):
        return None
    values = dict(zip(keys, stats, strict=True))
    try:
        minutes = parse_minutes(values.get("minutes"))
        made_fg, attempted_fg = parse_made_attempted(values.get("fieldGoalsMade-fieldGoalsAttempted"))
        made_ft, attempted_ft = parse_made_attempted(values.get("freeThrowsMade-freeThrowsAttempted"))
        box = box_score(
            points=float(values.get("points") or 0),
            rebounds=float(values.get("rebounds") or 0),
            assists=float(values.get("assists") or 0),
            steals=float(values.get("steals") or 0),
            blocks=float(values.get("blocks") or 0),
            turnovers=float(values.get("turnovers") or 0),
            field_goals_made=made_fg,
            field_goals_attempted=attempted_fg,
            free_throws_made=made_ft,
            free_throws_attempted=attempted_ft,
        )
    except (TypeError, ValueError):
        return None
    if minutes <= 0:
        return None
    return PlayerLine(player_id=player_id, team=team, minutes=minutes, box=box)


def download_player_summaries(games: list[Game], pause_seconds: float = 0.05) -> int:
    """Cache any missing ESPN summary for these games. Returns the download count."""

    downloaded = 0
    for index, game in enumerate(games, start=1):
        event_id = game.game_id.removeprefix("espn_")
        path = SUMMARY_DIR / f"{event_id}.json"
        if path.exists():
            continue
        ensure_summary(event_id)
        downloaded += 1
        if downloaded % 100 == 0:
            print(f"fetched {downloaded} summaries (through {index}/{len(games)})", flush=True)
        time.sleep(pause_seconds)
    print(f"player summaries ready ({downloaded} downloaded)", flush=True)
    return downloaded


def write_player_lines(games: list[Game]) -> pd.DataFrame:
    """Parse cached summaries into one row per player who played."""

    rows: list[dict[str, object]] = []
    missing = 0
    for game in games:
        event_id = game.game_id.removeprefix("espn_")
        path = SUMMARY_DIR / f"{event_id}.json"
        if not path.exists():
            missing += 1
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for line in player_lines_from_summary(payload):
            rows.append(
                {
                    "game_id": game.game_id,
                    "player_id": line.player_id,
                    "team": line.team,
                    "minutes": line.minutes,
                    "box": line.box,
                }
            )
    frame = pd.DataFrame(rows, columns=["game_id", "player_id", "team", "minutes", "box"])
    PLAYER_LINES_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PLAYER_LINES_CSV, index=False)
    print(
        f"wrote {len(frame)} player lines for {frame['game_id'].nunique() if not frame.empty else 0} games "
        f"to {PLAYER_LINES_CSV} ({missing} games had no summary)",
        flush=True,
    )
    return frame


def load_player_lines() -> dict[str, list[PlayerLine]]:
    """Player lines keyed by game id. An empty dict when the file has not been built."""

    if not PLAYER_LINES_CSV.is_file():
        return {}
    frame = pd.read_csv(PLAYER_LINES_CSV, dtype={"game_id": str, "player_id": str, "team": str})
    if frame.empty:
        return {}
    grouped: dict[str, list[PlayerLine]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        grouped[str(row.game_id)].append(
            PlayerLine(
                player_id=str(row.player_id),
                team=str(row.team),
                minutes=float(row.minutes),
                box=float(row.box),
            )
        )
    return grouped
