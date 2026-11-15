"""One regular-season game, after the two team logs have been joined."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Game:
    game_id: str
    game_date: date
    season_start: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_fg: int | None = None
    home_fga: int | None = None
    home_fg3: int | None = None
    away_fg: int | None = None
    away_fga: int | None = None
    away_fg3: int | None = None
    home_fg3a: int | None = None
    away_fg3a: int | None = None
    home_rebounds: int | None = None
    away_rebounds: int | None = None
    home_orb: int | None = None
    away_orb: int | None = None
    home_fta: int | None = None
    away_fta: int | None = None
    home_turnovers: int | None = None
    away_turnovers: int | None = None

    @property
    def has_shooting(self) -> bool:
        values = (
            self.home_fg,
            self.home_fga,
            self.home_fg3,
            self.away_fg,
            self.away_fga,
            self.away_fg3,
        )
        return all(value is not None for value in values) and self.home_fga > 0 and self.away_fga > 0

    @property
    def home_win(self) -> int:
        return int(self.home_score > self.away_score)
