from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fp.battle import Battle


@dataclass
class TimeManager:
    """Keeps track of time pressure and critical turns."""

    last_time_allocation: float = 1.0
    high_leverage_turn: bool = False

    def reset(self) -> None:
        self.last_time_allocation = 1.0
        self.high_leverage_turn = False

    def update_from_metrics(self, metrics: Optional[Dict]) -> None:
        if not metrics:
            self.high_leverage_turn = False
            return
        momentum = metrics.get("momentum", 0.5)
        has_wincon = metrics.get("has_win_condition", False)
        hp_ratio = metrics.get("hp_advantage", {}).get("hp_ratio", 1.0)

        self.high_leverage_turn = momentum < 0.4 or not has_wincon or hp_ratio < 0.8
        self.last_time_allocation = 1.5 if self.high_leverage_turn else 1.0

    def should_skip_deep_search(self, battle: "Battle") -> bool:
        if not battle:
            return False
        if battle.team_preview:
            return False
        if battle.time_remaining is None:
            return False
        if battle.time_remaining > 45:
            return False
        return not self.high_leverage_turn
