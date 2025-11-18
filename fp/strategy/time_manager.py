from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

from config import FoulPlayConfig

if TYPE_CHECKING:
    from fp.battle import Battle


@dataclass
class TimeManager:
    """Keeps track of time pressure and critical turns."""

    bank_ms: float = FoulPlayConfig.search_time_ms * 15
    last_time_allocation: float = 0.0
    high_leverage_turn: bool = False
    safety_buffer_s: float = 6.0
    allocation_history: list[float] = field(default_factory=list)
    min_allocation_ms: int = 75
    max_allocation_ms: int = 4000

    def reset(self) -> None:
        self.bank_ms = FoulPlayConfig.search_time_ms * 15
        self.last_time_allocation = 0.0
        self.high_leverage_turn = False
        self.allocation_history = []

    def update_from_metrics(self, metrics: Optional[Dict]) -> None:
        if not metrics:
            self.high_leverage_turn = False
            return
        momentum = metrics.get("momentum", 0.5)
        has_wincon = metrics.get("has_win_condition", False)
        hp_ratio = metrics.get("hp_advantage", {}).get("hp_ratio", 1.0)

        self.high_leverage_turn = momentum < 0.4 or not has_wincon or hp_ratio < 0.8
        self.last_time_allocation = 1.5 if self.high_leverage_turn else 1.0

    def _sync_clock(self, battle: "Battle") -> None:
        if not battle or battle.time_remaining is None:
            return
        safe_seconds = max(0.0, battle.time_remaining - self.safety_buffer_s)
        self.bank_ms = max(safe_seconds * 1000.0, self.min_allocation_ms * 3)

    @staticmethod
    def _estimate_turns_remaining(battle: "Battle") -> int:
        user_team = [battle.user.active] + battle.user.reserve if battle.user.active else battle.user.reserve
        opp_team = [battle.opponent.active] + battle.opponent.reserve if battle.opponent.active else battle.opponent.reserve
        user_alive = sum(1 for p in user_team if p and p.hp > 0)
        opp_alive = sum(1 for p in opp_team if p and p.hp > 0)
        return max(3, user_alive + opp_alive)

    def allocate_search_time(self, battle: "Battle", criticality: float) -> int:
        self._sync_clock(battle)
        leverage_bonus = 0.4 if self.high_leverage_turn else 0.0
        desired = FoulPlayConfig.search_time_ms * max(1.0, criticality + leverage_bonus)
        desired = min(desired, self.max_allocation_ms)
        remaining_turns = self._estimate_turns_remaining(battle)
        soft_limit = self.bank_ms / max(remaining_turns, 1)
        allocation = min(desired, soft_limit)
        allocation = max(self.min_allocation_ms, allocation)
        self.bank_ms = max(0.0, self.bank_ms - allocation)
        self.last_time_allocation = allocation
        self.allocation_history.append(allocation)
        return int(allocation)

    def should_skip_deep_search(self, battle: "Battle") -> bool:
        if not battle:
            return False
        if battle.team_preview:
            return False
        self._sync_clock(battle)
        if battle.time_remaining is None:
            return False
        if battle.time_remaining > 30:
            return False
        if self.high_leverage_turn:
            return False
        if battle.time_remaining <= 8:
            return True
        return self.bank_ms < self.min_allocation_ms * 2
