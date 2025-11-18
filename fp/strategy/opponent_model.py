from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

from data import all_move_json

if TYPE_CHECKING:
    from fp.battle import Battle


@dataclass
class OpponentBehaviorStats:
    aggressive_actions: int = 0
    passive_actions: int = 0
    risky_actions: int = 0
    safe_actions: int = 0
    sacks: int = 0
    double_switches: int = 0

    def aggression_index(self) -> float:
        total = self.aggressive_actions + self.passive_actions
        if total == 0:
            return 0.5
        return self.aggressive_actions / total

    def risk_tolerance(self) -> float:
        total = self.risky_actions + self.safe_actions
        if total == 0:
            return 0.5
        return self.risky_actions / total

    def sack_rate(self) -> float:
        actions = self.aggressive_actions + self.passive_actions
        if actions == 0:
            return 0.0
        return min(1.0, self.sacks / actions)

    def double_switch_rate(self) -> float:
        actions = self.aggressive_actions + self.passive_actions
        if actions == 0:
            return 0.0
        return min(1.0, self.double_switches / actions)


class OpponentModel:
    """Online opponent model that tracks tendencies during a match."""

    def __init__(self) -> None:
        self.stats = OpponentBehaviorStats()
        self.last_recorded_turn: Optional[int] = None
        self.turn_history: list[Dict] = []

    def reset(self) -> None:
        self.__init__()

    def _categorize_move(self, move_name: str) -> Dict:
        category = "unknown"
        accuracy = 100
        is_risky = False
        if move_name.startswith("switch"):
            category = "switch"
        else:
            move_data = all_move_json.get(move_name)
            if move_data:
                category = move_data.get("category", "unknown")
                accuracy = move_data.get("accuracy", 100) or 100
                is_risky = isinstance(accuracy, (int, float)) and accuracy < 90
        return {
            "category": category,
            "accuracy": accuracy,
            "is_risky": is_risky,
        }

    def observe_turn(self, battle: "Battle") -> None:
        if not battle or not battle.opponent:
            return

        turn = getattr(battle, "turn", None)
        if turn is None or turn == self.last_recorded_turn:
            return

        last_move = battle.opponent.last_used_move.move
        if not last_move:
            return

        info = self._categorize_move(last_move)
        if info["category"] in ("physical", "special"):
            self.stats.aggressive_actions += 1
        elif info["category"] == "status":
            self.stats.passive_actions += 1
        elif info["category"] == "switch":
            self.stats.passive_actions += 1
            if last_move.endswith(battle.user.active.name if battle.user.active else ""):
                self.stats.double_switches += 1
        else:
            self.stats.passive_actions += 1

        if info["is_risky"]:
            self.stats.risky_actions += 1
        else:
            self.stats.safe_actions += 1

        if battle.opponent.active and battle.opponent.active.hp <= 0:
            self.stats.sacks += 1

        self.turn_history.append(
            {
                "turn": turn,
                "move": last_move,
                "category": info["category"],
                "is_risky": info["is_risky"],
            }
        )
        self.last_recorded_turn = turn

    def get_profile(self) -> Dict[str, float]:
        return {
            "aggression": self.stats.aggression_index(),
            "risk_tolerance": self.stats.risk_tolerance(),
            "sack_rate": self.stats.sack_rate(),
            "double_switch_rate": self.stats.double_switch_rate(),
        }
