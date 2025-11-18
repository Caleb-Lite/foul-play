from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, TYPE_CHECKING

from data import all_move_json
from fp.helpers import type_effectiveness_modifier

if TYPE_CHECKING:
    from fp.battle import Battle


@dataclass
class MoveRiskProfile:
    name: str
    expected_value: float
    fail_chance: float
    variance: float
    description: str = ""


class RiskRewardAnalyzer:
    """Estimates expected value and risk for candidate moves."""

    def __init__(self) -> None:
        self.turn_cache: Dict[str, MoveRiskProfile] = {}

    def reset(self) -> None:
        self.turn_cache = {}

    def _estimate_damage(self, move_name: str, attacker, defender) -> float:
        move_data = all_move_json.get(move_name)
        if not move_data:
            return 0.0

        category = move_data.get("category")
        if category not in ("physical", "special"):
            return 0.0

        base_power = move_data.get("basePower", 0)
        move_type = move_data.get("type", "normal")
        effectiveness = type_effectiveness_modifier(move_type, defender.types)
        stab = 1.5 if move_type in attacker.types else 1.0

        return base_power * effectiveness * stab

    def evaluate_moves(
        self,
        battle: "Battle",
        position_metrics: Optional[Dict] = None,
        candidate_moves: Optional[Iterable[str]] = None,
    ) -> Dict[str, MoveRiskProfile]:
        if not battle.user.active or not battle.opponent.active:
            return self.turn_cache

        if candidate_moves is None:
            self.turn_cache = {}
            moves = [m.name for m in battle.user.active.moves]
        else:
            moves = list(candidate_moves)
            if not self.turn_cache:
                self.turn_cache = {}

        if not moves:
            return self.turn_cache

        moves_to_score = [mv for mv in moves if mv not in self.turn_cache]
        if not moves_to_score:
            return {mv: self.turn_cache[mv] for mv in moves if mv in self.turn_cache}

        hp = battle.opponent.active.hp or 1
        leverage_multiplier = 1.0
        if position_metrics:
            momentum = position_metrics.get("momentum", 0.5)
            leverage_multiplier = 1.5 if momentum < 0.4 else 0.75 if momentum > 0.7 else 1.0

        for move_name in moves_to_score:
            move_data = all_move_json.get(move_name, {})
            accuracy = move_data.get("accuracy", 100) or 100
            fail_chance = 1.0 - min(accuracy, 100) / 100.0
            raw_damage = self._estimate_damage(
                move_name, battle.user.active, battle.opponent.active
            )
            expected_value = max(0.0, raw_damage * (1 - fail_chance))
            variance = raw_damage * fail_chance * leverage_multiplier

            description = "raw"
            if fail_chance >= 0.3:
                description = "high-risk"
            elif expected_value / hp > 0.7:
                description = "finisher"

            self.turn_cache[move_name] = MoveRiskProfile(
                name=move_name,
                expected_value=expected_value,
                fail_chance=fail_chance,
                variance=variance,
                description=description,
            )

        return {mv: self.turn_cache[mv] for mv in moves if mv in self.turn_cache}

    def serialize_turn(
        self,
        battle: "Battle",
        selected_move: str,
        position_metrics: Dict,
        policy: Dict[str, float],
    ) -> Dict:
        return {
            "turn": battle.turn,
            "selected_move": selected_move,
            "hp_diff": position_metrics.get("hp_advantage", {}).get("hp_diff"),
            "momentum": position_metrics.get("momentum"),
            "win_condition": position_metrics.get("has_win_condition"),
            "policy": policy,
            "risk_profiles": {
                move: profile.__dict__ for move, profile in self.turn_cache.items()
            },
        }
