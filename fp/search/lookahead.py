from __future__ import annotations

from typing import Dict, Iterable

from data import all_move_json
from fp.battle import Battle
from fp.helpers import type_effectiveness_modifier
from fp.strategy.risk import MoveRiskProfile


def _estimate_damage(move_name: str, attacker, defender) -> float:
    move_data = all_move_json.get(move_name)
    if not move_data:
        return 0.0
    if move_data.get("category") not in ("physical", "special"):
        return 0.0
    base_power = move_data.get("basePower", 0)
    move_type = move_data.get("type", "normal")
    effectiveness = type_effectiveness_modifier(move_type, defender.types)
    stab = 1.5 if move_type in attacker.types else 1.0
    return base_power * effectiveness * stab


def _opponent_best_response(battle: Battle, aggression: float) -> float:
    if not battle.opponent.active or not battle.user.active:
        return 0.0

    best_damage = 0.0
    for move in battle.opponent.active.moves:
        move_data = all_move_json.get(move.name)
        if not move_data or move_data.get("category") not in ("physical", "special"):
            continue
        accuracy = move_data.get("accuracy", 100) or 100
        raw = _estimate_damage(move.name, battle.opponent.active, battle.user.active)
        score = raw * (accuracy / 100.0)
        if aggression > 0.7:
            score *= 1.1
        if score > best_damage:
            best_damage = score

    return best_damage


def evaluate_candidate_lines(
    battle: Battle,
    candidate_moves: Iterable[str],
    opponent_profile: Dict[str, float],
    risk_profiles: Dict[str, MoveRiskProfile],
    position_metrics: Dict,
) -> Dict[str, float]:
    evaluations: Dict[str, float] = {}
    opponent_damage = _opponent_best_response(
        battle, opponent_profile.get("aggression", 0.5)
    )
    tempo_bias = position_metrics.get("tempo", {}).get("tempo_score", 0.5) - 0.5
    safety_penalty = (
        0.2 if position_metrics.get("positional_safety", {}).get("opponent_setup_window") else 0.0
    )

    for move_name in candidate_moves:
        profile = risk_profiles.get(move_name)
        if not profile:
            continue

        if move_name.startswith("switch"):
            evaluations[move_name] = tempo_bias - safety_penalty
            continue

        net_damage = profile.expected_value - opponent_damage
        variance_penalty = profile.variance * (1.0 - opponent_profile.get("risk_tolerance", 0.5))
        score = (net_damage / 100.0) + tempo_bias - variance_penalty - safety_penalty
        if profile.description == "finisher":
            score += 0.3
        evaluations[move_name] = score

    return evaluations
