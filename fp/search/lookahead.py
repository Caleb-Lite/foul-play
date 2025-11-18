from __future__ import annotations

from typing import Dict, Iterable

import constants
from fp.battle import Battle
from fp.strategy.damage import estimate_damage
from fp.strategy.risk import MoveRiskProfile


def _opponent_best_response(battle: Battle, aggression: float) -> float:
    if not battle.opponent.active or not battle.user.active:
        return 0.0

    best_damage = 0.0
    for move in battle.opponent.active.moves:
        if move.disabled or move.current_pp <= 0:
            continue
        damage = estimate_damage(battle, move.name, attacker_side="opponent")
        score = damage.expected_damage
        if aggression > 0.7:
            score *= 1.1
        best_damage = max(best_damage, score)

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
    double_switch_bias = opponent_profile.get("double_switch_success", 0.0)
    sack_bias = opponent_profile.get("sack_timing", 0.0)

    for move_name in candidate_moves:
        profile = risk_profiles.get(move_name)
        if not profile:
            continue

        if move_name.startswith(constants.SWITCH_STRING):
            active = battle.user.active
            max_hp = active.max_hp if active and active.max_hp else 1.0
            hazard_penalty = profile.hazard_cost / max(1.0, max_hp)
            evaluations[move_name] = tempo_bias - safety_penalty - hazard_penalty - double_switch_bias * 0.4
            continue

        net_damage = profile.expected_value - opponent_damage
        variance_penalty = profile.variance * (1.0 - opponent_profile.get("risk_tolerance", 0.5))
        score = (net_damage / 100.0) + tempo_bias - variance_penalty - safety_penalty
        if profile.description == "finisher":
            score += 0.3 + sack_bias * 0.2
        else:
            score += sack_bias * 0.05
        evaluations[move_name] = score

    return evaluations
