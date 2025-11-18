from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import constants
from data import all_move_json
from fp.helpers import normalize_name, type_effectiveness_modifier
from fp.search.poke_engine_helpers import poke_engine_get_damage_rolls

logger = logging.getLogger(__name__)


@dataclass
class DamageEstimate:
    """Container for a single move's expected damage output."""

    name: str
    expected_damage: float
    variance: float
    min_damage: float
    max_damage: float
    hit_chance: float
    crit_chance: float
    on_hit_mean: float

    @property
    def fail_chance(self) -> float:
        return max(0.0, 1.0 - self.hit_chance)

    @classmethod
    def zero(cls, name: str) -> "DamageEstimate":
        return cls(
            name=name,
            expected_damage=0.0,
            variance=0.0,
            min_damage=0.0,
            max_damage=0.0,
            hit_chance=0.0,
            crit_chance=0.0,
            on_hit_mean=0.0,
        )


def _accuracy_from_move(move_data: dict) -> float:
    accuracy = move_data.get("accuracy", 100)
    if isinstance(accuracy, bool):
        return 1.0 if accuracy else 0.0
    try:
        return max(0.0, min(1.0, float(accuracy) / 100.0))
    except (TypeError, ValueError):
        return 1.0


def _crit_chance_from_move(move_data: dict) -> float:
    if move_data.get("willCrit"):
        return 1.0
    if move_data.get("noCrit"):
        return 0.0

    # Showdown defaults to 1/24 unless explicitly modified; the JSON supplied here
    # does not expose the crit ratio, so we stick to the baseline.
    return 1.0 / 24.0


def _fallback_damage_estimate(
    name: str,
    move_data: dict,
    attacker,
    defender,
) -> DamageEstimate:
    if not attacker or not defender:
        return DamageEstimate.zero(name)

    base_power = move_data.get("basePower", 0)
    move_type = move_data.get("type", "normal")
    effectiveness = type_effectiveness_modifier(move_type, defender.types)
    stab = 1.5 if move_type in attacker.types else 1.0

    raw = base_power * effectiveness * stab
    min_damage = raw * 0.85
    max_damage = raw
    on_hit_mean = (min_damage + max_damage) / 2.0
    hit_chance = _accuracy_from_move(move_data)

    expected = hit_chance * on_hit_mean
    # Very rough variance approximation that keeps the units in HP.
    spread = max_damage - min_damage
    on_hit_variance = (spread ** 2) / 12 if spread > 0 else 0.0
    variance = hit_chance * on_hit_variance + hit_chance * (1 - hit_chance) * (on_hit_mean ** 2)

    return DamageEstimate(
        name=name,
        expected_damage=expected,
        variance=variance,
        min_damage=min_damage,
        max_damage=max_damage,
        hit_chance=hit_chance,
        crit_chance=_crit_chance_from_move(move_data),
        on_hit_mean=on_hit_mean,
    )


def estimate_damage(
    battle,
    move_name: str,
    attacker_side: Literal["user", "opponent"] = "user",
) -> DamageEstimate:
    """Return the expected damage output for a single move using poke-engine rolls."""

    if move_name.startswith("switch"):
        return DamageEstimate.zero(move_name)

    normalized_move = normalize_name(move_name)
    move_data = all_move_json.get(normalized_move)
    if not move_data:
        return DamageEstimate.zero(normalized_move)

    if move_data.get("category") not in ("physical", "special"):
        return DamageEstimate.zero(normalized_move)

    attacker = battle.user.active if attacker_side == "user" else battle.opponent.active
    defender = battle.opponent.active if attacker_side == "user" else battle.user.active
    if not attacker or not defender:
        return DamageEstimate.zero(normalized_move)

    accuracy = _accuracy_from_move(move_data)
    crit_chance = _crit_chance_from_move(move_data)

    try:
        if attacker_side == "user":
            rolls, _ = poke_engine_get_damage_rolls(
                battle, normalized_move, constants.DO_NOTHING_MOVE, True
            )
        else:
            _, rolls = poke_engine_get_damage_rolls(
                battle, constants.DO_NOTHING_MOVE, normalized_move, False
            )
    except Exception as exc:  # pragma: no cover - fallback path
        logger.debug("Falling back to heuristic damage estimate for %s: %s", normalized_move, exc)
        return _fallback_damage_estimate(normalized_move, move_data, attacker, defender)

    if not rolls:
        return _fallback_damage_estimate(normalized_move, move_data, attacker, defender)

    max_noncrit = rolls[0]
    max_crit = rolls[1] if len(rolls) > 1 else max_noncrit
    min_noncrit = max_noncrit * 0.85
    min_crit = max_crit * 0.85

    mean_noncrit = (min_noncrit + max_noncrit) / 2.0
    mean_crit = (min_crit + max_crit) / 2.0
    on_hit_mean = (1 - crit_chance) * mean_noncrit + crit_chance * mean_crit

    spread_noncrit = max_noncrit - min_noncrit
    spread_crit = max_crit - min_crit
    var_noncrit = (spread_noncrit ** 2) / 12 if spread_noncrit > 0 else 0.0
    var_crit = (spread_crit ** 2) / 12 if spread_crit > 0 else 0.0
    on_hit_variance = (1 - crit_chance) * var_noncrit + crit_chance * var_crit
    on_hit_variance += (1 - crit_chance) * (mean_noncrit - on_hit_mean) ** 2
    on_hit_variance += crit_chance * (mean_crit - on_hit_mean) ** 2

    expected_damage = accuracy * on_hit_mean
    variance = accuracy * on_hit_variance + accuracy * (1 - accuracy) * (on_hit_mean ** 2)

    return DamageEstimate(
        name=normalized_move,
        expected_damage=expected_damage,
        variance=variance,
        min_damage=min_noncrit,
        max_damage=max_crit,
        hit_chance=accuracy,
        crit_chance=crit_chance,
        on_hit_mean=on_hit_mean,
    )
