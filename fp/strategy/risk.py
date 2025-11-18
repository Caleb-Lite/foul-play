from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, TYPE_CHECKING

import constants
from fp.helpers import normalize_name, type_effectiveness_modifier
from fp.strategy.damage import DamageEstimate, estimate_damage

if TYPE_CHECKING:
    from fp.battle import Battle
    from fp.battle import Pokemon

logger = logging.getLogger(__name__)


SPIKES_FRACTIONS = {1: 1 / 8, 2: 3 / 16, 3: 1 / 4}
TOXIC_SPIKES_PENALTY = {1: 1 / 8, 2: 3 / 16}


@dataclass
class MoveRiskProfile:
    name: str
    expected_value: float
    fail_chance: float
    variance: float
    min_damage: float = 0.0
    max_damage: float = 0.0
    hazard_cost: float = 0.0
    crit_chance: float = 0.0
    description: str = ""


class RiskRewardAnalyzer:
    """Estimates expected value and risk for candidate moves."""

    def __init__(self) -> None:
        self.turn_cache: Dict[str, MoveRiskProfile] = {}

    def reset(self) -> None:
        self.turn_cache = {}

    @staticmethod
    def _stealth_rock_damage(pokemon: "Pokemon") -> float:
        if not pokemon or not pokemon.types:
            return 0.0
        multiplier = type_effectiveness_modifier("rock", pokemon.types)
        return multiplier / 8.0 * pokemon.max_hp

    @staticmethod
    def _is_grounded(pokemon: "Pokemon") -> bool:
        if not pokemon:
            return False
        if pokemon.has_type("flying"):
            return False
        if pokemon.ability == "levitate" and not pokemon.terastallized:
            return False
        if pokemon.item == "airballoon":
            return False
        volatiles = set(pokemon.volatile_statuses)
        if "magnetrise" in volatiles or "telekinesis" in volatiles:
            return False
        return True

    def _resolve_switch_target(self, battle: "Battle", move_name: str) -> Optional["Pokemon"]:
        if " " not in move_name:
            return None
        _, identifier = move_name.split(" ", 1)
        identifier = identifier.strip()
        normalized_identifier = normalize_name(identifier)
        for pokemon in battle.user.reserve:
            if normalize_name(pokemon.name) == normalized_identifier:
                return pokemon
            index = getattr(pokemon, "index", None)
            if identifier.isdigit() and index and index == int(identifier):
                return pokemon
        return None

    def _estimate_switch_hazard_cost(self, battle: "Battle", target: "Pokemon") -> float:
        if not target:
            return 0.0

        if target.item == "heavydutyboots" and not target.knocked_off:
            return 0.0

        hazards = battle.user.side_conditions
        hazard_damage = 0.0

        if hazards.get("stealthrock"):
            hazard_damage += self._stealth_rock_damage(target)

        spikes_layers = int(hazards.get("spikes", 0))
        if spikes_layers and self._is_grounded(target):
            hazard_damage += target.max_hp * SPIKES_FRACTIONS.get(spikes_layers, SPIKES_FRACTIONS[3])

        toxic_layers = int(hazards.get("toxicspikes", 0))
        if (
            toxic_layers
            and self._is_grounded(target)
            and target.status is None
            and not target.has_type("poison")
            and not target.has_type("steel")
        ):
            hazard_damage += target.max_hp * TOXIC_SPIKES_PENALTY.get(
                min(toxic_layers, 2), TOXIC_SPIKES_PENALTY[1]
            )

        return hazard_damage

    @contextmanager
    def _temporary_tera(self, battle: "Battle", should_tera: bool):
        pokemon = battle.user.active
        applied = False
        if not should_tera or not pokemon or pokemon.terastallized or not pokemon.tera_type:
            yield applied
            return

        original_types = pokemon.types
        original_state = pokemon.terastallized
        try:
            pokemon.terastallized = True
            pokemon.types = (pokemon.tera_type,)
            applied = True
            yield applied
        finally:
            pokemon.types = original_types
            pokemon.terastallized = original_state

    @staticmethod
    def _split_move_metadata(move_name: str) -> tuple[str, Dict[str, bool]]:
        normalized = move_name.strip()
        metadata = {"tera": False, "mega": False}
        if normalized.endswith("-tera"):
            metadata["tera"] = True
            normalized = normalized[: -len("-tera")]
        if normalized.endswith("-mega"):
            metadata["mega"] = True
            normalized = normalized[: -len("-mega")]
        return normalized, metadata

    def _build_profile_from_damage(
        self,
        move_name: str,
        damage: DamageEstimate,
        leverage_multiplier: float,
        description: str,
        hazard_cost: float = 0.0,
    ) -> MoveRiskProfile:
        return MoveRiskProfile(
            name=move_name,
            expected_value=damage.expected_damage - hazard_cost,
            fail_chance=damage.fail_chance,
            variance=damage.variance * leverage_multiplier,
            min_damage=damage.min_damage,
            max_damage=damage.max_damage,
            hazard_cost=hazard_cost,
            crit_chance=damage.crit_chance,
            description=description,
        )

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

        leverage_multiplier = 1.0
        if position_metrics:
            momentum = position_metrics.get("momentum", 0.5)
            leverage_multiplier = 1.5 if momentum < 0.4 else 0.75 if momentum > 0.7 else 1.0

        for move_name in moves_to_score:
            if move_name.startswith(constants.SWITCH_STRING):
                target = self._resolve_switch_target(battle, move_name)
                hazard_cost = self._estimate_switch_hazard_cost(battle, target)
                description = "switch" if hazard_cost <= 1 else "switch-hazard"
                self.turn_cache[move_name] = MoveRiskProfile(
                    name=move_name,
                    expected_value=-hazard_cost,
                    fail_chance=0.0,
                    variance=hazard_cost * leverage_multiplier * 0.05,
                    min_damage=-hazard_cost,
                    max_damage=0.0,
                    hazard_cost=hazard_cost,
                    crit_chance=0.0,
                    description=description,
                )
                continue

            normalized_name, metadata = self._split_move_metadata(move_name)
            normalized_name = normalize_name(normalized_name)

            with self._temporary_tera(battle, metadata["tera"]):
                damage = estimate_damage(battle, normalized_name, attacker_side="user")

            description = (
                "tera"
                if metadata["tera"]
                else "mega" if metadata["mega"] else "raw"
            )
            if damage.fail_chance >= 0.3:
                description = "high-risk"

            if (
                battle.opponent.active
                and battle.opponent.active.hp
                and damage.max_damage >= battle.opponent.active.hp
            ):
                description = "finisher"

            self.turn_cache[move_name] = self._build_profile_from_damage(
                move_name, damage, leverage_multiplier, description
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
