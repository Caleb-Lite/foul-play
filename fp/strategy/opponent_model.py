from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

import constants
from data import all_move_json
from fp.helpers import normalize_name, type_effectiveness_modifier

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fp.battle import Battle
    from fp.battle import Pokemon


@dataclass
class PokemonObservation:
    name: str
    moves_revealed: set[str] = field(default_factory=set)
    tera_type: Optional[str] = None
    tera_turn: Optional[int] = None
    last_seen_turn: Optional[int] = None

    def record_move(self, move_name: str) -> bool:
        if move_name in self.moves_revealed:
            return False
        self.moves_revealed.add(move_name)
        return True


@dataclass
class OpponentBehaviorStats:
    aggressive_actions: int = 0
    passive_actions: int = 0
    risky_actions: int = 0
    safe_actions: int = 0
    sacks: int = 0
    double_switch_attempts: int = 0
    double_switch_successes: int = 0
    double_switches: int = 0
    sack_turns: list[int] = field(default_factory=list)

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

    def double_switch_success_rate(self) -> float:
        if self.double_switch_attempts == 0:
            return 0.0
        return self.double_switch_successes / self.double_switch_attempts

    def sack_timing_index(self) -> float:
        if not self.sack_turns:
            return 0.0
        early_sacks = sum(1 for turn in self.sack_turns if turn <= 8)
        return early_sacks / len(self.sack_turns)


class OpponentModel:
    """Online opponent model that tracks tendencies during a match."""

    def __init__(self) -> None:
        self.stats = OpponentBehaviorStats()
        self.last_recorded_turn: Optional[int] = None
        self.turn_history: list[Dict] = []
        self.pokemon_logs: Dict[str, PokemonObservation] = {}
        self.pending_sacks: Dict[str, int] = {}
        self.fainted_pokemon: set[str] = set()
        self.sack_events: list[Dict] = []
        self.double_switch_history: list[Dict] = []
        self.tera_record: Optional[Dict] = None

    def reset(self) -> None:
        self.stats = OpponentBehaviorStats()
        self.last_recorded_turn = None
        self.turn_history = []
        self.pokemon_logs = {}
        self.pending_sacks = {}
        self.fainted_pokemon = set()
        self.sack_events = []
        self.double_switch_history = []
        self.tera_record = None

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

    def _get_pokemon_log(self, pokemon_name: str) -> PokemonObservation:
        pokemon_name = normalize_name(pokemon_name)
        if pokemon_name not in self.pokemon_logs:
            self.pokemon_logs[pokemon_name] = PokemonObservation(name=pokemon_name)
        return self.pokemon_logs[pokemon_name]

    def _find_opponent_pokemon(self, battle: "Battle", pokemon_name: str) -> Optional["Pokemon"]:
        if not pokemon_name:
            return None
        normalized = normalize_name(pokemon_name)
        active = battle.opponent.active
        if active and normalize_name(active.name) == normalized:
            return active
        for pokemon in battle.opponent.reserve:
            if normalize_name(pokemon.name) == normalized:
                return pokemon
        return None

    @staticmethod
    def _best_effectiveness(attacker: "Pokemon", defender: "Pokemon") -> float:
        if not attacker or not defender:
            return 0.0
        best = 0.0
        for move in attacker.moves:
            move_data = all_move_json.get(move.name)
            if not move_data or move_data.get("category") not in ("physical", "special"):
                continue
            move_type = move_data.get("type", "normal")
            effectiveness = type_effectiveness_modifier(move_type, defender.types)
            if effectiveness > best:
                best = effectiveness
        return best

    def _assess_double_switch(self, battle: "Battle") -> float:
        if not battle.user.active or not battle.opponent.active:
            return 0.0
        opp_pressure = self._best_effectiveness(battle.opponent.active, battle.user.active)
        user_pressure = self._best_effectiveness(battle.user.active, battle.opponent.active)
        return opp_pressure - user_pressure

    def _update_sack_state(self, battle: "Battle", turn: int) -> None:
        active = battle.opponent.active
        if active and active.hp > 0 and active.max_hp:
            hp_ratio = active.hp / active.max_hp
            if hp_ratio <= 0.25:
                self.pending_sacks.setdefault(normalize_name(active.name), turn)

        for tracked in list(self.pending_sacks.keys()):
            pokemon = self._find_opponent_pokemon(battle, tracked)
            if not pokemon or not pokemon.max_hp:
                continue
            hp_ratio = pokemon.hp / pokemon.max_hp
            if hp_ratio > 0.5:
                self.pending_sacks.pop(tracked, None)

        roster = []
        if battle.opponent.active:
            roster.append(battle.opponent.active)
        roster.extend(battle.opponent.reserve)

        for pokemon in roster:
            if not pokemon or pokemon.hp > 0:
                continue
            normalized = normalize_name(pokemon.name)
            if normalized in self.fainted_pokemon:
                continue
            self.fainted_pokemon.add(normalized)
            sack_start = self.pending_sacks.pop(normalized, None)
            duration = turn - sack_start if sack_start is not None else 0
            self.sack_events.append(
                {"pokemon": normalized, "turn": turn, "duration": duration}
            )
            self.stats.sacks += 1
            self.stats.sack_turns.append(turn)
            if sack_start is not None:
                logger.info(
                    "Opponent sacked {} on turn {} after {} turns at low HP".format(
                        pokemon.name, turn, duration
                    )
                )
            else:
                logger.info("Opponent lost {} on turn {} (forced KO)".format(pokemon.name, turn))

    def observe_turn(self, battle: "Battle") -> None:
        if not battle or not battle.opponent:
            return

        turn = getattr(battle, "turn", None)
        if turn is None or turn == self.last_recorded_turn:
            return

        self._update_sack_state(battle, turn)

        last_move = battle.opponent.last_used_move.move
        if not last_move:
            return

        opp_last_pokemon = battle.opponent.last_used_move.pokemon_name or (
            battle.opponent.active.name if battle.opponent.active else None
        )
        if opp_last_pokemon:
            observation = self._get_pokemon_log(opp_last_pokemon)
            observation.last_seen_turn = turn
            if last_move and not last_move.startswith(constants.SWITCH_STRING):
                if observation.record_move(last_move):
                    logger.info("Opponent {} revealed move {}".format(opp_last_pokemon, last_move))

        active = battle.opponent.active
        if active and active.terastallized:
            observation = self._get_pokemon_log(active.name)
            if observation.tera_turn is None:
                observation.tera_type = active.tera_type
                observation.tera_turn = turn
                self.tera_record = {
                    "pokemon": normalize_name(active.name),
                    "type": active.tera_type,
                    "turn": turn,
                }
                logger.info(
                    "Opponent terastallized {} into {} on turn {}".format(
                        active.name, active.tera_type, turn
                    )
                )

        user_last_move = battle.user.last_used_move.move
        if (
            user_last_move
            and last_move.startswith(constants.SWITCH_STRING)
            and user_last_move.startswith(constants.SWITCH_STRING)
            and battle.opponent.last_used_move.turn == battle.user.last_used_move.turn
        ):
            delta = self._assess_double_switch(battle)
            self.stats.double_switch_attempts += 1
            if delta > 0.1:
                self.stats.double_switch_successes += 1
            self.double_switch_history.append(
                {
                    "turn": turn,
                    "delta": delta,
                    "opponent_active": battle.opponent.active.name if battle.opponent.active else None,
                    "user_active": battle.user.active.name if battle.user.active else None,
                }
            )
            logger.info(
                "Double switch detected (opponent delta {:.2f}) - {} vs {}".format(
                    delta,
                    battle.opponent.active.name if battle.opponent.active else "unknown",
                    battle.user.active.name if battle.user.active else "unknown",
                )
            )
            self.stats.double_switches += 1

        info = self._categorize_move(last_move)
        if info["category"] in ("physical", "special"):
            self.stats.aggressive_actions += 1
        elif info["category"] == "status":
            self.stats.passive_actions += 1
        elif info["category"] == "switch":
            self.stats.passive_actions += 1
        else:
            self.stats.passive_actions += 1

        if info["is_risky"]:
            self.stats.risky_actions += 1
        else:
            self.stats.safe_actions += 1

        self.turn_history.append(
            {
                "turn": turn,
                "move": last_move,
                "category": info["category"],
                "is_risky": info["is_risky"],
            }
        )
        self.last_recorded_turn = turn

    def get_profile(self) -> Dict[str, object]:
        return {
            "aggression": self.stats.aggression_index(),
            "risk_tolerance": self.stats.risk_tolerance(),
            "sack_rate": self.stats.sack_rate(),
            "double_switch_rate": self.stats.double_switch_rate(),
            "double_switch_success": self.stats.double_switch_success_rate(),
            "sack_timing": self.stats.sack_timing_index(),
            "move_reveals": {
                name: sorted(observation.moves_revealed)
                for name, observation in self.pokemon_logs.items()
                if observation.moves_revealed
            },
            "tera_turn": self.tera_record["turn"] if self.tera_record else None,
            "tera_type": self.tera_record["type"] if self.tera_record else None,
        }
