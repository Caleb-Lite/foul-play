from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Iterable, List

from data.pkmn_sets import (
    MOVES_STRING,
    RAW_COUNT,
    TERA_TYPE_STRING,
    SmogonSets,
)
from fp.helpers import normalize_name

if TYPE_CHECKING:
    from fp.battle import Battle

HAZARD_MOVES = {"stealthrock", "spikes", "stickyweb", "toxicspikes", "ceaselessedge"}
PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport"}
SETUP_MOVES = {
    "swordsdance",
    "dragondance",
    "nastyplot",
    "quiverdance",
    "bulkup",
    "calmmind",
    "geomancy",
    "shellsmash",
}


class UsageScout:
    """Generates Bayesian-style previews from ladder usage statistics."""

    def _raw_info(self, pokemon) -> dict:
        normalized = normalize_name(getattr(pokemon, "name", ""))
        base_name = normalize_name(getattr(pokemon, "base_name", normalized))
        info = SmogonSets.get_raw_pkmn_sets_from_pkmn_name(normalized, base_name)
        return info or {}

    def _usage_count(self, pokemon) -> float:
        normalized = normalize_name(getattr(pokemon, "name", ""))
        counts = SmogonSets.all_pkmn_counts.get(normalized, {})
        return float(counts.get(RAW_COUNT, 1)) or 1.0

    def predict_lead_probabilities(self, opponent_team: Iterable) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for pokemon in opponent_team:
            if not pokemon:
                continue
            info = self._raw_info(pokemon)
            moves = info.get(MOVES_STRING, [])
            hazard_weight = sum(weight for move, weight in moves if move in HAZARD_MOVES)
            pivot_weight = sum(weight for move, weight in moves if move in PIVOT_MOVES)
            raw_usage = self._usage_count(pokemon)
            speed_bonus = getattr(pokemon, "speed", 0) / 200.0
            score = (math.log(raw_usage + 1) + 1.0) * (
                1 + hazard_weight * 1.8 + pivot_weight * 1.2 + speed_bonus
            )
            scores[pokemon.name] = max(score, 0.05)

        total = sum(scores.values())
        if not scores or not total:
            team = [p for p in opponent_team if p]
            if not team:
                return {}
            uniform = 1.0 / len(team)
            return {p.name: uniform for p in team}
        return {name: score / total for name, score in scores.items()}

    def predict_tera_types(self, opponent_team: Iterable) -> Dict[str, List[tuple[str, float]]]:
        tera_predictions: Dict[str, List[tuple[str, float]]] = {}
        for pokemon in opponent_team:
            if not pokemon:
                continue
            info = self._raw_info(pokemon)
            tera_types = info.get(TERA_TYPE_STRING, [])
            if not tera_types:
                continue
            tera_predictions[pokemon.name] = [
                (tera_type, round(probability, 3)) for tera_type, probability in tera_types[:3]
            ]
        return tera_predictions

    def identify_win_conditions(self, opponent_team: Iterable) -> Dict[str, float]:
        threat_scores: Dict[str, float] = {}
        for pokemon in opponent_team:
            if not pokemon:
                continue
            info = self._raw_info(pokemon)
            moves = info.get(MOVES_STRING, [])
            move_weights = {move: weight for move, weight in moves}
            setup_pressure = sum(
                move_weights.get(move, 0.0) for move in SETUP_MOVES
            )
            hazard_pressure = sum(
                move_weights.get(move, 0.0) for move in HAZARD_MOVES
            )
            usage = math.log(self._usage_count(pokemon) + 1) / 10.0
            score = setup_pressure * 1.5 + hazard_pressure * 0.5 + usage
            if score > 0.2:
                threat_scores[pokemon.name] = round(score, 3)
        return threat_scores

    def build_preview_report(self, battle: "Battle") -> Dict[str, Dict]:
        opponent_team = [pkmn for pkmn in battle.opponent.reserve if pkmn]
        report = {
            "leads": self.predict_lead_probabilities(opponent_team),
            "tera": self.predict_tera_types(opponent_team),
            "threats": self.identify_win_conditions(opponent_team),
        }
        return report
