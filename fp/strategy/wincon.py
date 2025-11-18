from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from data import all_move_json
from fp.helpers import type_effectiveness_modifier

if TYPE_CHECKING:
    from fp.battle import Battle, Pokemon


SETUP_MOVES = {
    "swordsdance",
    "dragondance",
    "nastyplot",
    "quiverdance",
    "calmmind",
    "bulkup",
    "curse",
    "irondefense",
    "shellsmash",
    "geomancy",
    "tidyup",
    "coil",
}

PRIORITY_MOVES = {"iceshard", "suckerpunch", "shadowsneak", "machpunch", "extremespeed"}
HAZARD_CONTROL_MOVES = {"rapidspin", "defog", "mortalspin", "ceaselessedge", "tidyup"}


@dataclass
class WinCondition:
    name: str
    hp_ratio: float
    role: str
    blockers: List[str] = field(default_factory=list)


class WinConditionTracker:
    """Identifies win conditions and critical preservation targets."""

    def __init__(self) -> None:
        self.primary: List[WinCondition] = []
        self.backup: List[WinCondition] = []
        self.do_not_sack: List[str] = []
        self.critical_targets: List[str] = []
        self.predicted_opponent_tera: Dict[str, List[tuple[str, float]]] = {}
        self.predicted_threats: Dict[str, float] = {}
        self.predicted_leads: Dict[str, float] = {}

    def reset(self) -> None:
        self.__init__()

    def _classify_role(self, pokemon: "Pokemon") -> str:
        if not pokemon:
            return "unknown"
        move_names = {mv.name for mv in pokemon.moves}
        if SETUP_MOVES & move_names:
            return "sweeper"
        if HAZARD_CONTROL_MOVES & move_names:
            return "utility"
        if pokemon.hp > pokemon.max_hp * 0.5 and any(
            mv.name in {"recover", "roost", "strengthsap", "morningsun"} for mv in pokemon.moves
        ):
            return "wall"
        return "pivot"

    def _evaluate_sweeper_blockers(self, sweeper: "Pokemon", opponents: List["Pokemon"]) -> List[str]:
        blockers: List[str] = []
        for opp in opponents:
            if not opp or opp.hp <= 0:
                continue
            for move in sweeper.moves:
                move_data = all_move_json.get(move.name, {})
                move_type = move_data.get("type", "normal")
                eff = type_effectiveness_modifier(move_type, opp.types)
                if eff < 1.0:
                    blockers.append(opp.name)
                    break
        return blockers

    def analyze_battle(self, battle: "Battle") -> Dict:
        self.primary = []
        self.backup = []
        self.do_not_sack = []
        self.critical_targets = []

        our_team = [battle.user.active] + battle.user.reserve if battle.user.active else battle.user.reserve
        opponents = [battle.opponent.active] + battle.opponent.reserve if battle.opponent.active else battle.opponent.reserve

        for pokemon in filter(None, our_team):
            hp_ratio = (pokemon.hp / pokemon.max_hp) if pokemon.max_hp else 0.0
            role = self._classify_role(pokemon)
            blockers = self._evaluate_sweeper_blockers(pokemon, opponents)
            wincon = WinCondition(
                name=pokemon.name,
                hp_ratio=hp_ratio,
                role=role,
                blockers=blockers,
            )
            if role == "sweeper" and hp_ratio > 0.4:
                self.primary.append(wincon)
            elif role in ("wall", "utility") and hp_ratio > 0.25:
                self.backup.append(wincon)

            if role in ("wall", "utility"):
                self.do_not_sack.append(pokemon.name)

        for opp in filter(None, opponents):
            if opp.hp <= 0:
                continue
            if opp.name in self.critical_targets:
                continue
            have_priority = any(mv.name in PRIORITY_MOVES for mv in opp.moves)
            if have_priority or opp.boosts.get("attack", 0) >= 2:
                self.critical_targets.append(opp.name)

        return {
            "primary": [wc.__dict__ for wc in self.primary],
            "backup": [wc.__dict__ for wc in self.backup],
            "preserve": self.do_not_sack,
            "targets": self.critical_targets,
            "predictions": {
                "opponent_tera": self.predicted_opponent_tera,
                "opponent_threats": self.predicted_threats,
                "opponent_leads": self.predicted_leads,
            },
        }

    def update_from_metrics(self, metrics: Optional[Dict]) -> None:
        # Metrics may come from position evaluation; this method keeps compatibility.
        if not metrics:
            return
        wincon_info = metrics.get("win_conditions")
        if not wincon_info:
            return
        self.do_not_sack = wincon_info.get("preserve", self.do_not_sack)

    def ingest_predictions(
        self,
        tera_predictions: Optional[Dict[str, List[tuple[str, float]]]],
        threat_scores: Optional[Dict[str, float]],
        lead_probabilities: Optional[Dict[str, float]] = None,
    ) -> None:
        if tera_predictions:
            self.predicted_opponent_tera = tera_predictions
        if threat_scores:
            self.predicted_threats = threat_scores
            for name, score in threat_scores.items():
                if score > 0.4 and name not in self.critical_targets:
                    self.critical_targets.append(name)
        if lead_probabilities:
            self.predicted_leads = lead_probabilities

    def evaluate(self, battle: "Battle") -> Dict:
        return self.analyze_battle(battle)
