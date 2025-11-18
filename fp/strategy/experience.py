from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fp.battle import Battle
    from fp.strategy.risk import RiskRewardAnalyzer


EXPERIENCE_LOG_PATH = Path("data/experience_log.jsonl")


@dataclass
class ExperienceTracker:
    """Stores self-play experience so models can be trained offline."""

    log_path: Path = EXPERIENCE_LOG_PATH

    def reset(self) -> None:
        # No in-memory state currently, but keep for compatibility.
        pass

    def record_turn(
        self,
        battle: "Battle",
        selected_move: str,
        position_metrics: Dict,
        policy: Dict[str, float],
        risk_analyzer: Optional["RiskRewardAnalyzer"] = None,
    ) -> None:
        try:
            payload = {
                "battle_tag": getattr(battle, "battle_tag", ""),
                "turn": battle.turn,
                "selected_move": selected_move,
                "position": position_metrics,
                "policy": policy,
            }
            if risk_analyzer is not None and risk_analyzer.turn_cache:
                payload["risk"] = {
                    move: profile.__dict__
                    for move, profile in risk_analyzer.turn_cache.items()
                }

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except Exception:
            # Logging errors should never propagate to gameplay.
            pass
