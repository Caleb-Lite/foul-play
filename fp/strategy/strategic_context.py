from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .opponent_model import OpponentModel
from .risk import RiskRewardAnalyzer
from .wincon import WinConditionTracker
from .experience import ExperienceTracker
from .time_manager import TimeManager


@dataclass
class StrategicContext:
    """Container that keeps long-lived strategic models for a battle."""

    opponent_model: OpponentModel = field(default_factory=OpponentModel)
    risk_analyzer: RiskRewardAnalyzer = field(default_factory=RiskRewardAnalyzer)
    wincon_tracker: WinConditionTracker = field(default_factory=WinConditionTracker)
    experience_tracker: ExperienceTracker = field(default_factory=ExperienceTracker)
    time_manager: TimeManager = field(default_factory=TimeManager)
    last_position_metrics: Optional[Dict] = None

    def reset_for_new_battle(self) -> None:
        self.opponent_model.reset()
        self.risk_analyzer.reset()
        self.wincon_tracker.reset()
        self.experience_tracker.reset()
        self.time_manager.reset()
        self.last_position_metrics = None

    def update_position_metrics(self, metrics: Dict) -> None:
        self.last_position_metrics = metrics
        self.wincon_tracker.update_from_metrics(metrics)
        self.time_manager.update_from_metrics(metrics)
