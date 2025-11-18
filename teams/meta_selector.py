from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

TEAM_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "teams"))
STATE_PATH = Path("data/team_meta_state.json")


class MetaTeamSelector:
    """Learns which local teams perform best for each format."""

    def __init__(self) -> None:
        self.state = {"teams": {}}
        if STATE_PATH.exists():
            try:
                self.state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.state = {"teams": {}}

    def _save(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STATE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2)

    def _relative_key(self, file_path: Path) -> str:
        try:
            return str(file_path.relative_to(TEAM_DIR))
        except ValueError:
            return file_path.name

    def _list_team_files(self, directory: Path) -> List[Path]:
        if not directory.exists():
            return []
        files = []
        for entry in directory.iterdir():
            if entry.is_file() and not entry.name.startswith("."):
                files.append(entry)
        return files

    def _score_team(self, rel_key: str) -> float:
        stats = self.state["teams"].get(rel_key, {})
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        total = wins + losses
        if total == 0:
            return 0.5
        win_rate = wins / total
        # Light penalty to encourage exploration if sample size is small
        confidence = min(1.0, total / 20.0)
        return (0.5 * (1 - confidence)) + (win_rate * confidence)

    def _choose_best_from_directory(self, directory: Path) -> str:
        files = self._list_team_files(directory)
        if not files:
            raise ValueError(f"No teams available in {directory}")
        scored = []
        for file_path in files:
            rel = self._relative_key(file_path)
            scored.append((self._score_team(rel), file_path))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_score = scored[0][0]
        top_candidates = [path for score, path in scored if score >= top_score - 0.05]
        return str(random.choice(top_candidates))

    def resolve_team_path(self, name: Optional[str], pokemon_format: Optional[str]) -> str:
        if name and name.lower() not in ("auto", "meta"):
            explicit_path = TEAM_DIR / name
            if explicit_path.is_dir():
                return self._choose_best_from_directory(explicit_path)
            if explicit_path.is_file():
                return str(explicit_path)
            raise ValueError(f"Unknown team path: {name}")

        search_dir = TEAM_DIR / (pokemon_format or "")
        if not search_dir.exists():
            search_dir = TEAM_DIR
        return self._choose_best_from_directory(search_dir)

    def record_result(self, team_identifier: str, pokemon_format: str, won: bool) -> None:
        if not team_identifier:
            return
        stats = self.state["teams"].setdefault(team_identifier, {"wins": 0, "losses": 0, "formats": {}})
        if won:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        format_stats = stats["formats"].setdefault(pokemon_format, {"wins": 0, "losses": 0})
        if won:
            format_stats["wins"] += 1
        else:
            format_stats["losses"] += 1
        self._save()
