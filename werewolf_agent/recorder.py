from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .models import Event, Player


class GameRecorder:
    """Writes every event immediately while maintaining isolated player views."""

    def __init__(self, root: Path, players: Iterable[Player], run_id: str, verbose: bool = False):
        self.run_dir = root / run_id
        self.audit_dir = self.run_dir / "audit"
        self.players_dir = self.run_dir / "players"
        self.factions_dir = self.run_dir / "factions"
        self.calls_dir = self.audit_dir / "llm_calls"
        for path in (self.audit_dir, self.players_dir, self.factions_dir, self.calls_dir):
            path.mkdir(parents=True, exist_ok=True)

        self.players = {player.player_id: player for player in players}
        for player_id in self.players:
            (self.players_dir / player_id).mkdir(exist_ok=True)
        self.events: list[Event] = []
        self.call_seq = 0
        self.verbose = verbose

    @staticmethod
    def create_run_id(seed: int | None) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = f"seed{seed}" if seed is not None else "random"
        return f"game_{stamp}_{suffix}"

    def write_json(self, relative_path: str, value: Any) -> None:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_jsonl(self, path: Path, value: Any) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    def emit(self, message: str) -> None:
        line = message.replace("\r", " ").replace("\n", " ")
        with (self.run_dir / "console.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.verbose:
            print(line, flush=True)

    def record(
        self,
        *,
        day: int,
        phase: str,
        kind: str,
        content: str,
        visibility: str = "public",
        speaker: str | None = None,
        recipients: Iterable[str] = (),
        data: dict[str, Any] | None = None,
    ) -> Event:
        recipient_ids = list(dict.fromkeys(recipients))
        event = Event(
            seq=len(self.events) + 1,
            day=day,
            phase=phase,
            kind=kind,
            content=content,
            visibility=visibility,
            speaker=speaker,
            recipients=recipient_ids,
            data=data or {},
        )
        self.events.append(event)
        payload = event.as_dict()
        self._append_jsonl(self.audit_dir / "events.jsonl", payload)

        if visibility == "public":
            self._append_jsonl(self.run_dir / "public_events.jsonl", payload)
            targets = list(self.players)
        else:
            targets = recipient_ids

        for player_id in targets:
            self._append_jsonl(self.players_dir / player_id / "visible_events.jsonl", payload)
        if visibility == "wolves":
            self._append_jsonl(self.factions_dir / "wolves.jsonl", payload)
        scope = {"public": "公开", "private": "私密", "wolves": "狼人", "admin": "审计"}.get(visibility, visibility)
        if visibility != "public" and recipient_ids:
            scope = f"{scope}->{','.join(recipient_ids)}"
        actor = f" {speaker}" if speaker else " 系统"
        self.emit(f"[第{day}天][{phase}][{scope}]{actor}: {content}")
        return event

    def visible_events(self, player_id: str) -> list[dict[str, Any]]:
        return [
            event.as_dict()
            for event in self.events
            if event.visibility == "public" or player_id in event.recipients
        ]

    def record_llm_call(
        self,
        *,
        player_id: str,
        purpose: str,
        messages: list[dict[str, str]],
        response: str | None,
        error: str | None,
        attempt: int,
    ) -> None:
        self.call_seq += 1
        safe_purpose = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in purpose)
        filename = f"{self.call_seq:04d}_{player_id}_{safe_purpose}_try{attempt}.json"
        self.write_json(
            f"audit/llm_calls/{filename}",
            {
                "player_id": player_id,
                "purpose": purpose,
                "attempt": attempt,
                "messages": messages,
                "response": response,
                "error": error,
            },
        )

    def write_markdown_transcript(self) -> None:
        lines = ["# 狼人杀公共对局记录", ""]
        for event in self.events:
            if event.visibility != "public":
                continue
            speaker = f" **{event.speaker}**" if event.speaker else ""
            lines.append(f"- 第 {event.day} 天 / {event.phase}{speaker}: {event.content}")
        (self.run_dir / "public_transcript.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
