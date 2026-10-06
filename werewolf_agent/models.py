from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    VILLAGER = "villager"

    @property
    def faction(self) -> str:
        return "wolves" if self is Role.WEREWOLF else "village"


ROLE_NAMES = {
    Role.WEREWOLF: "狼人",
    Role.SEER: "预言家",
    Role.WITCH: "女巫",
    Role.HUNTER: "猎人",
    Role.VILLAGER: "平民",
}


@dataclass
class Player:
    player_id: str
    name: str
    role: Role
    alive: bool = True

    @property
    def faction(self) -> str:
        return self.role.faction


@dataclass
class Event:
    seq: int
    day: int
    phase: str
    kind: str
    content: str
    visibility: str = "public"
    speaker: str | None = None
    recipients: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "day": self.day,
            "phase": self.phase,
            "kind": self.kind,
            "speaker": self.speaker,
            "content": self.content,
            "visibility": self.visibility,
            "recipients": self.recipients,
            "data": self.data,
        }
