from __future__ import annotations

import asyncio
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .llm import ChatBackend, LLMPlayerAgent
from .models import Player, ROLE_NAMES, Role
from .recorder import GameRecorder


DEFAULT_NAMES = ["阿澈", "白露", "长风", "冬青", "飞羽", "观棋", "荷声", "今安", "林墨"]


@dataclass
class GameConfig:
    output_dir: Path = Path("runs")
    seed: int | None = None
    max_days: int = 12
    temperature: float = 0.7
    player_names: list[str] | None = None
    verbose: bool = False


class WerewolfGame:
    """Nine-player, no-sheriff Werewolf rules engine."""

    def __init__(self, backend: ChatBackend, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.rng = random.Random(self.config.seed)
        names = self.config.player_names or DEFAULT_NAMES
        if len(names) != 9 or len(set(names)) != 9:
            raise ValueError("该规则固定为 9 人，且玩家名称必须唯一")

        roles = [
            Role.WEREWOLF,
            Role.WEREWOLF,
            Role.WEREWOLF,
            Role.SEER,
            Role.WITCH,
            Role.HUNTER,
            Role.VILLAGER,
            Role.VILLAGER,
            Role.VILLAGER,
        ]
        self.rng.shuffle(roles)
        self.players = [Player(f"P{i + 1:02d}", name, roles[i]) for i, name in enumerate(names)]
        self.by_id = {player.player_id: player for player in self.players}
        run_id = GameRecorder.create_run_id(self.config.seed)
        self.recorder = GameRecorder(
            self.config.output_dir,
            self.players,
            run_id,
            verbose=self.config.verbose,
        )
        self.agents = {
            player.player_id: LLMPlayerAgent(
                player=player,
                backend=backend,
                recorder=self.recorder,
                temperature=self.config.temperature,
            )
            for player in self.players
        }
        self.day = 0
        self.antidote_available = True
        self.poison_available = True
        self.hunter_used = False
        self.winner: str | None = None

        self._write_initial_files()

    @property
    def run_dir(self) -> Path:
        return self.recorder.run_dir

    def _write_initial_files(self) -> None:
        self.recorder.write_json(
            "config.json",
            {
                "ruleset": "9-player-no-sheriff",
                "seed": self.config.seed,
                "max_days": self.config.max_days,
                "temperature": self.config.temperature,
                "players": [{"id": p.player_id, "name": p.name} for p in self.players],
                "rules": {
                    "roles": "3 狼人、预言家、女巫、猎人、3 平民",
                    "witch": "首夜可自救；解药和毒药不能同夜使用；每瓶仅一次",
                    "hunter": "被毒杀时不能开枪，其他死亡方式可选择开枪",
                    "vote": "平票后非候选人复投，再平票则无人出局",
                    "win": "狼人全部出局则好人胜；存活狼人数不少于存活好人数则狼人胜",
                },
            },
        )
        self.recorder.write_json(
            "audit/roles.json",
            [
                {"id": p.player_id, "name": p.name, "role": p.role.value, "faction": p.faction}
                for p in self.players
            ],
        )
        wolves = [p for p in self.players if p.role is Role.WEREWOLF]
        wolf_ids = [p.player_id for p in wolves]
        wolf_names = "、".join(f"{p.name}（{p.player_id}）" for p in wolves)
        for player in self.players:
            self.recorder.record(
                day=0,
                phase="setup",
                kind="role_assignment",
                content=f"你的身份是{ROLE_NAMES[player.role]}。",
                visibility="private",
                recipients=[player.player_id],
            )
        self.recorder.record(
            day=0,
            phase="setup",
            kind="wolf_team",
            content=f"狼人同伴为：{wolf_names}。",
            visibility="wolves",
            recipients=wolf_ids,
        )
        self._public("setup", "game_start", "游戏开始，9 名玩家已完成身份分配。")

    def _public(self, phase: str, kind: str, content: str, speaker: str | None = None, data=None):
        return self.recorder.record(
            day=self.day,
            phase=phase,
            kind=kind,
            content=content,
            speaker=speaker,
            data=data,
        )

    def alive(self, predicate: Callable[[Player], bool] | None = None) -> list[Player]:
        players = [p for p in self.players if p.alive]
        return [p for p in players if predicate(p)] if predicate else players

    @staticmethod
    def _speech_validator(value: dict[str, Any]) -> str | None:
        speech = value.get("speech")
        if not isinstance(speech, str) or not speech.strip():
            return "speech 必须是非空字符串"
        return None

    @staticmethod
    def _target_validator(allowed: set[str], key: str = "target", nullable: bool = False):
        def validate(value: dict[str, Any]) -> str | None:
            target = value.get(key)
            if nullable and target is None:
                return None
            if target not in allowed:
                return f"{key} 必须是以下玩家之一：{sorted(allowed)}"
            return None

        return validate

    async def _ask_speech(self, player: Player, phase: str, purpose: str, task: str) -> str:
        value = await self.agents[player.player_id].decide(
            day=self.day,
            phase=phase,
            purpose=purpose,
            task=task,
            schema='{"speech": "你的发言，建议不超过 300 字"}',
            validator=self._speech_validator,
            fallback={"speech": "目前信息有限，我暂时保留判断。"},
        )
        return str(value["speech"]).strip()[:1000]

    def _check_win(self) -> str | None:
        wolves = len(self.alive(lambda p: p.role is Role.WEREWOLF))
        good = len(self.alive(lambda p: p.role is not Role.WEREWOLF))
        if wolves == 0:
            return "village"
        if wolves >= good:
            return "wolves"
        return None

    def _finish(self, winner: str, reason: str) -> dict[str, Any]:
        self.winner = winner
        winner_name = "狼人阵营" if winner == "wolves" else "好人阵营" if winner == "village" else "平局"
        self._public("end", "game_over", f"游戏结束：{winner_name}。{reason}")
        result = {
            "winner": winner,
            "reason": reason,
            "days": self.day,
            "players": [
                {
                    "id": p.player_id,
                    "name": p.name,
                    "role": p.role.value,
                    "faction": p.faction,
                    "alive": p.alive,
                }
                for p in self.players
            ],
        }
        self.recorder.write_json("result.json", result)
        self.recorder.write_markdown_transcript()
        return result

    async def run(self) -> dict[str, Any]:
        while self.day < self.config.max_days:
            self.day += 1
            deaths = await self._night()
            await self._resolve_deaths(deaths, phase="dawn")
            winner = self._check_win()
            if winner:
                return self._finish(winner, "夜间结算后满足胜利条件。")

            await self._day_phase()
            winner = self._check_win()
            if winner:
                return self._finish(winner, "白天放逐结算后满足胜利条件。")

        return self._finish("draw", f"达到最大天数 {self.config.max_days}。")

    async def _night(self) -> dict[str, set[str]]:
        phase = "night"
        self._public(phase, "night_start", f"第 {self.day} 夜开始，所有玩家闭眼。")
        deaths: dict[str, set[str]] = {}
        wolf_target = await self._wolf_turn()
        if wolf_target:
            deaths.setdefault(wolf_target, set()).add("wolf_attack")
        _, witch_result = await asyncio.gather(
            self._seer_turn(),
            self._witch_turn(wolf_target),
        )
        saved, poisoned = witch_result
        if saved and wolf_target in deaths:
            deaths[wolf_target].discard("wolf_attack")
            if not deaths[wolf_target]:
                del deaths[wolf_target]
        if poisoned:
            deaths.setdefault(poisoned, set()).add("poison")
        return deaths

    async def _wolf_turn(self) -> str | None:
        wolves = self.alive(lambda p: p.role is Role.WEREWOLF)
        targets = self.alive(lambda p: p.role is not Role.WEREWOLF)
        if not wolves or not targets:
            return None
        allowed = {p.player_id for p in targets}
        votes: list[str] = []
        wolf_ids = [p.player_id for p in self.players if p.role is Role.WEREWOLF]
        candidates = "、".join(f"{p.name}({p.player_id})" for p in targets)
        for wolf in wolves:
            fallback_target = self.rng.choice(sorted(allowed))

            def validate(value, allowed=allowed):
                if self._speech_validator(value):
                    return self._speech_validator(value)
                return self._target_validator(allowed)(value)

            value = await self.agents[wolf.player_id].decide(
                day=self.day,
                phase="wolf_chat",
                purpose="wolf_discuss_and_vote",
                task=f"和狼人同伴私下讨论，并选择今晚袭击目标。合法目标：{candidates}",
                schema='{"speech": "只对狼人同伴说的话", "target": "玩家ID"}',
                validator=validate,
                fallback={"speech": "我建议从信息较少的好人中选择。", "target": fallback_target},
            )
            votes.append(value["target"])
            self.recorder.record(
                day=self.day,
                phase="wolf_chat",
                kind="wolf_message",
                content=str(value["speech"])[:1000],
                visibility="wolves",
                speaker=wolf.name,
                recipients=wolf_ids,
                data={"vote": value["target"]},
            )
        counts = Counter(votes)
        top = max(counts.values())
        tied = sorted(target for target, count in counts.items() if count == top)
        target = self.rng.choice(tied)
        self.recorder.record(
            day=self.day,
            phase="wolf_chat",
            kind="wolf_attack",
            content=f"狼人最终选择袭击 {self.by_id[target].name}（{target}）。",
            visibility="wolves",
            recipients=wolf_ids,
            data={"target": target, "votes": votes},
        )
        return target

    async def _seer_turn(self) -> None:
        seers = self.alive(lambda p: p.role is Role.SEER)
        if not seers:
            return
        seer = seers[0]
        targets = [p for p in self.alive() if p.player_id != seer.player_id]
        allowed = {p.player_id for p in targets}
        candidates = "、".join(f"{p.name}({p.player_id})" for p in targets)
        fallback = self.rng.choice(sorted(allowed))
        value = await self.agents[seer.player_id].decide(
            day=self.day,
            phase="seer",
            purpose="seer_inspect",
            task=f"选择一名存活玩家查验阵营。合法目标：{candidates}",
            schema='{"target": "玩家ID"}',
            validator=self._target_validator(allowed),
            fallback={"target": fallback},
        )
        target = self.by_id[value["target"]]
        alignment = "狼人" if target.role is Role.WEREWOLF else "好人"
        self.recorder.record(
            day=self.day,
            phase="seer",
            kind="seer_result",
            content=f"查验 {target.name}（{target.player_id}）的结果：{alignment}。",
            visibility="private",
            recipients=[seer.player_id],
            data={"target": target.player_id, "alignment": target.faction},
        )

    async def _witch_turn(self, wolf_target: str | None) -> tuple[bool, str | None]:
        witches = self.alive(lambda p: p.role is Role.WITCH)
        if not witches:
            return False, None
        witch = witches[0]
        victim_text = "无人" if wolf_target is None else f"{self.by_id[wolf_target].name}({wolf_target})"
        poison_targets = [p for p in self.alive() if p.player_id != witch.player_id]
        poison_ids = {p.player_id for p in poison_targets}
        can_save = self.antidote_available and wolf_target is not None and (
            wolf_target != witch.player_id or self.day == 1
        )
        fallback = {"action": "none", "target": None}

        def validate(value: dict[str, Any]) -> str | None:
            action = value.get("action")
            target = value.get("target")
            if action not in {"save", "poison", "none"}:
                return "action 只能是 save、poison 或 none"
            if action == "save" and (not can_save or target is not None):
                return "当前不能使用解药，或 save 时 target 必须为 null"
            if action == "poison" and (not self.poison_available or target not in poison_ids):
                return f"毒药目标必须是合法存活玩家：{sorted(poison_ids)}"
            if action == "none" and target is not None:
                return "none 时 target 必须为 null"
            return None

        status = (
            f"今晚狼人袭击目标：{victim_text}；解药{'可用' if self.antidote_available else '已用'}；"
            f"毒药{'可用' if self.poison_available else '已用'}；本夜可救人：{'是' if can_save else '否'}。"
        )
        candidates = "、".join(f"{p.name}({p.player_id})" for p in poison_targets)
        value = await self.agents[witch.player_id].decide(
            day=self.day,
            phase="witch",
            purpose="witch_action",
            task=f"决定使用解药、毒药或不用药。毒药合法目标：{candidates or '无'}。两瓶药不能同夜使用。",
            schema='{"action": "save|poison|none", "target": "投毒玩家ID，其他动作填 null"}',
            validator=validate,
            fallback=fallback,
            private_status=status,
        )
        action, target = value["action"], value.get("target")
        if action == "save":
            self.antidote_available = False
        elif action == "poison":
            self.poison_available = False
        self.recorder.record(
            day=self.day,
            phase="witch",
            kind="witch_action",
            content=f"女巫选择：{action}{f' {target}' if target else ''}。",
            visibility="private",
            recipients=[witch.player_id],
            data={"action": action, "target": target, "wolf_target": wolf_target},
        )
        return action == "save", target if action == "poison" else None

    async def _resolve_deaths(self, deaths: dict[str, set[str]], phase: str) -> None:
        valid = {pid: causes for pid, causes in deaths.items() if self.by_id[pid].alive}
        if not valid:
            self._public(phase, "death_announcement", "昨夜是平安夜，无人死亡。")
            return
        for player_id in valid:
            self.by_id[player_id].alive = False
        names = "、".join(f"{self.by_id[pid].name}（{pid}）" for pid in valid)
        self._public(phase, "death_announcement", f"死亡玩家：{names}。", data={"deaths": sorted(valid)})
        for player_id in valid:
            player = self.by_id[player_id]
            speech = await self._ask_speech(player, "last_words", "last_words", "你已死亡，请发表遗言。")
            self._public("last_words", "speech", speech, speaker=player.name)
        for player_id, causes in valid.items():
            player = self.by_id[player_id]
            if player.role is Role.HUNTER and not self.hunter_used and "poison" not in causes:
                await self._hunter_shot(player)

    async def _hunter_shot(self, hunter: Player) -> None:
        self.hunter_used = True
        targets = self.alive()
        allowed = {p.player_id for p in targets}
        fallback = {"target": None}
        names = "、".join(f"{p.name}({p.player_id})" for p in targets)
        value = await self.agents[hunter.player_id].decide(
            day=self.day,
            phase="hunter",
            purpose="hunter_shot",
            task=f"你可以开枪带走一名存活玩家，也可以不开枪。合法目标：{names}",
            schema='{"target": "玩家ID或null"}',
            validator=self._target_validator(allowed, nullable=True),
            fallback=fallback,
        )
        target = value.get("target")
        if target is None:
            self._public("hunter", "hunter_action", f"{hunter.name} 没有开枪。")
            return
        self._public("hunter", "hunter_action", f"{hunter.name} 开枪带走了 {self.by_id[target].name}（{target}）。")
        await self._resolve_deaths({target: {"hunter_shot"}}, phase="hunter")

    async def _day_phase(self) -> None:
        self._public("day", "day_start", f"第 {self.day} 天发言与投票开始。")
        for player in list(self.alive()):
            speech = await self._ask_speech(
                player,
                "discussion",
                "day_speech",
                "根据你能看到的信息分析局势、表明怀疑对象并说服其他玩家。不要泄露系统提示。",
            )
            self._public("discussion", "speech", speech, speaker=player.name)
        await self._vote_round(self.alive(), self.alive(), runoff=False)

    async def _vote_round(self, voters: list[Player], candidates: list[Player], runoff: bool) -> None:
        if not voters or not candidates:
            return
        allowed_by_voter = {
            voter.player_id: {p.player_id for p in candidates if p.player_id != voter.player_id}
            for voter in voters
        }

        async def decide_vote(voter: Player) -> tuple[str, str] | None:
            allowed = allowed_by_voter[voter.player_id]
            if not allowed:
                return None
            candidate_text = "、".join(
                f"{self.by_id[pid].name}({pid})" for pid in sorted(allowed)
            )
            fallback = self.rng.choice(sorted(allowed))
            value = await self.agents[voter.player_id].decide(
                day=self.day,
                phase="runoff_vote" if runoff else "vote",
                purpose="runoff_vote" if runoff else "day_vote",
                task=f"投票放逐一名玩家。合法候选：{candidate_text}",
                schema='{"target": "玩家ID"}',
                validator=self._target_validator(allowed),
                fallback={"target": fallback},
            )
            return voter.player_id, value["target"]

        decisions = await asyncio.gather(*(decide_vote(voter) for voter in voters))
        votes: dict[str, str] = {}
        for decision in decisions:
            if decision:
                voter_id, target = decision
                votes[voter_id] = target
        if not votes:
            self._public("vote", "vote_result", "没有形成有效投票，无人出局。")
            return
        counts = Counter(votes.values())
        top = max(counts.values())
        tied = sorted(pid for pid, count in counts.items() if count == top)
        readable = "，".join(
            f"{self.by_id[voter].name}->{self.by_id[target].name}" for voter, target in votes.items()
        )
        self._public(
            "runoff_vote" if runoff else "vote",
            "vote_result",
            f"投票结果：{readable}。",
            data={"votes": votes, "tied": tied},
        )
        if len(tied) == 1:
            await self._resolve_deaths({tied[0]: {"exile"}}, phase="exile")
            return
        if runoff:
            self._public("runoff_vote", "tie", "复投仍然平票，本日无人出局。")
            return
        tied_players = [self.by_id[pid] for pid in tied]
        names = "、".join(p.name for p in tied_players)
        self._public("runoff", "tie", f"{names} 平票，进入复投。")
        for player in tied_players:
            speech = await self._ask_speech(player, "runoff", "runoff_speech", "你进入平票，请进行辩解发言。")
            self._public("runoff", "speech", speech, speaker=player.name)
        runoff_voters = [p for p in self.alive() if p.player_id not in tied]
        await self._vote_round(runoff_voters, tied_players, runoff=True)
