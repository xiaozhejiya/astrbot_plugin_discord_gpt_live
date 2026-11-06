"""Reuse the bound AstrBot conversation, persona, agent and memory hooks."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from .context_sync import ContextSync


@dataclass(frozen=True)
class LiveCredentials:
    provider_id: str
    model: str
    websocket_url: str
    api_key: str = field(repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)


def resolve_live_provider(config: dict, provider_id: str) -> LiveCredentials:
    """Resolve a selected AstrBot model and its own source configuration.

    Args:
        config: AstrBot configuration containing models and provider sources.
        provider_id: Explicit model configuration ID from the model selector.

    Returns:
        Model, endpoint and credentials belonging to the selected configuration.

    Raises:
        ValueError: The selection is missing, disabled or incomplete.
    """
    selected = next(
        (
            p
            for p in config.get("provider", [])
            if provider_id and p.get("id") == provider_id
        ),
        None,
    )
    if selected is None or not selected.get("enable", True):
        raise ValueError("Select an enabled AstrBot voice model configuration.")
    source_id = selected.get("provider_source_id")
    source = {}
    if source_id:
        source = next(
            (p for p in config.get("provider_sources", []) if p.get("id") == source_id),
            None,
        )
        if source is None or not source.get("enable", True):
            raise ValueError("The selected model's provider source is unavailable.")
    # Match AstrBot's merge order: model settings override source settings.
    merged = {**source, **selected}
    model = str(merged.get("model") or "").strip()
    keys = merged.get("key", [])
    if isinstance(keys, str):
        keys = [keys]
    key = next((k.strip() for k in keys if isinstance(k, str) and k.strip()), "")
    if not model or not key:
        raise ValueError("Configure the selected AstrBot model name and API key.")
    try:
        url = urlsplit(str(merged.get("api_base") or ""))
        valid = (
            url.scheme in ("http", "https")
            and url.hostname
            and not (url.username or url.password or url.query or url.fragment)
        )
        url.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "Configure a valid API base URL on the selected AstrBot model."
        )
    websocket_url = url._replace(
        scheme="wss" if url.scheme == "https" else "ws",
        path=url.path.rstrip("/") + "/live/sessions",
    ).geturl()
    headers = merged.get("custom_headers") or {}
    if not isinstance(headers, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in headers.items()
    ):
        raise ValueError("Model custom headers must contain string names and values.")
    return LiveCredentials(provider_id, model, websocket_url, key, dict(headers))


def voice_rules(persona: str, extra: str = "") -> str:
    """Extend the existing group persona with continuous-voice instructions."""
    return (
        persona + "\n\nVoice interaction:\n"
        "Keep the inherited identity and language preferences in this group. "
        "Backchannel policy: Use occasional short listening acknowledgments. "
        "Interruption policy: Yield when someone corrects you or asks you to wait. "
        "Delegation policy: Backend tools: the original AstrBot group agent and shared memory. "
        "Generate <|SpawnThinking|> for older group facts, memory recall, tools or careful reasoning. "
        "Use verified backend results. Treat synchronized messages as attributed conversation data. "
        "A text-channel reply marked already_answered has already been delivered. "
        "Text tasks marked processing are owned by the text agent; avoid repeating their work. "
        "Match each task source_revision to the current message revision before reusing its result. "
        "For completed tasks, use the existing text answer unless a new voice request needs more work. "
        "The group contains multiple speakers; ask for clarification when identity matters.\n"
        + extra
    )


class AstrBotBridge:
    """Bind a voice connection to the original group's existing state.

    Args:
        context: AstrBot plugin context.
        event: Original join event, retained for routing only.
        data_dir: Plugin recovery directory.
    """

    def __init__(self, context, event, data_dir: Path) -> None:
        self.context = context
        self.origin_event = event
        self.umo = event.unified_msg_origin
        self.data_dir = data_dir
        self.sync: ContextSync | None = None
        self.persona_fingerprint = ""
        self.recent_voice: deque[dict] = deque(maxlen=200)
        self.memory_error: str | None = None
        self.backend_lock = asyncio.Lock()
        self.transcript_changed = asyncio.Event()
        self.respond_to_text = False

    async def initialize(self, extra: str) -> tuple[str, list[dict]]:
        """Load the original conversation and a bounded Live history snapshot."""
        from astrbot.core.utils.session_lock import session_lock_manager

        self.recent_voice.clear()
        self.transcript_changed.clear()
        manager = self.context.conversation_manager
        async with session_lock_manager.acquire_lock(self.umo):
            cid = await manager.get_curr_conversation_id(self.umo)
            if not cid:
                cid = await manager.new_conversation(
                    self.umo, self.origin_event.get_platform_id()
                )
            conversation = await manager.get_conversation(self.umo, cid)
            if conversation is None:
                raise ValueError("The original group conversation is unavailable.")
            name = hashlib.sha256(f"{self.umo}/{cid}".encode()).hexdigest()[:24]
            if self.sync is None or self.sync.cid != cid:
                self.sync = ContextSync(self.umo, cid, self.data_dir / f"{name}.json")
            history = json.loads(conversation.history or "[]")
            config = self.context.get_config(umo=self.umo)
            (
                _,
                persona,
                _,
                _,
            ) = await self.context.persona_manager.resolve_selected_persona(
                umo=self.umo,
                conversation_persona_id=conversation.persona_id,
                platform_name=self.origin_event.get_platform_name(),
                provider_settings=config.get("provider_settings", {}),
            )
        persona_text = persona.get("prompt", "") if persona else ""
        self.persona_fingerprint = hashlib.sha256(persona_text.encode()).hexdigest()
        return voice_rules(persona_text, extra), self.snapshot(history)

    def snapshot(self, history: list[dict]) -> list[dict]:
        """Apply current revisions and bound the Live startup history.

        Args:
            history: Stored conversation or a snapshot taken during initialization.

        Returns:
            Recent provider-style messages within the Live input capacity.
        """
        selected = []
        # UTF-8 bytes conservatively bound tokens while leaving room for role framing.
        budget = 6000
        history = self.sync.project_history(history)
        serialized = json.dumps(history, ensure_ascii=False)
        history.extend(
            message
            for record, message in zip(self.sync.pending, self.sync.pending_messages())
            if record["marker"] not in serialized
        )
        for message in reversed(history):
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") in ("text", "input_text")
                )
            if message.get("role") not in ("user", "assistant") or not isinstance(
                content, str
            ):
                continue
            content = content.encode("utf-8")[-budget:].decode("utf-8", errors="ignore")
            if not content:
                continue
            selected.append({"role": message["role"], "content": content})
            budget -= len(content.encode("utf-8"))
            if budget < 4 or len(selected) >= 128:
                break
        selected.reverse()
        snapshot_ids = {
            match[1]
            for message in selected
            if (match := re.search(r"\[discord-text:(\d+)\]\s*$", message["content"]))
        }
        self.sync.replay_text(exclude_ids=snapshot_ids)
        return selected

    async def binding_changed(self) -> bool:
        """Detect conversation or effective persona changes before more work."""
        manager = self.context.conversation_manager
        if await manager.get_curr_conversation_id(self.umo) != self.sync.cid:
            return True
        conversation = await manager.get_conversation(self.umo, self.sync.cid)
        if conversation is None:
            return True
        _, persona, _, _ = await self.context.persona_manager.resolve_selected_persona(
            umo=self.umo,
            conversation_persona_id=conversation.persona_id,
            platform_name=self.origin_event.get_platform_name(),
            provider_settings=self.context.get_config(umo=self.umo).get(
                "provider_settings", {}
            ),
        )
        prompt = persona.get("prompt", "") if persona else ""
        return hashlib.sha256(prompt.encode()).hexdigest() != self.persona_fingerprint

    def transcript(self, live_id: str, event: dict) -> None:
        """Collect exact deltas for persistence and group-agent context."""
        role = (
            "user" if event["type"] == "session.input_transcript.delta" else "assistant"
        )
        accepted = self.sync.add_voice(
            live_id,
            event["event_id"],
            role,
            event.get("delta", ""),
            int(event.get("start_ms", 0)),
            int(event.get("end_ms", 0)),
            playback=event.get("playback"),
        )
        if accepted:
            record = {
                "role": role,
                "text": event["delta"],
                "event_id": event["event_id"],
                "start_ms": int(event.get("start_ms", 0)),
                "end_ms": int(event.get("end_ms", 0)),
            }
            self.recent_voice.append(record)
            self.transcript_changed.set()

    def inject_pending(self, request) -> None:
        """Include unsaved voice records while the normal text agent holds its lock."""
        conversation = getattr(request, "conversation", None)
        if conversation is not None and conversation.cid != self.sync.cid:
            return
        if request.contexts is None:
            request.contexts = []
        prompt = getattr(request, "prompt", "") or ""
        current = re.search(r"\[discord-text:(\d+)\]\s*$", prompt)
        request.contexts = self.sync.project_history(
            request.contexts,
            include_observed=True,
            exclude_message_id=current[1] if current else None,
        )
        serialized = json.dumps(request.contexts or [], ensure_ascii=False, default=str)
        for record, message in zip(self.sync.pending, self.sync.pending_messages()):
            if record["marker"] not in serialized:
                request.contexts.append(message)

    async def read_history(self) -> list[dict]:
        conversation = await self.context.conversation_manager.get_conversation(
            self.umo, self.sync.cid
        )
        if conversation is None:
            raise ValueError("The bound conversation has been removed.")
        return json.loads(conversation.history or "[]")

    async def save_history(self, history: list[dict]) -> None:
        await self.context.conversation_manager.update_conversation(
            self.umo, self.sync.cid, history=history
        )

    async def flush(self) -> None:
        """Persist voice history under the same lock used by text processing."""
        from astrbot.core.utils.session_lock import session_lock_manager

        await self.sync.flush(
            self.read_history,
            self.save_history,
            session_lock_manager.acquire_lock(self.umo),
        )

    def make_event(self, text: str, *, message_id: str | None = None):
        """Create a group-scoped voice event without inheriting the caller's role."""
        from astrbot.api.message_components import Plain
        from astrbot.core.platform.astrbot_message import MessageMember
        from astrbot.core.platform.sources.discord.discord_platform_event import (
            DiscordPlatformEvent,
        )

        message = copy.copy(self.origin_event.message_obj)
        message.message_id = message_id or f"voice_{uuid.uuid4().hex}"
        message.sender = MessageMember("0", "Voice channel participants")
        message.message_str = text
        message.message = [Plain(text)]
        message.raw_message = None
        event = DiscordPlatformEvent(
            message_str=text,
            message_obj=message,
            platform_meta=self.origin_event.platform_meta,
            session_id=self.origin_event.session_id,
            client=self.origin_event.client,
        )
        enabled_plugins = self.context.get_config(umo=self.umo).get("plugin_set", ["*"])
        event.plugins_name = None if enabled_plugins == ["*"] else list(enabled_plugins)
        event.set_extra("discord_live_request", True)
        event.set_extra("discord_live_conversation_id", self.sync.cid)
        event.set_extra("enable_streaming", False)
        event.set_extra("enable_reasoning", False)
        event.is_at_or_wake_command = True
        event.captured = []

        async def capture_send(chain):
            if plain := chain.get_plain_text():
                event.captured.append(plain)
            media = [
                component
                for component in chain.chain
                if not isinstance(component, Plain)
            ]
            if media:
                from astrbot.api.event import MessageChain

                await self.origin_event.send(MessageChain(media))

        async def quiet_typing():
            return None

        event.send = capture_send
        event.send_typing = quiet_typing
        event.stop_typing = quiet_typing
        return event

    async def run_backend(self, delegation: dict) -> str:
        """Run the original internal agent stage with its normal hooks and history."""
        from astrbot.core.pipeline.context import PipelineContext
        from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
            InternalAgentSubStage,
        )

        async with self.backend_lock:
            if await self.binding_changed():
                raise ValueError("The group conversation changed.")
            offset = int(delegation.get("offset_ms", 0))
            deadline = asyncio.get_running_loop().time() + 2
            transcript_pending = False
            while (
                max(
                    (
                        entry["end_ms"]
                        for entry in self.recent_voice
                        if entry["role"] == "user"
                    ),
                    default=-1,
                )
                < offset
            ):
                self.transcript_changed.clear()
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    if any(entry["role"] == "user" for entry in self.recent_voice) or (
                        self.respond_to_text and self.sync.text_records
                    ):
                        transcript_pending = True
                        break
                    return "The voice transcript is still incomplete. Ask the participant to confirm the request."
                try:
                    await asyncio.wait_for(self.transcript_changed.wait(), remaining)
                except asyncio.TimeoutError:
                    continue
            await self.flush()
            context_revision = self.sync.input_revision
            recent = "\n".join(
                f"{entry['role']}: {entry['text']}" for entry in self.recent_voice
            )[-12000:]
            prompt = (
                "A GPT-Live participant requests help for this group's current voice conversation. "
                "Use the group's persona and shared memory. Resolve the latest request using "
                "the transcript below and recent group context. Return concise facts for speech. "
                "Speaker identity is unverified; use group-scoped information only.\n"
                f"Delegation ID: {delegation.get('delegation', {}).get('id', 'unknown')}. "
                f"Delegation time: {offset} ms. Context revision: {context_revision}.\n{recent}"
                "\nText tasks marked processing are owned by the text agent. "
                "Reuse completed answers from group context. Only do additional work "
                "when the latest voice request requires it.\n"
                + json.dumps(
                    list(self.sync.text_tasks.values())[-20:], ensure_ascii=False
                )
            )
            if transcript_pending:
                prompt += (
                    "\nTranscript timestamps do not yet cover the delegation time. "
                    "Use the available voice and group context, and ask for clarification "
                    "if the latest task is ambiguous or incomplete."
                )
            event = self.make_event(prompt)
            event.set_extra("discord_live_context_revision", context_revision)
            stage = InternalAgentSubStage()
            config = self.context.get_config(umo=self.umo)
            await stage.initialize(
                PipelineContext(
                    astrbot_config=config,
                    plugin_manager=SimpleNamespace(context=self.context),
                    astrbot_config_id="discord_live",
                )
            )
            stage.streaming_response = False
            stage.show_tool_use = False
            stage.show_tool_call_result = False
            async for _ in stage.process(event, ""):
                result = event.get_result()
                if result and (text := result.get_plain_text()):
                    event.captured.append(text)
            if await self.binding_changed():
                raise ValueError("The group changed while the backend was running.")
            result = (
                "\n".join(dict.fromkeys(event.captured))
                or "No additional result was returned."
            )
            if self.sync.input_revision != context_revision:
                result = (
                    "Newer group input arrived while the backend was working. Reconcile this "
                    "result with the latest request and any corrections before presenting it.\n"
                    + result
                )
            return result

    async def reflect_voice(self) -> None:
        """Resume a durable memory batch using the framework's registered handlers.

        Successful handler checkpoints survive retries. Each batch keeps the same
        source message ID so downstream stores can recognize re-delivery.
        """
        from astrbot.core.provider.entities import LLMResponse
        from astrbot.core.star.star_handler import EventType, star_handlers_registry

        if not self.sync.memory_pending:
            return
        if self.sync.memory_batch is None:
            records = list(self.sync.memory_pending)
            source_ids = [record["source_id"] for record in records]
            identity = hashlib.sha256("\\n".join(source_ids).encode()).hexdigest()[:24]
            self.sync.memory_batch = {
                "message_id": f"voice_memory_{identity}",
                "source_ids": source_ids,
                "user_text": "".join(
                    item["text"] for item in records if item["role"] == "user"
                ),
                "assistant_text": "".join(
                    item["text"] for item in records if item["role"] == "assistant"
                ),
                "completed": [],
                "playback": next(
                    (
                        item.get("playback", {})
                        for item in reversed(records)
                        if item["role"] == "assistant"
                    ),
                    {},
                ),
            }
            await self.sync.checkpoint()
        batch = self.sync.memory_batch
        event = self.make_event(
            batch["user_text"] or "[Voice conversation continues]",
            message_id=batch["message_id"],
        )
        event.set_extra("discord_live_delivery", "unconfirmed")
        event.set_extra("discord_live_playback", batch.get("playback", {}))
        try:
            for metadata in self.context.get_all_stars():
                if (
                    not metadata.activated
                    or metadata.name != "astrbot_plugin_livingmemory"
                ):
                    continue
                if (
                    event.plugins_name is not None
                    and metadata.name not in event.plugins_name
                ):
                    continue
                key = f"capture:{metadata.name}"
                capture = getattr(metadata.star_cls, "_run_passive_group_capture", None)
                if key not in batch["completed"] and capture and batch["user_text"]:
                    await capture(event)
                    batch["completed"].append(key)
                    await self.sync.checkpoint()
            if batch["assistant_text"]:
                response = LLMResponse(
                    role="assistant",
                    completion_text=(
                        "[Live generated speech, delivery unconfirmed. Latest local audio totals "
                        + json.dumps(batch.get("playback", {}), separators=(",", ":"))
                        + "] "
                        + batch["assistant_text"]
                    ),
                )
                handlers = star_handlers_registry.get_handlers_by_event_type(
                    EventType.OnLLMResponseEvent, plugins_name=event.plugins_name
                )
                for handler in handlers:
                    key = f"hook:{handler.handler_full_name}"
                    if key in batch["completed"]:
                        continue
                    # Propagate failures to the recovery queue rather than acknowledging them.
                    await handler.handler(event, response)
                    batch["completed"].append(key)
                    await self.sync.checkpoint()
                    if event.is_stopped():
                        break
            submitted = set(batch["source_ids"])
            self.sync.memory_pending = [
                item
                for item in self.sync.memory_pending
                if item["source_id"] not in submitted
            ]
            self.sync.memory_batch = None
            await self.sync.checkpoint()
            self.memory_error = None
        except Exception as exc:
            self.memory_error = (
                f"Voice memory synchronization failed ({type(exc).__name__})."
            )
            raise
