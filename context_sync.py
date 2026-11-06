"""Incremental shared context with recoverable, idempotent history writes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path


class ContextSync:
    """Keep pending voice history separate from the live text delivery queue.

    Args:
        umo: Bound AstrBot message origin.
        conversation_id: Conversation that owns every pending voice record.
        journal_path: Recovery file in the plugin data directory.
    """

    def __init__(self, umo: str, conversation_id: str, journal_path: Path) -> None:
        self.umo = umo
        self.cid = conversation_id
        self.path = journal_path
        self.pending: list[dict] = []
        self.memory_pending: list[dict] = []
        self.memory_batch: dict | None = None
        self.journal_lock = asyncio.Lock()
        self.text_queue: deque[tuple[str, str]] = deque()
        self.text_records: dict[str, dict] = {}
        self.text_tasks: dict[str, dict] = {}
        self.snapshot_ids: set[str] = set()
        self.seen: set[str] = set()
        self.seen_order: deque[str] = deque()
        self.flush_lock = asyncio.Lock()
        self.saved_count = 0
        self.text_ack_count = 0
        self.revision = 0
        self.input_revision = 0
        self.text_history_dirty = False
        if self.path.exists():
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if (document["umo"], document["cid"]) != (umo, conversation_id):
                raise ValueError(
                    "Recovery journal binding differs from this conversation."
                )
            self.pending = document["pending"]
            self.text_records = document.get("text_records", {})
            self.text_tasks = document.get("text_tasks", {})
            for task in self.text_tasks.values():
                if task["state"] == "processing":
                    task["state"] = "unknown_after_restart"
            self.text_history_dirty = document.get("text_history_dirty", False)
            self.memory_pending = document.get("memory_pending", list(self.pending))
            self.memory_batch = document.get("memory_batch")
            for entry in self.pending + self.memory_pending:
                self.seen.add(entry["source_id"])
                self.seen_order.append(entry["source_id"])

    def add_voice(
        self,
        live_id: str,
        event_id: str,
        role: str,
        text: str,
        start_ms: int,
        end_ms: int,
        playback: dict | None = None,
    ) -> bool:
        """Accept one transcript delta exactly once.

        Args:
            live_id: OpenAI Live session ID.
            event_id: Transcript event ID.
            role: User or assistant transcript direction.
            text: Exact transcript fragment.
            start_ms: Fragment start on the Live timeline.
            end_ms: Fragment end on the Live timeline.
            playback: Local audio progress, with end-user delivery remaining unconfirmed.

        Returns:
            Whether a new nonempty fragment was accepted.
        """
        source_id = f"{live_id}/{event_id}"
        if not text or source_id in self.seen:
            return False
        if role not in ("user", "assistant"):
            raise ValueError("Unsupported transcript role.")
        if max(len(self.pending), len(self.memory_pending)) >= 4096:
            raise OverflowError("Voice history is waiting for storage.")
        self.seen.add(source_id)
        self.seen_order.append(source_id)
        while len(self.seen_order) > 16384:
            self.seen.discard(self.seen_order.popleft())
        marker = hashlib.sha256(source_id.encode()).hexdigest()[:24]
        self.pending.append(
            {
                "source_id": source_id,
                "marker": f"[discord-live:{marker}]",
                "role": role,
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "playback": dict(playback or {}),
            }
        )
        self.memory_pending.append(dict(self.pending[-1]))
        self.revision += 1
        if role == "user":
            self.input_revision += 1
        return True

    def update_playback(self, live_id: str, snapshot: dict) -> None:
        """Refresh local playback evidence on unsaved output and memory records.

        Args:
            live_id: Session owning the audio stream.
            snapshot: Cumulative generated, dequeued and discarded local audio totals.
        """
        if not live_id:
            return
        playback = {
            key: value
            for key, value in snapshot.items()
            if key
            in {
                "output_generated_ms",
                "output_handed_to_discord_ms",
                "output_discarded_ms",
                "output_buffer_ms",
            }
        }
        changed = False
        for record in self.pending + self.memory_pending:
            if record["role"] == "assistant" and record["source_id"].startswith(
                live_id + "/"
            ):
                if record.get("playback") != playback:
                    record["playback"] = dict(playback)
                    changed = True
        if changed:
            self.revision += 1

    def pending_messages(self) -> list[dict]:
        """Return provider-compatible history with stable deduplication markers."""
        result = []
        for entry in self.pending:
            speaker = "Voice channel participant"
            if entry["role"] == "assistant":
                speaker = "Live generated speech (delivery unconfirmed)"
                if entry.get("playback"):
                    speaker += " local audio totals " + json.dumps(
                        entry["playback"], separators=(",", ":")
                    )
            result.append(
                {
                    "role": entry["role"],
                    "content": (
                        f"{entry['marker']} {speaker} "
                        f"({entry['start_ms']}-{entry['end_ms']} ms): {entry['text']}"
                    ),
                }
            )
        return result

    def add_text(self, message_id: str, author: str, text: str, answered=False) -> None:
        """Queue a new attributed Discord message.

        Args:
            message_id: Stable Discord message ID.
            author: Message author's display name.
            text: Current message content.
            answered: Whether this is a text response already delivered.
        """
        message_id = str(message_id)
        if message_id in self.text_records or not text:
            return
        record = {
            "message": message_id,
            "author": author[:40],
            "text": text[:20000],
            "answered": bool(answered),
            "revision": 0,
            "deleted": False,
        }
        if message_id not in self.snapshot_ids:
            self._queue_text(record)
        self.text_records[message_id] = record
        self.input_revision += 1
        while len(self.text_records) > 2048:
            del self.text_records[next(iter(self.text_records))]
        self.revision += 1

    def revise_text(
        self, message_id: str, text: str | None = None, *, deleted: bool = False
    ) -> bool:
        """Replace a message version and discard its queued outdated fragments.

        Args:
            message_id: Discord message being edited or deleted.
            text: Replacement content when available.
            deleted: Whether the message was deleted.

        Returns:
            Whether the effective version changed.
        """
        message_id = str(message_id)
        previous = self.text_records.get(message_id)
        if (
            previous
            and previous["deleted"] == deleted
            and (deleted or previous["text"] == text)
        ):
            return False
        record = dict(
            previous
            or {
                "message": message_id,
                "author": "Discord participant",
                "text": "",
                "answered": False,
                "revision": 0,
            }
        )
        record.update(
            text=(text or "")[:20000],
            deleted=deleted,
            revision=record["revision"] + 1,
        )
        previous_queue = self.text_queue
        self.text_queue = deque(
            packet
            for packet in previous_queue
            if json.loads(packet[1])["message"] != message_id
            or json.loads(packet[1]).get("origin") == "discord_text_task"
        )
        try:
            if not deleted and record["text"]:
                self._queue_text(record)
        except (ValueError, OverflowError):
            self.text_queue = previous_queue
            raise
        self.text_records[message_id] = record
        self.input_revision += 1
        while len(self.text_records) > 2048:
            del self.text_records[next(iter(self.text_records))]
        self.text_history_dirty = True
        self.revision += 1
        return True

    def set_text_task(
        self, message_id: str, state: str, attempt: str, *, replay: bool = False
    ) -> None:
        """Publish the latest text-agent attempt without duplicating its work.

        Args:
            message_id: Discord request whose text pipeline owns the task.
            state: Processing, completed, ended_without_result or unknown_after_restart.
            attempt: Stable identifier for this execution attempt.
            replay: Whether to resend an existing state to a replacement Live session.
        """
        if state not in {
            "processing",
            "completed",
            "ended_without_result",
            "unknown_after_restart",
        }:
            raise ValueError("Unknown text task state.")
        previous = self.text_tasks.get(message_id)
        if state != "processing" and (
            previous is None or previous["attempt"] != attempt
        ):
            return
        if (
            state == "ended_without_result"
            and not replay
            and previous["state"] != "processing"
        ):
            return
        changed = previous is None or (previous["state"], previous["attempt"]) != (
            state,
            attempt,
        )
        if not changed and not replay:
            return
        record = {
            "origin": "discord_text_task",
            "message": message_id,
            "state": state,
            "attempt": attempt,
            "source_revision": (
                self.text_records.get(message_id, {}).get("revision", 0)
                if state == "processing"
                and (previous is None or previous["attempt"] != attempt)
                else previous.get("source_revision", 0)
            ),
        }
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        if len(payload.encode("utf-8")) > 480:
            raise ValueError("Text task metadata exceeds the append budget.")
        queue = deque(
            packet
            for packet in self.text_queue
            if not (
                json.loads(packet[1]).get("origin") == "discord_text_task"
                and json.loads(packet[1])["message"] == message_id
            )
        )
        if len(queue) >= 1024:
            raise OverflowError("Text synchronization queue is full.")
        queue.append((f"task_{uuid.uuid4().hex}", payload))
        self.text_queue = queue
        self.text_tasks[message_id] = record
        while len(self.text_tasks) > 256:
            del self.text_tasks[next(iter(self.text_tasks))]
        if changed:
            self.revision += 1
            self.input_revision += 1

    def replay_text(self, exclude_ids: set[str] | None = None) -> None:
        """Queue a bounded recent window for a replacement Live session.

        Args:
            exclude_ids: Messages already included in the startup history.
        """
        self.text_queue.clear()
        self.snapshot_ids = set(exclude_ids or ())
        budget = 16000
        recent = []
        for record in reversed(self.text_records.values()):
            if record["deleted"] or record["message"] in (exclude_ids or set()):
                continue
            encoded = record["text"].encode("utf-8")
            content = encoded[-budget:].decode("utf-8", errors="ignore")
            recent.append({**record, "text": content})
            budget -= len(content.encode("utf-8"))
            if budget <= 0 or len(recent) >= 50:
                break
        for record in reversed(recent):
            self._queue_text(record)
        for task in list(self.text_tasks.values()):
            self.set_text_task(
                task["message"], task["state"], task["attempt"], replay=True
            )

    def project_history(
        self,
        history: list[dict],
        *,
        include_observed: bool = False,
        exclude_message_id: str | None = None,
    ) -> list[dict]:
        """Apply known Discord edits to tagged messages in a history view.

        Args:
            history: Original conversation history.
            include_observed: Include recent channel messages outside agent history.
            exclude_message_id: Current prompt, which the runner appends separately.

        Returns:
            A copied view with current text versions and deleted messages omitted.
        """
        result = []
        represented = set()
        for message in history:
            content = message.get("content", "")
            if isinstance(content, list):
                text = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") in ("text", "input_text")
                )
            else:
                text = content if isinstance(content, str) else ""
            match = re.search(r"\[discord-text:(\d+)\]\s*$", text)
            if match:
                represented.add(match[1])
            record = self.text_records.get(match[1]) if match else None
            if record:
                if record["deleted"]:
                    continue
                message = {
                    **message,
                    "content": (
                        f"{record['author']}: {record['text']}\n"
                        f"[discord-text:{record['message']}]"
                    ),
                }
            result.append(message)
        if include_observed:
            recent = []
            budget = 16000
            for record in reversed(self.text_records.values()):
                if (
                    record["deleted"]
                    or record["message"] in represented
                    or record["message"] == exclude_message_id
                ):
                    continue
                content = (
                    record["text"]
                    .encode("utf-8")[-budget:]
                    .decode("utf-8", errors="ignore")
                )
                recent.append(
                    {
                        "role": "assistant" if record["answered"] else "user",
                        "content": (
                            f"{record['author']}: {content}\n"
                            f"[discord-text:{record['message']}]"
                        ),
                    }
                )
                budget -= len(content.encode("utf-8"))
                if budget <= 0 or len(recent) >= 50:
                    break
            result.extend(reversed(recent))
        return result

    def _queue_text(self, record: dict) -> None:
        """Build byte-bounded append events shared by add, edit and replay paths."""
        packets = []
        remaining = record["text"]
        part = 0
        while remaining:
            count = min(250, len(remaining))
            while count:
                payload = json.dumps(
                    {
                        "origin": "discord_text",
                        "message": record["message"][:60],
                        "author": record["author"],
                        "already_answered": record["answered"],
                        "revision": record["revision"],
                        "part": part,
                        "content": remaining[:count],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if len(payload.encode("utf-8")) <= 480:
                    break
                count -= 1
            if not count:
                raise ValueError("Text metadata exceeds the append budget.")
            packets.append((f"text_{uuid.uuid4().hex}", payload))
            remaining = remaining[count:]
            part += 1
        if len(self.text_queue) + len(packets) > 1024:
            raise OverflowError("Text synchronization queue is full.")
        self.text_queue.extend(packets)

    def next_text_chunk(self) -> tuple[str, str] | None:
        """Return the first chunk until the server confirms its delivery."""
        return self.text_queue[0] if self.text_queue else None

    def ack_text(self, event_id: str) -> None:
        """Advance the text watermark after a matching acknowledgment."""
        if self.text_queue and self.text_queue[0][0] == event_id:
            self.text_queue.popleft()
            self.text_ack_count += 1

    async def checkpoint(self) -> None:
        """Atomically persist pending records without blocking the event loop."""
        payload = json.dumps(
            {
                "umo": self.umo,
                "cid": self.cid,
                "pending": list(self.pending),
                "memory_pending": list(self.memory_pending),
                "memory_batch": self.memory_batch,
                "text_records": self.text_records,
                "text_tasks": self.text_tasks,
                "text_history_dirty": self.text_history_dirty,
            },
            ensure_ascii=False,
        )
        async with self.journal_lock:
            task = asyncio.create_task(asyncio.to_thread(self._write_journal, payload))
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise

    def _write_journal(self, payload: str) -> None:
        """Replace the recovery file atomically from a worker thread."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)

    async def flush(
        self,
        read_history: Callable[[], Awaitable[list[dict]]],
        save_history: Callable[[list[dict]], Awaitable[None]],
        session_lock,
    ) -> None:
        """Merge voice records while holding the original text session lock.

        Args:
            read_history: Read the latest bound conversation history.
            save_history: Persist the merged history.
            session_lock: AstrBot's lock for the bound message origin.
        """
        async with self.flush_lock:
            if not self.pending and not self.text_history_dirty:
                await self.checkpoint()
                return
            await self.checkpoint()
            async with session_lock:
                submitted_revision = self.revision
                records = list(self.pending)
                messages = self.pending_messages()
                history = self.project_history(await read_history())
                serialized = json.dumps(history, ensure_ascii=False)
                for record, message in zip(records, messages):
                    if record["marker"] not in serialized:
                        history.append(message)
                await save_history(history)
                submitted = {entry["source_id"] for entry in records}
                self.pending = [
                    entry
                    for entry in self.pending
                    if entry["source_id"] not in submitted
                ]
                self.saved_count += len(submitted)
                if self.revision == submitted_revision:
                    self.text_history_dirty = False
            await self.checkpoint()
