"""OpenAI Live WebSocket lifecycle and continuous audio transport."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import uuid
from collections.abc import Awaitable, Callable

import aiohttp

TEXT_OFF = (
    "Text-channel messages are background context. Wait for a voice participant "
    "to ask about them before commenting aloud. Continue natural voice conversation."
)
TEXT_ON = (
    "You may naturally respond aloud to relevant new text-channel messages. "
    "Consider already_answered and in-progress work to avoid duplicate replies or tasks."
)


class LiveSession:
    """Own one continuous Live connection and its background tasks.

    Args:
        api_key: Credential from the selected AstrBot model configuration.
        websocket_url: Live endpoint derived from the configured API base URL.
        headers: Additional request headers from the model configuration.
        model: Voice model, independent of the group's reasoning model.
        voice: OpenAI voice name.
        instructions: Inherited persona and voice behavior.
        history: Relevant prior text messages.
        bridge: Continuous audio bridge.
        on_transcript: Synchronous transcript collector.
        on_delegate: Async callback returning a backend result.
        respond_to_text: Whether text activity may trigger speech.
    """

    def __init__(
        self,
        *,
        api_key: str,
        websocket_url: str,
        model: str,
        voice: str,
        instructions: str,
        history: list[dict],
        bridge,
        on_transcript: Callable[[dict], None],
        on_delegate: Callable[[dict], Awaitable[str]],
        respond_to_text=False,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.websocket_url = websocket_url
        self.headers = dict(headers or {})
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.history = history
        self.bridge = bridge
        self.on_transcript = on_transcript
        self.on_delegate = on_delegate
        self.respond_to_text = bool(respond_to_text)
        self.live_id = ""
        self.ready = asyncio.Event()
        self.terminal = asyncio.Event()
        self.closed = asyncio.Event()
        self.error: str | None = None
        self.usage: dict | None = None
        self.ws = None
        self.http = None
        self.tasks: set[asyncio.Task] = set()
        self.acks: dict[str, asyncio.Future] = {}
        self.delegations: set[str] = set()
        self.close_lock = asyncio.Lock()
        self.send_lock = asyncio.Lock()
        self.closing = False

    async def start(self) -> None:
        """Connect, await readiness, then start real-time audio transmission."""
        self.http = aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=None, sock_connect=20)
        )
        try:
            self.ws = await self.http.ws_connect(
                self.websocket_url,
                headers={**self.headers, "Authorization": f"Bearer {self.api_key}"},
                heartbeat=20,
                max_msg_size=8 * 1024 * 1024,
            )
            self.tasks.add(asyncio.create_task(self._receive()))
            await self.send(
                {
                    "type": "session.start",
                    "event_id": f"start_{uuid.uuid4().hex}",
                    "session": {
                        "model": self.model,
                        "instructions": self.instructions
                        + "\n"
                        + (TEXT_ON if self.respond_to_text else TEXT_OFF)
                        + "\nInitial history predates this voice connection and is "
                        "background context only. Do not read it aloud, summarize it, "
                        "or resume old requests on joining. Wait for fresh voice input "
                        "or an eligible new text message before responding. "
                        "Previously delivered assistant replies are not pending speech.",
                        "input": [
                            {
                                "role": message["role"],
                                "content": [
                                    {
                                        "type": (
                                            "output_text"
                                            if message["role"] == "assistant"
                                            else "input_text"
                                        ),
                                        "text": message["content"],
                                    }
                                ],
                            }
                            for message in self.history
                        ],
                        "audio": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "output": {"voice": self.voice},
                        },
                        "delegation": {"type": "client"},
                    },
                }
            )
            await asyncio.wait_for(self.ready.wait(), 30)
            if self.error:
                raise RuntimeError(self.error)
            self.tasks.add(asyncio.create_task(self._send_audio()))
        except BaseException:
            await self.close()
            raise

    async def send(self, event: dict) -> None:
        """Serialize outbound messages on the Live connection."""
        async with self.send_lock:
            if self.ws is None or self.ws.closed:
                raise ConnectionError("Live connection is closed.")
            await self.ws.send_json(event)

    async def _receive(self) -> None:
        """Route protocol events until transport or session termination."""
        try:
            async for message in self.ws:
                if message.type == aiohttp.WSMsgType.TEXT:
                    await self.handle_event(json.loads(message.data))
                elif message.type == aiohttp.WSMsgType.ERROR:
                    raise ConnectionError("Live WebSocket receive failed.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Avoid logging credentials or raw service responses.
            self.error = f"Live receive failed ({type(exc).__name__})."
        finally:
            if not self.closing and not self.terminal.is_set():
                self.error = self.error or "Live connection ended unexpectedly."
            self.ready.set()
            for future in self.acks.values():
                if not future.done():
                    future.set_exception(ConnectionError("Live connection ended."))

    async def handle_event(self, event: dict) -> None:
        """Dispatch one decoded event without blocking audio on backend work."""
        event_type = event.get("type")
        if event_type == "session.started":
            self.live_id = event["session"]["id"]
            self.ready.set()
        elif event_type == "session.output_audio.delta":
            try:
                pcm = base64.b64decode(event["delta"], validate=True)
            except binascii.Error as exc:
                raise ValueError("Invalid Live audio encoding.") from exc
            self.bridge.feed_output(pcm)
        elif event_type in (
            "session.input_transcript.delta",
            "session.output_transcript.delta",
        ):
            if event_type == "session.output_transcript.delta":
                event = {
                    **event,
                    "playback": {
                        key: value
                        for key, value in getattr(self.bridge, "stats", {}).items()
                        if key
                        in {
                            "output_generated_ms",
                            "output_handed_to_discord_ms",
                            "output_discarded_ms",
                            "output_buffer_ms",
                        }
                    },
                }
            self.on_transcript(event)
        elif event_type == "session.delegation.created":
            delegation_id = event["delegation"]["id"]
            if delegation_id not in self.delegations:
                if len(self.delegations) >= 4096:
                    raise OverflowError("Live delegation history limit reached.")
                self.delegations.add(delegation_id)
                task = asyncio.create_task(self._delegate(event))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
        elif event_type == "session.closed":
            self.usage = event.get("usage")
            self.terminal.set()
        elif event_type == "error":
            error = event.get("error", {})
            identifier = error.get("client_event_id") or event.get("client_event_id")
            code = str(error.get("code", "service_error"))[:80]
            future = self.acks.get(identifier)
            if future is not None and not future.done():
                future.set_exception(RuntimeError(f"Live command failed: {code}"))
            else:
                self.error = f"Live service error: {code}"
                self.ready.set()
        elif event_type in (
            "session.thinking.appended",
            "session.commentary.appended",
            "session.instructions.appended",
        ):
            future = self.acks.get(event.get("client_event_id"))
            if future is not None and not future.done():
                future.set_result(None)

    async def append_context(
        self,
        content: str,
        *,
        event_id: str | None = None,
        delegation_id: str | None = None,
        kind: str = "thinking",
    ) -> None:
        """Append bounded context and wait for server acknowledgment.

        Args:
            content: Context within the service's 500-token append limit.
            event_id: Stable ID for matching the acknowledgment.
            delegation_id: Associated backend delegation, if any.
            kind: Thinking, commentary or instructions event.
        """
        if kind not in ("thinking", "commentary", "instructions"):
            raise ValueError("Unsupported append kind.")
        identifier = event_id or f"append_{uuid.uuid4().hex}"
        if identifier in self.acks:
            raise ValueError("Append event ID is already in flight.")
        future = asyncio.get_running_loop().create_future()
        self.acks[identifier] = future
        try:
            await self.send(
                {
                    "type": f"session.{kind}.append",
                    "event_id": identifier,
                    "delegation_id": delegation_id,
                    "content": content,
                }
            )
            await asyncio.wait_for(future, 15)
        finally:
            self.acks.pop(identifier, None)

    async def update_text_policy(self, enabled: bool) -> None:
        """Apply a text-to-voice behavior change after its acknowledgment."""
        await self.append_context(TEXT_ON if enabled else TEXT_OFF, kind="instructions")
        self.respond_to_text = bool(enabled)

    async def _delegate(self, event: dict) -> None:
        """Run the group agent independently of audio receive and playback."""
        try:
            try:
                result = await asyncio.wait_for(self.on_delegate(event), 60)
            except asyncio.TimeoutError:
                result = "The group agent timed out. Explain the delay and ask whether to retry."
            except Exception:
                result = "The group agent could not complete this request. Explain the failure."
            remaining = result or "The group agent returned no additional information."
            while remaining and not self.closing:
                count = min(400, len(remaining))
                while len(remaining[:count].encode("utf-8")) > 480:
                    count -= 1
                await self.append_context(
                    remaining[:count],
                    kind="commentary",
                    delegation_id=event["delegation"]["id"],
                )
                remaining = remaining[count:]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = (
                f"Live delegation result delivery failed ({type(exc).__name__})."
            )

    async def _send_audio(self) -> None:
        """Pace audio at the sample rate and stop on excessive drift."""
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        try:
            while not self.closing:
                if self.bridge.error:
                    raise RuntimeError(self.bridge.error)
                if loop.time() - deadline > 1:
                    raise TimeoutError("Audio sending fell behind real time.")
                await self.send(
                    {
                        "type": "session.input_audio.append",
                        "audio": base64.b64encode(self.bridge.read_input()).decode(
                            "ascii"
                        ),
                    }
                )
                deadline += 0.02
                await asyncio.sleep(max(0, deadline - loop.time()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"Live audio sending failed ({type(exc).__name__})."

    async def close(self) -> None:
        """Close once, await final usage when possible, then release all tasks."""
        async with self.close_lock:
            if self.closed.is_set():
                return
            self.closing = True
            try:
                if self.ws and not self.ws.closed and self.live_id:
                    try:
                        await self.send({"type": "session.close"})
                        await asyncio.wait_for(self.terminal.wait(), 15)
                    except (ConnectionError, asyncio.TimeoutError):
                        self.error = self.error or "Live final usage was not confirmed."
            finally:
                for future in self.acks.values():
                    if not future.done():
                        future.set_exception(ConnectionError("Live session closed."))
                current = asyncio.current_task()
                tasks = [task for task in self.tasks if task is not current]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if self.ws and not self.ws.closed:
                    await self.ws.close()
                if self.http:
                    await self.http.close()
                self.bridge.close()
                self.api_key = ""
                self.closed.set()
