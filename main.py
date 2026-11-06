"""AstrBot plugin for shared Discord group conversations through GPT-Live."""

from __future__ import annotations

import asyncio
import inspect
import re
import time
import traceback
import uuid
from dataclasses import dataclass, field
from functools import partial, update_wrapper
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.tool import ToolSet

from .astrbot_bridge import AstrBotBridge, resolve_live_provider
from .audio_bridge import DiscordAudioBridge
from .voice_session import LiveSession


@dataclass
class VoiceRuntime:
    platform_id: str
    guild_id: int
    text_channel_id: int
    channel: object
    event: object
    adapter: AstrBotBridge
    connection_task: asyncio.Task | None = None
    voice: object = None
    live: LiveSession | None = None
    audio: DiscordAudioBridge | None = None
    tasks: set = field(default_factory=set)
    early_text: list = field(default_factory=list)
    early_revisions: dict[str, tuple[str | None, bool]] = field(default_factory=dict)
    stop_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closing: bool = False
    restart: bool = False
    rebuilding: bool = False
    close_requested: bool = False
    error: str | None = None
    started: float = field(default_factory=time.monotonic)


class DiscordGPTLivePlugin(Star):
    """Use the existing Discord client and original group's shared context."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context, config)
        self.logger = getattr(self, "logger", logger)
        self.config = config
        self.sessions: dict[tuple[str, int], VoiceRuntime] = {}
        self.listeners: dict[str, tuple[object, list]] = {}
        self.cleanup_tasks: set[asyncio.Task] = set()
        self.join_lock = asyncio.Lock()
        self.last_errors: dict[tuple[str, int], str] = {}

    @filter.command("live_join")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def join(self, event: AstrMessageEvent):
        """Join the caller's voice channel and bind this text conversation."""
        if event.get_platform_name() != "discord":
            yield event.plain_result("此命令用于 Discord 服务器语音频道。")
            return
        if getattr(self, "terminating", False):
            yield event.plain_result("插件正在卸载，请稍后重试。")
            return
        raw = event.message_obj.raw_message
        guild = getattr(raw, "guild", None)
        if guild is None:
            yield event.plain_result("请在服务器的文字频道中使用此命令。")
            return
        member = guild.get_member(int(event.get_sender_id()))
        channel = member.voice.channel if member and member.voice else None
        if channel is None or channel.type.name != "voice":
            yield event.plain_result("请先进入一个普通语音频道。")
            return
        config = self.context.get_config(umo=event.unified_msg_origin)
        if config.get("platform_settings", {}).get("unique_session", False):
            yield event.plain_result(
                "该群聊启用了成员隔离，请先为语音绑定共享群聊会话。"
            )
            return
        key = (event.get_platform_id(), guild.id)
        async with self.join_lock:
            if key in self.sessions:
                yield event.plain_result(
                    "这个服务器已有语音会话，请使用 live_status 或 live_leave。"
                )
                return
            # Reserve by channel across adapters before Discord updates its member cache.
            occupant = next(
                (
                    active.event.client.user
                    for active in self.sessions.values()
                    if active.guild_id == guild.id and active.channel.id == channel.id
                ),
                None,
            )
            if occupant is None:
                occupant = next(
                    (
                        member
                        for member in channel.members
                        if member.bot and member.id != event.client.user.id
                    ),
                    None,
                )
            if occupant is not None:
                yield event.plain_result(
                    f"语音连接未启动：该频道已有机器人「{occupant.display_name}」或它正在加入，"
                    "不能再加入第二个机器人。请先让它退出，或选择其他语音频道。"
                )
                return
            adapter = AstrBotBridge(self.context, event, StarTools.get_data_dir())
            runtime = VoiceRuntime(
                key[0], guild.id, int(event.get_group_id()), channel, event, adapter
            )
            self.sessions[key] = runtime
        try:
            runtime.connection_task = asyncio.create_task(self.connect(runtime))
            await runtime.connection_task
        except asyncio.CancelledError:
            if not runtime.close_requested:
                await self.stop(runtime)
                raise
            return
        except Exception as exc:
            runtime.error = f"Connection failed ({type(exc).__name__})."
            await self.stop(runtime)
            locations = " > ".join(
                f"{Path(frame.filename).name}:{frame.lineno}"
                for frame in traceback.extract_tb(exc.__traceback__)
            )
            self.logger.error(
                "Discord Live startup failed: %s [%s]", type(exc).__name__, locations
            )
            yield event.plain_result(
                f"语音连接失败（{type(exc).__name__}）。请检查语音依赖、UDP 网络和 Live 配置。"
            )
            return
        yield event.plain_result(
            "已加入语音频道，并绑定当前群聊的人格与上下文。文字触发语音"
            + ("已开启。" if runtime.live.respond_to_text else "已关闭。")
        )

    async def connect(self, runtime: VoiceRuntime) -> None:
        """Attach listeners before loading the snapshot and opening voice."""
        from discord.voice import VoiceClient
        from discord.voice.receive.reader import PacketDecryptor

        # Stable Pycord 2.8.1 lacks the receive-side DAVE correction.
        if "dave.decrypt" not in inspect.getsource(PacketDecryptor.decrypt_rtp):
            raise RuntimeError("Install the documented Pycord DAVE receive revision.")
        client = runtime.event.client
        permissions = runtime.channel.permissions_for(runtime.channel.guild.me)
        if not (permissions.view_channel and permissions.connect and permissions.speak):
            raise PermissionError("Missing Discord voice channel permissions.")
        if runtime.channel.guild.voice_client:
            raise RuntimeError("The Discord client already owns a voice connection.")
        resolve_live_provider(
            self.context.get_config(), str(self.config.get("voice_provider_id", ""))
        )
        previous = self.listeners.get(runtime.platform_id)
        if previous is not None and previous[0] is not client:
            for callback, event_name in previous[1]:
                previous[0].remove_listener(callback, event_name)
            del self.listeners[runtime.platform_id]
        if runtime.platform_id not in self.listeners:
            listeners = [
                (partial(self.observe_message, runtime.platform_id), "on_message"),
                (
                    partial(self.observe_edit, runtime.platform_id),
                    "on_raw_message_edit",
                ),
                (
                    partial(self.observe_delete, runtime.platform_id),
                    "on_raw_message_delete",
                ),
                (
                    partial(self.observe_voice_state, runtime.platform_id),
                    "on_voice_state_update",
                ),
            ]
            self.listeners[runtime.platform_id] = (client, listeners)
            try:
                for callback, event_name in listeners:
                    # Pycord logs __name__ even when an explicit event name is supplied.
                    update_wrapper(callback, callback.func)
                    # Remove callbacks left by a failed registration in an unloaded instance.
                    for registered in tuple(
                        getattr(client, "_event_handlers", {}).get(event_name, ())
                    ):
                        owner = getattr(
                            getattr(registered, "func", None), "__self__", None
                        )
                        if type(owner).__module__ == __name__ and getattr(
                            owner, "terminating", False
                        ):
                            client.remove_listener(registered, event_name)
                    client.add_listener(callback, event_name)
            except Exception:
                for callback, event_name in listeners:
                    client.remove_listener(callback, event_name)
                self.listeners.pop(runtime.platform_id, None)
                raise
        runtime.voice = VoiceClient(client, runtime.channel)
        await runtime.channel.connect(
            cls=lambda client, channel: runtime.voice, timeout=30
        )
        await runtime.channel.guild.change_voice_state(
            channel=runtime.channel, self_deaf=False, self_mute=False
        )
        deadline = time.monotonic() + 15
        state = runtime.voice._connection
        while state.dave_protocol_version > 0 and (
            state.dave_session is None or not state.dave_session.ready
        ):
            if time.monotonic() >= deadline:
                raise TimeoutError("DAVE encryption keys did not become ready.")
            await asyncio.sleep(0.05)
        await self.start_streams(runtime)

    async def start_streams(self, runtime: VoiceRuntime) -> None:
        """Start a fresh Live stream using the existing Discord voice connection.

        Args:
            runtime: Current server connection and bound group adapter.
        """
        client = runtime.event.client
        credentials = resolve_live_provider(
            self.context.get_config(), str(self.config.get("voice_provider_id", ""))
        )
        instructions, history = await runtime.adapter.initialize(
            str(self.config.get("voice_style_extra", ""))
        )
        runtime.rebuilding = False
        for message in list(runtime.early_text):
            await self.observe_message(runtime.platform_id, message)
        runtime.early_text.clear()
        for message_id, (content, deleted) in runtime.early_revisions.items():
            runtime.adapter.sync.revise_text(message_id, content, deleted=deleted)
        runtime.early_revisions.clear()
        history = runtime.adapter.snapshot(history)
        runtime.restart = False
        if runtime.error:
            raise RuntimeError(runtime.error)
        runtime.audio = DiscordAudioBridge(
            int(client.user.id),
            is_user_allowed=lambda uid: (
                (member := runtime.channel.guild.get_member(uid)) is not None
                and not member.bot
            ),
        )
        runtime.voice.start_recording(runtime.audio.sink)
        runtime.voice.play(runtime.audio.source)
        runtime.adapter.respond_to_text = bool(
            self.config.get("respond_to_text_in_voice", False)
        )
        runtime.live = LiveSession(
            api_key=credentials.api_key,
            model=credentials.model,
            websocket_url=credentials.websocket_url,
            headers=credentials.headers,
            voice=str(self.config.get("voice", "marin")),
            instructions=instructions,
            history=history,
            bridge=runtime.audio,
            on_transcript=lambda payload: runtime.adapter.transcript(
                runtime.live.live_id, payload
            ),
            on_delegate=runtime.adapter.run_backend,
            respond_to_text=runtime.adapter.respond_to_text,
        )
        await runtime.live.start()
        for coroutine in (
            self.sync_text(runtime),
            self.sync_history(runtime),
            self.watch(runtime),
        ):
            runtime.tasks.add(asyncio.create_task(coroutine))

    async def observe_message(self, platform_id: str, message) -> None:
        """Capture raw text, including messages that do not wake the text bot."""
        if message.guild is None:
            return
        runtime = self.sessions.get((platform_id, message.guild.id))
        if (
            runtime is None
            or runtime.closing
            or message.channel.id != runtime.text_channel_id
        ):
            return
        if runtime.adapter.sync is None or getattr(runtime, "rebuilding", False):
            if len(runtime.early_text) >= 200:
                runtime.error = "Too many messages during initialization."
                return
            runtime.early_text.append(message)
            return
        text = message.content
        if not text and getattr(message, "attachments", None):
            text = "[Attachments shared in the text channel]"
        if not text:
            return
        # Discord replies may prefix status text with one or more mentions.
        status_text = re.sub(r"^(?:\s*<@!?[0-9]+>)*\s*", "", text)
        if message.author.bot and status_text.startswith(
            ("已加入语音频道", "语音连接", "语音会话", "Live 状态")
        ):
            return
        try:
            runtime.adapter.sync.add_text(
                str(message.id),
                message.author.display_name,
                text,
                answered=message.author.bot,
            )
        except (OverflowError, ValueError) as exc:
            runtime.error = f"Text synchronization failed ({type(exc).__name__})."

    async def observe_edit(self, platform_id: str, payload) -> None:
        """Apply content changes and request one replacement Live context.

        Args:
            platform_id: Discord adapter ID.
            payload: Discord raw message-update event.
        """
        if "content" not in payload.data:
            return
        for (platform, _), runtime in self.sessions.items():
            if (
                platform == platform_id
                and not runtime.closing
                and payload.channel_id == runtime.text_channel_id
            ):
                if runtime.adapter.sync is None or runtime.rebuilding:
                    if len(runtime.early_revisions) >= 200:
                        runtime.error = (
                            "Too many message revisions during initialization."
                        )
                        return
                    runtime.early_revisions[str(payload.message_id)] = (
                        payload.data["content"],
                        False,
                    )
                else:
                    runtime.restart |= runtime.adapter.sync.revise_text(
                        str(payload.message_id), payload.data["content"]
                    )

    async def observe_delete(self, platform_id: str, payload) -> None:
        """Invalidate a deleted message before replacing the Live context."""
        for (platform, _), runtime in self.sessions.items():
            if (
                platform == platform_id
                and not runtime.closing
                and payload.channel_id == runtime.text_channel_id
            ):
                if runtime.adapter.sync is None or runtime.rebuilding:
                    if len(runtime.early_revisions) >= 200:
                        runtime.error = (
                            "Too many message revisions during initialization."
                        )
                        return
                    runtime.early_revisions[str(payload.message_id)] = (None, True)
                else:
                    runtime.restart |= runtime.adapter.sync.revise_text(
                        str(payload.message_id), deleted=True
                    )

    async def observe_voice_state(
        self, platform_id: str, member, before, after
    ) -> None:
        """Release departed members' audio and RTP timing state.

        Args:
            platform_id: Discord adapter ID.
            member: Member whose voice state changed.
            before: Previous Discord voice state.
            after: Updated Discord voice state.
        """
        runtime = self.sessions.get((platform_id, member.guild.id))
        if (
            runtime is not None
            and runtime.audio is not None
            and before.channel is not None
            and before.channel.id == runtime.channel.id
            and (after.channel is None or after.channel.id != runtime.channel.id)
        ):
            runtime.audio.remove_user(member.id)

    @filter.on_llm_request(priority=10000)
    async def on_request(self, event, request) -> None:
        """Share pending voice context and reject work for a replaced conversation."""
        expected_cid = event.get_extra("discord_live_conversation_id")
        if event.get_extra("discord_live_request") and expected_cid:
            conversation = getattr(request, "conversation", None)
            if conversation is None or conversation.cid != expected_cid:
                event.stop_event()
                return
        for runtime in self.sessions.values():
            if (
                not runtime.closing
                and runtime.adapter.sync is not None
                and runtime.adapter.umo == event.unified_msg_origin
            ):
                prompt = getattr(request, "prompt", None)
                if not event.get_extra("discord_live_request") and isinstance(
                    prompt, str
                ):
                    source_id = str(event.message_obj.message_id)
                    if source_id.isdigit():
                        marker = f"[discord-text:{source_id}]"
                        if not prompt.rstrip().endswith(marker):
                            request.prompt = prompt + "\n" + marker
                        sync = runtime.adapter.sync
                        attempt = uuid.uuid4().hex
                        try:
                            sync.set_text_task(source_id, "processing", attempt)
                        except (ValueError, OverflowError) as exc:
                            runtime.error = f"Text task synchronization failed ({type(exc).__name__})."
                            continue
                        event.set_extra(
                            "discord_live_text_task", (sync, source_id, attempt)
                        )

                        def finished(
                            _,
                            sync=sync,
                            source_id=source_id,
                            attempt=attempt,
                            runtime=runtime,
                        ):
                            try:
                                sync.set_text_task(
                                    source_id, "ended_without_result", attempt
                                )
                            except (ValueError, OverflowError) as exc:
                                runtime.error = f"Text task synchronization failed ({type(exc).__name__})."

                        task = asyncio.current_task()
                        if task is not None:
                            task.add_done_callback(finished)
                runtime.adapter.inject_pending(request)

    @filter.on_llm_response(priority=-10000)
    async def on_response(self, event, response) -> None:
        """Publish text-agent completion independently of Discord delivery."""
        if event.get_extra("discord_live_request"):
            return
        task = event.get_extra("discord_live_text_task")
        if task is None:
            return
        sync, source_id, attempt = task
        state = "completed" if response is not None else "ended_without_result"
        try:
            sync.set_text_task(source_id, state, attempt)
        except (ValueError, OverflowError):
            self.logger.warning("Text task result remains unconfirmed in Live.")

    @filter.on_llm_request(priority=-10000)
    async def restrict_voice_tools(self, event, request) -> None:
        """Apply group-only tool restrictions after other request hooks."""
        if event.get_extra("discord_live_request") and request.func_tool:
            # Group-scoped recall is usable before precise speaker attribution.
            allowed = {
                "recall_long_term_memory",
                "get_group_message_history",
                "knowledge_base_query",
                "search_knowledge_base",
            }
            request.func_tool = ToolSet(
                [tool for tool in request.func_tool.tools if tool.name in allowed]
            )

    async def sync_text(self, runtime: VoiceRuntime) -> None:
        """Send background text with backpressure and acknowledged watermarks."""
        try:
            while not runtime.closing:
                desired = bool(self.config.get("respond_to_text_in_voice", False))
                if desired != runtime.live.respond_to_text:
                    await runtime.live.update_text_policy(desired)
                    runtime.adapter.respond_to_text = desired
                packet = runtime.adapter.sync.next_text_chunk()
                if packet:
                    event_id, text = packet
                    await runtime.live.append_context(text, event_id=event_id)
                    runtime.adapter.sync.ack_text(event_id)
                else:
                    await asyncio.sleep(0.15)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            runtime.error = f"Live text synchronization failed ({type(exc).__name__})."

    async def sync_history(self, runtime: VoiceRuntime) -> None:
        """Retry recoverable history and memory work independently of audio."""
        previous_error = None
        while not runtime.closing:
            await asyncio.sleep(2)
            try:
                runtime.adapter.sync.update_playback(
                    runtime.live.live_id, runtime.audio.stats
                )
                await runtime.adapter.flush()
                await runtime.adapter.reflect_voice()
                previous_error = None
                runtime.adapter.memory_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = f"History or memory remains pending ({type(exc).__name__})."
                runtime.adapter.memory_error = error
                if error != previous_error:
                    self.logger.warning("%s", error)
                previous_error = error

    async def watch(self, runtime: VoiceRuntime) -> None:
        """Supervise both audio transports, empty channels and binding changes."""
        empty_since = None
        last_binding_check = 0.0
        last_audio_report = time.monotonic()
        try:
            while not runtime.closing:
                now = time.monotonic()
                runtime.error = (
                    runtime.error or runtime.live.error or runtime.audio.error
                )
                if runtime.error or not runtime.voice.is_connected():
                    break
                if not runtime.voice.is_recording() or not runtime.voice.is_playing():
                    runtime.error = "Discord audio receiving or playback stopped."
                    break
                if not any(not member.bot for member in runtime.channel.members):
                    empty_since = empty_since or now
                    if now - empty_since >= 5:
                        break
                else:
                    empty_since = None
                if now - last_binding_check >= 2:
                    runtime.restart = (
                        runtime.restart or await runtime.adapter.binding_changed()
                    )
                    last_binding_check = now
                if runtime.restart:
                    break
                if now - last_audio_report >= 30:
                    self.logger.info(
                        "Discord Live audio metrics: %s", runtime.audio.stats
                    )
                    last_audio_report = now
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            runtime.error = f"Voice supervision failed ({type(exc).__name__})."
        if runtime.restart and not runtime.error and runtime.voice.is_connected():
            task = asyncio.create_task(self.rebuild(runtime))
            runtime.connection_task = task
        else:
            task = asyncio.create_task(self.stop(runtime))
        self.cleanup_tasks.add(task)
        task.add_done_callback(self.cleanup_tasks.discard)

    async def rebuild(self, runtime: VoiceRuntime) -> None:
        """Replace Live state while retaining the server's Discord connection.

        Args:
            runtime: Active connection whose text history or persona changed.
        """
        failed = False
        async with runtime.stop_lock:
            if runtime.closing or runtime.close_requested:
                return
            runtime.rebuilding = True
            runtime.restart = False
            workers = [
                task for task in runtime.tasks if task is not asyncio.current_task()
            ]
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            runtime.tasks.clear()
            try:
                if runtime.voice.is_recording():
                    runtime.voice.stop_recording()
                runtime.voice.stop()
                await runtime.live.close()
                runtime.audio.close()
                if stats := getattr(runtime.audio, "stats", None):
                    runtime.adapter.sync.update_playback(runtime.live.live_id, stats)
                try:
                    await asyncio.wait_for(runtime.adapter.flush(), 15)
                except (ValueError, asyncio.TimeoutError):
                    # A reset may remove the old conversation; retain its journal binding.
                    await runtime.adapter.sync.checkpoint()
                if not runtime.close_requested:
                    await self.start_streams(runtime)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                runtime.error = f"Live context rebuild failed ({type(exc).__name__})."
                failed = True
            finally:
                runtime.rebuilding = False
        if failed:
            await self.stop(runtime)

    async def stop(self, runtime: VoiceRuntime) -> None:
        """Release one connection and retain recoverable history on storage errors."""
        runtime.close_requested = True
        connection = runtime.connection_task
        if (
            connection
            and connection is not asyncio.current_task()
            and not connection.done()
        ):
            connection.cancel()
            await asyncio.gather(connection, return_exceptions=True)
        async with runtime.stop_lock:
            if runtime.closing:
                return
            runtime.closing = True
            tasks = [
                task for task in runtime.tasks if task is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                try:
                    if runtime.voice and runtime.voice.is_recording():
                        runtime.voice.stop_recording()
                except Exception as exc:
                    runtime.error = (
                        f"Discord recording cleanup failed ({type(exc).__name__})."
                    )
                    self.logger.warning("%s", runtime.error)
                if runtime.live:
                    await runtime.live.close()
                    runtime.error = runtime.error or getattr(
                        runtime.live, "error", None
                    )
            except Exception as exc:
                runtime.error = (
                    f"Voice transport cleanup failed ({type(exc).__name__})."
                )
                self.logger.warning("%s", runtime.error)
            finally:
                try:
                    if runtime.audio:
                        runtime.audio.close()
                    if runtime.voice:
                        try:
                            await runtime.voice.disconnect(force=True)
                        except Exception as exc:
                            runtime.error = f"Discord voice disconnect failed ({type(exc).__name__})."
                            self.logger.warning("%s", runtime.error)
                    if runtime.adapter.sync:
                        if runtime.live and (
                            stats := getattr(runtime.audio, "stats", None)
                        ):
                            runtime.adapter.sync.update_playback(
                                runtime.live.live_id, stats
                            )
                        try:
                            await asyncio.wait_for(runtime.adapter.flush(), 15)
                            await asyncio.wait_for(runtime.adapter.reflect_voice(), 15)
                        except Exception:
                            try:
                                await runtime.adapter.sync.checkpoint()
                                runtime.error = "Voice history or memory remains pending for recovery."
                            except Exception as exc:
                                runtime.error = f"Voice history and recovery save failed ({type(exc).__name__})."
                            self.logger.warning("%s", runtime.error)
                finally:
                    key = (runtime.platform_id, runtime.guild_id)
                    if runtime.error:
                        self.last_errors[key] = runtime.error
                        while len(self.last_errors) > 512:
                            del self.last_errors[next(iter(self.last_errors))]
                    if self.sessions.get(key) is runtime:
                        self.sessions.pop(key)

    @filter.command("live_leave")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def leave(self, event: AstrMessageEvent):
        """Close the current server's voice conversation."""
        guild = getattr(event.message_obj.raw_message, "guild", None)
        runtime = (
            self.sessions.get((event.get_platform_id(), guild.id)) if guild else None
        )
        if runtime:
            await self.stop(runtime)
        if runtime and runtime.error:
            yield event.plain_result(
                "语音会话已退出，存在待处理错误。请用 live_status 查看。"
            )
        else:
            yield event.plain_result(
                "语音会话已退出。" if runtime else "当前服务器没有语音会话。"
            )

    @filter.command("live_status")
    async def status(self, event: AstrMessageEvent):
        """Display connection and synchronization status without credentials."""
        guild = getattr(event.message_obj.raw_message, "guild", None)
        runtime = (
            self.sessions.get((event.get_platform_id(), guild.id)) if guild else None
        )
        if runtime is None:
            error = (
                self.last_errors.get((event.get_platform_id(), guild.id))
                if guild
                else None
            )
            message = "当前服务器没有语音会话。"
            if error:
                message += f"\n最近错误 {error}"
            yield event.plain_result(message)
            return
        sync = runtime.adapter.sync
        state = "运行中" if runtime.live and runtime.live.ready.is_set() else "连接中"
        if runtime.rebuilding:
            state = "更新上下文中"
        yield event.plain_result(
            f"Live 状态 {state}\n语音频道 {runtime.channel.name}\n"
            f"持续 {int(time.monotonic() - runtime.started)} 秒\n"
            f"待保存转录 {len(sync.pending) if sync else 0}\n"
            f"待同步文字 {len(sync.text_queue) if sync else 0}\n"
            f"待处理记忆 {len(sync.memory_pending) if sync else 0}\n"
            f"最近错误 {runtime.error or runtime.adapter.memory_error or '无'}"
        )

    async def terminate(self) -> None:
        """Unregister listeners and await all owned cleanup tasks."""
        self.terminating = True
        for runtime in list(self.sessions.values()):
            await self.stop(runtime)
        for client, listeners in self.listeners.values():
            for callback, event_name in listeners:
                client.remove_listener(callback, event_name)
        self.listeners.clear()
        if self.cleanup_tasks:
            await asyncio.gather(*self.cleanup_tasks, return_exceptions=True)
