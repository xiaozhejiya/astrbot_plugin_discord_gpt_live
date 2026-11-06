"""Plugin entry-point behavior with controlled platform objects."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from discord_gpt_live.main import DiscordGPTLivePlugin


@pytest.mark.asyncio
async def test_text_observer_filters_channels_and_duplicate_messages():
    plugin = object.__new__(DiscordGPTLivePlugin)
    sync = SimpleNamespace(
        add_text=lambda *args, **kwargs: captured.append((args, kwargs))
    )
    runtime = SimpleNamespace(
        text_channel_id=10,
        guild_id=20,
        adapter=SimpleNamespace(sync=sync),
        early_text=[],
        closing=False,
        restart=False,
    )
    plugin.sessions = {("discord", 20): runtime}
    captured = []
    message = SimpleNamespace(
        guild=SimpleNamespace(id=20),
        channel=SimpleNamespace(id=11),
        id=30,
        author=SimpleNamespace(display_name="Alice", bot=False),
        content="Hello",
        attachments=[],
    )
    await plugin.observe_message("discord", message)
    assert captured == []
    message.channel.id = 10
    await plugin.observe_message("discord", message)
    assert captured[0][0][:3] == ("30", "Alice", "Hello")


@pytest.mark.asyncio
async def test_request_hook_keeps_normal_tools_and_injects_pending_voice():
    plugin = object.__new__(DiscordGPTLivePlugin)
    adapter = SimpleNamespace(
        umo="group",
        sync=object(),
        inject_pending=lambda req: req.contexts.append("voice"),
    )
    plugin.sessions = {("discord", 20): SimpleNamespace(adapter=adapter, closing=False)}
    event = SimpleNamespace(unified_msg_origin="group", get_extra=lambda key: None)
    request = SimpleNamespace(contexts=[], func_tool="original")
    await plugin.on_request(event, request)
    assert request.contexts == ["voice"]
    assert request.func_tool == "original"


@pytest.mark.asyncio
async def test_shutdown_is_idempotent():
    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.sessions = {}
    plugin.listeners = {}
    plugin.cleanup_tasks = set()
    await plugin.terminate()
    await plugin.terminate()


@pytest.mark.asyncio
async def test_stop_cancels_in_progress_connection():
    from discord_gpt_live.main import VoiceRuntime

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.sessions = {}
    plugin.logger = SimpleNamespace(warning=lambda *args: None)
    runtime = VoiceRuntime(
        "discord",
        20,
        10,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(sync=None),
    )
    pending = asyncio.create_task(asyncio.sleep(60))
    runtime.connection_task = pending
    plugin.sessions[("discord", 20)] = runtime
    await plugin.stop(runtime)
    assert pending.cancelled()
    assert not plugin.sessions


def test_initialization_supports_desktop_base_without_logger(monkeypatch):
    from astrbot.api.star import Star

    def legacy_init(self, context, config):
        self.context = context

    monkeypatch.setattr(Star, "__init__", legacy_init)
    plugin = DiscordGPTLivePlugin(SimpleNamespace(), {})
    assert hasattr(plugin, "logger")


@pytest.mark.asyncio
async def test_text_request_during_voice_initialization_is_safe():
    plugin = object.__new__(DiscordGPTLivePlugin)
    adapter = SimpleNamespace(
        umo="group",
        sync=None,
        inject_pending=lambda req: (_ for _ in ()).throw(AssertionError("not ready")),
    )
    plugin.sessions = {("discord", 20): SimpleNamespace(adapter=adapter, closing=False)}
    event = SimpleNamespace(unified_msg_origin="group", get_extra=lambda key: None)
    request = SimpleNamespace(contexts=[], func_tool=None)
    await plugin.on_request(event, request)
    assert request.contexts == []


@pytest.mark.asyncio
async def test_non_content_edit_does_not_restart_voice():
    plugin = object.__new__(DiscordGPTLivePlugin)
    runtime = SimpleNamespace(text_channel_id=10, restart=False, closing=False)
    plugin.sessions = {("discord", 20): runtime}
    await plugin.observe_edit(
        "discord", SimpleNamespace(channel_id=10, message_id=1, data={"embeds": []})
    )
    assert not runtime.restart


@pytest.mark.asyncio
async def test_rebuild_refreshes_live_without_rejoining_discord(tmp_path):
    from discord_gpt_live.main import VoiceRuntime

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.cleanup_tasks = set()
    plugin.config = {"voice_provider_id": "official/live"}
    adapter = SimpleNamespace(
        sync=SimpleNamespace(checkpoint=AsyncMock()), flush=AsyncMock()
    )
    runtime = VoiceRuntime(
        "discord", 20, 10, SimpleNamespace(), SimpleNamespace(), adapter
    )
    runtime.voice = SimpleNamespace(
        is_recording=lambda: True,
        stop_recording=lambda: None,
        stop=lambda: None,
        is_connected=lambda: True,
        disconnect=AsyncMock(),
    )
    runtime.live = SimpleNamespace(close=AsyncMock())
    runtime.audio = SimpleNamespace(close=lambda: None)
    runtime.restart = True
    plugin.sessions = {("discord", 20): runtime}
    plugin.start_streams = AsyncMock()
    await plugin.rebuild(runtime)
    plugin.start_streams.assert_awaited_once_with(runtime)
    runtime.voice.disconnect.assert_not_awaited()
    assert plugin.sessions[("discord", 20)] is runtime
    assert not runtime.restart and not runtime.rebuilding


@pytest.mark.asyncio
async def test_normal_text_request_is_tagged_for_future_edits(tmp_path):
    from discord_gpt_live.context_sync import ContextSync

    plugin = object.__new__(DiscordGPTLivePlugin)
    sync = ContextSync("group", "cid", tmp_path / "pending.json")
    sync.add_text("123", "Alice", "original question")
    adapter = SimpleNamespace(umo="group", sync=sync, inject_pending=lambda req: None)
    plugin.sessions = {("discord", 20): SimpleNamespace(adapter=adapter, closing=False)}
    event = SimpleNamespace(
        unified_msg_origin="group",
        get_extra=lambda key: None,
        message_obj=SimpleNamespace(message_id="123"),
        set_extra=lambda key, value: None,
    )
    request = SimpleNamespace(prompt="original question", contexts=[], func_tool=None)
    await plugin.on_request(event, request)
    assert request.prompt.endswith("[discord-text:123]")


@pytest.mark.asyncio
async def test_stop_releases_connection_even_when_recovery_file_fails():
    from discord_gpt_live.main import VoiceRuntime

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.sessions = {}
    plugin.last_errors = {}
    plugin.logger = SimpleNamespace(warning=lambda *args: None)
    adapter = SimpleNamespace(
        sync=SimpleNamespace(checkpoint=AsyncMock(side_effect=OSError("disk failed"))),
        flush=AsyncMock(side_effect=OSError("database failed")),
        reflect_voice=AsyncMock(),
    )
    runtime = VoiceRuntime(
        "discord", 20, 10, SimpleNamespace(), SimpleNamespace(), adapter
    )
    runtime.voice = SimpleNamespace(
        is_recording=lambda: False,
        disconnect=AsyncMock(),
    )
    runtime.live = SimpleNamespace(
        close=AsyncMock(side_effect=RuntimeError("transport"))
    )
    plugin.sessions[("discord", 20)] = runtime
    await plugin.stop(runtime)
    runtime.voice.disconnect.assert_awaited_once()
    assert not plugin.sessions
    assert plugin.last_errors[("discord", 20)]


@pytest.mark.asyncio
async def test_leave_cancels_a_rebuild_waiting_for_live_start():
    from discord_gpt_live.main import VoiceRuntime

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.sessions = {}
    plugin.last_errors = {}
    plugin.logger = SimpleNamespace(warning=lambda *args: None)
    adapter = SimpleNamespace(
        sync=SimpleNamespace(checkpoint=AsyncMock()),
        flush=AsyncMock(),
        reflect_voice=AsyncMock(),
    )
    runtime = VoiceRuntime(
        "discord", 20, 10, SimpleNamespace(), SimpleNamespace(), adapter
    )
    runtime.voice = SimpleNamespace(
        is_recording=lambda: False,
        stop=lambda: None,
        disconnect=AsyncMock(),
    )
    runtime.live = SimpleNamespace(close=AsyncMock())
    runtime.audio = SimpleNamespace(close=lambda: None)
    reached = asyncio.Event()

    async def start_streams(current):
        reached.set()
        await asyncio.sleep(60)

    plugin.start_streams = start_streams
    plugin.sessions[("discord", 20)] = runtime
    runtime.connection_task = asyncio.create_task(plugin.rebuild(runtime))
    await asyncio.wait_for(reached.wait(), 1)
    await asyncio.wait_for(plugin.stop(runtime), 1)
    assert not plugin.sessions
    assert runtime.connection_task.cancelled()


@pytest.mark.asyncio
async def test_replacement_discord_client_owns_listeners(monkeypatch):
    from unittest.mock import Mock
    from discord_gpt_live import main

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.config = {"voice_provider_id": "official/live"}
    plugin.context = SimpleNamespace(get_config=lambda: {})
    old_client = SimpleNamespace(remove_listener=Mock())
    old_callback = Mock()
    client = SimpleNamespace(add_listener=Mock())
    plugin.listeners = {"discord": (old_client, [(old_callback, "on_message")])}
    channel = SimpleNamespace(
        guild=SimpleNamespace(me=object(), voice_client=None),
        permissions_for=lambda member: SimpleNamespace(
            view_channel=True, connect=True, speak=True
        ),
        connect=AsyncMock(side_effect=RuntimeError("Stop before networking")),
    )
    runtime = SimpleNamespace(
        event=SimpleNamespace(client=client),
        channel=channel,
        platform_id="discord",
        voice=None,
    )
    monkeypatch.setattr(main, "resolve_live_provider", lambda *args: object())
    monkeypatch.setattr("discord.voice.VoiceClient", lambda *args: object())
    monkeypatch.setattr(main.inspect, "getsource", lambda value: "dave.decrypt")
    with pytest.raises(RuntimeError, match="Stop before networking"):
        await plugin.connect(runtime)
    old_client.remove_listener.assert_called_once_with(old_callback, "on_message")
    assert plugin.listeners["discord"][0] is client
    assert {call.args[1] for call in client.add_listener.call_args_list} == {
        "on_message",
        "on_raw_message_edit",
        "on_raw_message_delete",
        "on_voice_state_update",
    }


@pytest.mark.asyncio
async def test_departing_member_releases_audio_buffers():
    from unittest.mock import Mock

    plugin = object.__new__(DiscordGPTLivePlugin)
    audio = SimpleNamespace(remove_user=Mock())
    runtime = SimpleNamespace(channel=SimpleNamespace(id=30), audio=audio)
    plugin.sessions = {("discord", 20): runtime}
    member = SimpleNamespace(id=40, guild=SimpleNamespace(id=20))
    before = SimpleNamespace(channel=SimpleNamespace(id=30))
    after = SimpleNamespace(channel=SimpleNamespace(id=31))
    await plugin.observe_voice_state("discord", member, before, after)
    audio.remove_user.assert_called_once_with(40)
    audio.remove_user.reset_mock()
    await plugin.observe_voice_state("discord", member, before, before)
    audio.remove_user.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted", [False, True], ids=["edit", "delete"])
async def test_startup_replays_edits_and_deletes_before_sending_history(
    tmp_path, monkeypatch, deleted
):
    from discord_gpt_live import main
    from discord_gpt_live.context_sync import ContextSync
    from discord_gpt_live.main import VoiceRuntime
    from unittest.mock import Mock

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.config = {"voice_provider_id": "official/live"}
    plugin.context = SimpleNamespace(get_config=lambda: {})
    plugin.sync_text = AsyncMock()
    plugin.sync_history = AsyncMock()
    plugin.watch = AsyncMock()
    adapter = SimpleNamespace(sync=None, run_backend=AsyncMock())
    runtime = VoiceRuntime(
        "discord",
        20,
        10,
        SimpleNamespace(guild=SimpleNamespace()),
        SimpleNamespace(client=SimpleNamespace(user=SimpleNamespace(id=1))),
        adapter,
    )
    plugin.sessions = {("discord", 20): runtime}
    message = SimpleNamespace(
        guild=SimpleNamespace(id=20),
        channel=SimpleNamespace(id=10),
        id=123,
        author=SimpleNamespace(display_name="Alice", bot=False),
        content="Obsolete fact",
        attachments=[],
    )
    await plugin.observe_message("discord", message)
    payload = SimpleNamespace(
        channel_id=10, message_id=123, data={"content": "Corrected fact"}
    )
    if deleted:
        await plugin.observe_delete("discord", payload)
    else:
        await plugin.observe_edit("discord", payload)

    async def initialize(extra):
        adapter.sync = ContextSync("group", "cid", tmp_path / "pending.json")
        return "Persona", [
            {"role": "user", "content": "Obsolete fact\n[discord-text:123]"}
        ]

    from discord_gpt_live.astrbot_bridge import AstrBotBridge

    adapter.snapshot = lambda history: AstrBotBridge.snapshot(adapter, history)
    adapter.initialize = initialize
    fake_live = SimpleNamespace(start=AsyncMock())
    live_factory = Mock(return_value=fake_live)
    monkeypatch.setattr(main, "LiveSession", live_factory)
    monkeypatch.setattr(
        main,
        "resolve_live_provider",
        lambda *args: SimpleNamespace(
            api_key="test",
            model="configured-live",
            websocket_url="wss://voice.example.test/v1/live/sessions",
            headers={},
        ),
    )
    monkeypatch.setattr(
        main,
        "DiscordAudioBridge",
        lambda *args, **kwargs: SimpleNamespace(sink=None, source=None),
    )
    runtime.voice = SimpleNamespace(start_recording=Mock(), play=Mock())
    await plugin.start_streams(runtime)
    await asyncio.gather(*runtime.tasks)
    assert len(runtime.voice.start_recording.call_args.args) == 1
    history = live_factory.call_args.kwargs["history"]
    rendered = str(history) + str(list(adapter.sync.text_queue))
    assert "Obsolete fact" not in rendered
    assert rendered.count("Corrected fact") == (0 if deleted else 1)
    assert adapter.sync.text_records["123"]["deleted"] == deleted
    assert not runtime.restart


@pytest.mark.asyncio
async def test_voice_request_stops_when_agent_binds_a_new_conversation():
    from unittest.mock import Mock

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.sessions = {}
    extra = {"discord_live_request": True, "discord_live_conversation_id": "old"}
    event = SimpleNamespace(get_extra=extra.get, stop_event=Mock())
    request = SimpleNamespace(conversation=SimpleNamespace(cid="new"))
    await plugin.on_request(event, request)
    event.stop_event.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("completed", [False, True])
async def test_text_pipeline_state_is_shared_with_live(tmp_path, completed):
    from discord_gpt_live.context_sync import ContextSync

    plugin = object.__new__(DiscordGPTLivePlugin)
    sync = ContextSync("group", "cid", tmp_path / "pending.json")
    adapter = SimpleNamespace(umo="group", sync=sync, inject_pending=lambda req: None)
    runtime = SimpleNamespace(adapter=adapter, closing=False, error=None)
    plugin.sessions = {("discord", 20): runtime}
    extra = {}
    event = SimpleNamespace(
        unified_msg_origin="group",
        get_extra=extra.get,
        set_extra=lambda key, value: extra.__setitem__(key, value),
        message_obj=SimpleNamespace(message_id="123"),
    )
    request = SimpleNamespace(prompt="Question", contexts=[], func_tool=None)

    async def pipeline():
        await plugin.on_request(event, request)
        assert sync.text_tasks["123"]["state"] == "processing"
        if completed:
            await plugin.on_response(event, SimpleNamespace(completion_text="Answer"))

    await asyncio.create_task(pipeline())
    await asyncio.sleep(0)
    assert sync.text_tasks["123"]["state"] == (
        "completed" if completed else "ended_without_result"
    )


@pytest.mark.asyncio
async def test_stop_records_audio_discard_before_saving_pending_history(tmp_path):
    import numpy as np
    from discord_gpt_live.audio_bridge import DiscordAudioBridge
    from discord_gpt_live.context_sync import ContextSync
    from discord_gpt_live.main import VoiceRuntime

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.sessions = {}
    plugin.last_errors = {}
    plugin.logger = SimpleNamespace(warning=lambda *args: None)
    sync = ContextSync("group", "cid", tmp_path / "pending.json")
    sync.add_voice("live", "out1", "assistant", "Answer", 0, 200)
    audio = DiscordAudioBridge(99)
    audio.feed_output(np.full(4800, 1000, dtype="<i2").tobytes())
    audio.source.read()

    async def flush():
        assert sync.pending[0]["playback"]["output_handed_to_discord_ms"] == 20
        assert sync.pending[0]["playback"]["output_discarded_ms"] > 0
        await sync.checkpoint()

    adapter = SimpleNamespace(sync=sync, flush=flush, reflect_voice=AsyncMock())
    runtime = VoiceRuntime(
        "discord", 20, 10, SimpleNamespace(), SimpleNamespace(), adapter
    )
    runtime.audio = audio
    runtime.live = SimpleNamespace(
        live_id="live", close=AsyncMock(side_effect=audio.close)
    )
    runtime.voice = SimpleNamespace(is_recording=lambda: False, disconnect=AsyncMock())
    plugin.sessions[("discord", 20)] = runtime
    await plugin.stop(runtime)
    assert not plugin.sessions
    assert runtime.error is None


@pytest.mark.asyncio
async def test_connect_uses_real_discord_client_channel_and_permissions(monkeypatch):
    import discord
    from discord_gpt_live.main import VoiceRuntime
    from astrbot.core.platform.sources.discord.client import DiscordBotClient

    client = DiscordBotClient("test")
    from functools import partial

    old_plugin = object.__new__(DiscordGPTLivePlugin)
    old_plugin.terminating = True
    orphan = partial(old_plugin.observe_message, "discord")
    # The affected Pycord version appends the callback before logging its missing name.
    with pytest.raises(AttributeError):
        client.add_listener(orphan, "on_message")
    user = {
        "id": "99",
        "username": "test",
        "discriminator": "0000",
        "avatar": None,
        "bot": True,
    }
    client._connection.user = discord.ClientUser(state=client._connection, data=user)
    guild = discord.Guild(
        state=client._connection,
        data={
            "id": "20",
            "name": "Test",
            "owner_id": "99",
            "roles": [
                {
                    "id": "20",
                    "name": "@everyone",
                    "permissions": "8",
                    "position": 0,
                    "color": 0,
                    "colors": {
                        "primary_color": 0,
                        "secondary_color": None,
                        "tertiary_color": None,
                    },
                    "hoist": False,
                    "managed": False,
                    "mentionable": False,
                }
            ],
            "members": [
                {
                    "user": user,
                    "roles": [],
                    "joined_at": "2026-01-01T00:00:00+00:00",
                    "deaf": False,
                    "mute": False,
                }
            ],
            "channels": [
                {
                    "id": "30",
                    "type": 2,
                    "name": "Voice",
                    "position": 0,
                    "permission_overwrites": [],
                    "bitrate": 64000,
                    "user_limit": 0,
                }
            ],
            "voice_states": [],
        },
    )
    channel = guild.get_channel(30)
    monkeypatch.setattr(
        discord.VoiceChannel,
        "connect",
        AsyncMock(side_effect=RuntimeError("network boundary")),
    )
    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.config = {"voice_provider_id": "official/live"}
    plugin.listeners = {}
    plugin.context = SimpleNamespace(
        get_config=lambda: {
            "provider": [
                {
                    "id": "official/live",
                    "provider_source_id": "official",
                    "model": "configured-live",
                    "enable": True,
                }
            ],
            "provider_sources": [
                {
                    "id": "official",
                    "api_base": "https://api.openai.com/v1",
                    "key": ["test"],
                }
            ],
        }
    )
    runtime = VoiceRuntime(
        "discord",
        20,
        10,
        channel,
        SimpleNamespace(client=client),
        SimpleNamespace(sync=None),
    )
    try:
        with pytest.raises(RuntimeError, match="network boundary"):
            await plugin.connect(runtime)
        assert orphan not in client._event_handlers["on_message"]
    finally:
        for callback, name in plugin.listeners.get("discord", (None, []))[1]:
            client.remove_listener(callback, name)
        if runtime.voice is not None:
            await runtime.voice.disconnect(force=True)
        await client.close()


@pytest.mark.asyncio
async def test_partial_listener_registration_is_rolled_back():
    registered = []

    def add_listener(callback, name):
        registered.append((callback, name))
        raise RuntimeError("registration failed")

    def remove_listener(callback, name):
        if (callback, name) in registered:
            registered.remove((callback, name))

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.config = {"voice_provider_id": "official/live"}
    plugin.listeners = {}
    plugin.context = SimpleNamespace(
        get_config=lambda: {
            "provider": [
                {
                    "id": "official/live",
                    "provider_source_id": "official",
                    "model": "configured-live",
                    "enable": True,
                }
            ],
            "provider_sources": [
                {
                    "id": "official",
                    "api_base": "https://api.openai.com/v1",
                    "key": ["test"],
                }
            ],
        }
    )
    client = SimpleNamespace(add_listener=add_listener, remove_listener=remove_listener)
    channel = SimpleNamespace(
        guild=SimpleNamespace(me=object(), voice_client=None),
        permissions_for=lambda member: SimpleNamespace(
            view_channel=True, connect=True, speak=True
        ),
    )
    runtime = SimpleNamespace(
        event=SimpleNamespace(client=client), channel=channel, platform_id="discord"
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        await plugin.connect(runtime)
    assert registered == []
    assert not plugin.listeners


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped", ["input", "output"])
async def test_watch_closes_when_audio_transport_stops(stopped):
    from discord_gpt_live.main import VoiceRuntime

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.cleanup_tasks = set()
    plugin.stop = AsyncMock()
    adapter = SimpleNamespace(binding_changed=AsyncMock(return_value=False))
    runtime = VoiceRuntime(
        "discord",
        20,
        10,
        SimpleNamespace(members=[SimpleNamespace(bot=False)]),
        SimpleNamespace(),
        adapter,
    )
    runtime.live = SimpleNamespace(error=None)
    runtime.audio = SimpleNamespace(error=None)
    runtime.voice = SimpleNamespace(
        is_connected=lambda: True,
        is_recording=lambda: stopped != "input",
        is_playing=lambda: stopped != "output",
    )
    await asyncio.wait_for(plugin.watch(runtime), 0.3)
    await asyncio.gather(*plugin.cleanup_tasks)
    assert runtime.error
    plugin.stop.assert_awaited_once_with(runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("expires", [False, True])
async def test_connect_waits_for_usable_dave_keys(monkeypatch, expires):
    import discord.voice
    from discord_gpt_live.main import VoiceRuntime

    session = SimpleNamespace(ready=False)
    voice = SimpleNamespace(
        _connection=SimpleNamespace(dave_session=session, dave_protocol_version=1)
    )
    client = SimpleNamespace(
        add_listener=lambda *args: None, remove_listener=lambda *args: None
    )
    guild = SimpleNamespace(
        me=object(), voice_client=None, change_voice_state=AsyncMock()
    )
    channel = SimpleNamespace(
        guild=guild,
        connect=AsyncMock(),
        permissions_for=lambda member: SimpleNamespace(
            view_channel=True, connect=True, speak=True
        ),
    )
    monkeypatch.setattr(discord.voice, "VoiceClient", lambda *args: voice)
    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.config = {"voice_provider_id": "official/live"}
    plugin.listeners = {}
    plugin.start_streams = AsyncMock()
    plugin.context = SimpleNamespace(
        get_config=lambda: {
            "provider": [
                {
                    "id": "official/live",
                    "provider_source_id": "official",
                    "model": "configured-live",
                    "enable": True,
                }
            ],
            "provider_sources": [
                {
                    "id": "official",
                    "api_base": "https://api.openai.com/v1",
                    "key": ["test"],
                }
            ],
        }
    )
    runtime = VoiceRuntime(
        "discord", 20, 10, channel, SimpleNamespace(client=client), SimpleNamespace()
    )
    if expires:
        ticks = iter([0.0, 16.0])
        monkeypatch.setattr(
            "discord_gpt_live.main.time", SimpleNamespace(monotonic=lambda: next(ticks))
        )
        with pytest.raises(TimeoutError, match="DAVE"):
            await plugin.connect(runtime)
        plugin.start_streams.assert_not_awaited()
    else:
        task = asyncio.create_task(plugin.connect(runtime))
        try:
            await asyncio.sleep(0.01)
            plugin.start_streams.assert_not_awaited()
            session.ready = True
            await asyncio.wait_for(task, 1)
            plugin.start_streams.assert_awaited_once_with(runtime)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content, is_bot, ignored",
    [
        ("已加入语音频道，并绑定当前群聊。", True, True),
        ("<@123> 已加入语音频道，并绑定当前群聊。", True, True),
        ("<@!123>\n语音连接失败。", True, True),
        ("<@123> <@456> 语音会话已结束。", True, True),
        ("  <@123> Live 状态：已连接", True, True),
        ("Live 状态：已连接", True, True),
        ("<@123> 已加入语音频道，这是用户说的话。", False, False),
        ("<@123> 你好，我记得刚才的讨论。", True, False),
    ],
)
async def test_text_observer_filters_status_with_leading_mentions(
    content, is_bot, ignored, tmp_path
):
    from discord_gpt_live.context_sync import ContextSync

    plugin = object.__new__(DiscordGPTLivePlugin)
    sync = ContextSync("group", "cid", tmp_path / "pending.json")
    runtime = SimpleNamespace(
        text_channel_id=10,
        adapter=SimpleNamespace(sync=sync),
        closing=False,
    )
    plugin.sessions = {("discord", 20): runtime}
    message = SimpleNamespace(
        guild=SimpleNamespace(id=20),
        channel=SimpleNamespace(id=10),
        id=30,
        author=SimpleNamespace(display_name="Alice", bot=is_bot),
        content=content,
    )
    await plugin.observe_message("discord", message)
    if ignored:
        assert not sync.text_records
    else:
        assert sync.text_records["30"]["text"] == content


@pytest.fixture
def join_environment(monkeypatch, tmp_path):
    from discord_gpt_live.main import StarTools

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.sessions = {}
    plugin.join_lock = asyncio.Lock()
    plugin.context = SimpleNamespace(get_config=lambda **kwargs: {})
    monkeypatch.setattr(StarTools, "get_data_dir", lambda: tmp_path)

    async def connect(runtime):
        runtime.live = SimpleNamespace(respond_to_text=False)

    plugin.connect = AsyncMock(side_effect=connect)

    def event(platform_id, bot_id, channel_id=30, members=None):
        channel = SimpleNamespace(
            id=channel_id, type=SimpleNamespace(name="voice"), members=members or []
        )
        guild = SimpleNamespace(
            id=20,
            get_member=lambda uid: SimpleNamespace(
                voice=SimpleNamespace(channel=channel)
            ),
        )
        return SimpleNamespace(
            get_platform_name=lambda: "discord",
            get_platform_id=lambda: platform_id,
            get_sender_id=lambda: "99",
            get_group_id=lambda: "10",
            unified_msg_origin=f"{platform_id}:GroupMessage:10",
            message_obj=SimpleNamespace(raw_message=SimpleNamespace(guild=guild)),
            client=SimpleNamespace(
                user=SimpleNamespace(id=bot_id, display_name=platform_id)
            ),
            plain_result=lambda message: message,
        )

    return plugin, event


@pytest.mark.asyncio
async def test_join_rejects_an_existing_bot_in_target_channel(join_environment):
    plugin, event = join_environment
    other = SimpleNamespace(id=2, bot=True, display_name="Other bot")
    replies = [
        message async for message in plugin.join(event("first", 1, members=[other]))
    ]
    plugin.connect.assert_not_awaited()
    assert not plugin.sessions
    assert "Other bot" in replies[0]
    assert "不能" in replies[0]


@pytest.mark.asyncio
async def test_join_reserves_channel_across_bot_platforms_before_discord_update(
    join_environment,
):
    plugin, event = join_environment
    entered = asyncio.Event()
    release = asyncio.Event()

    async def connect(runtime):
        entered.set()
        if runtime.platform_id == "first":
            await release.wait()
        runtime.live = SimpleNamespace(respond_to_text=False)

    plugin.connect.side_effect = connect
    first = plugin.join(event("first", 1))
    pending = asyncio.create_task(anext(first))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        replies = [message async for message in plugin.join(event("second", 2))]
        assert len(plugin.sessions) == 1
        assert plugin.connect.await_count == 1
        assert "first" in replies[0]
        assert "不能" in replies[0]
    finally:
        release.set()
        await pending
        await first.aclose()


@pytest.mark.asyncio
async def test_join_allows_two_bots_in_different_voice_channels(join_environment):
    plugin, event = join_environment
    first = [message async for message in plugin.join(event("first", 1, 30))]
    second = [message async for message in plugin.join(event("second", 2, 31))]
    assert plugin.connect.await_count == 2
    assert len(plugin.sessions) == 2
    assert "已加入" in first[0] and "已加入" in second[0]


@pytest.mark.asyncio
async def test_join_allows_human_members_and_handles_repeat_command(join_environment):
    plugin, event = join_environment
    human = SimpleNamespace(id=99, bot=False, display_name="Alice")
    request = event("first", 1, members=[human])
    first = [message async for message in plugin.join(request)]
    second = [message async for message in plugin.join(request)]
    assert plugin.connect.await_count == 1
    assert "已加入" in first[0]
    assert "已有语音会话" in second[0]


@pytest.mark.asyncio
async def test_stop_closes_live_even_when_discord_recording_cleanup_raises():
    from unittest.mock import Mock
    from discord_gpt_live.main import VoiceRuntime

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.sessions = {}
    plugin.last_errors = {}
    plugin.logger = SimpleNamespace(warning=lambda *args: None)
    runtime = VoiceRuntime(
        "discord",
        20,
        10,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(sync=None),
    )
    runtime.voice = SimpleNamespace(
        is_recording=lambda: True,
        stop_recording=Mock(side_effect=RuntimeError("receiver already stopped")),
        disconnect=AsyncMock(),
    )
    runtime.live = SimpleNamespace(close=AsyncMock(), error=None)
    plugin.sessions[("discord", 20)] = runtime
    await plugin.stop(runtime)
    runtime.live.close.assert_awaited_once()
    runtime.voice.disconnect.assert_awaited_once()
    assert not plugin.sessions
    assert plugin.last_errors[("discord", 20)]
