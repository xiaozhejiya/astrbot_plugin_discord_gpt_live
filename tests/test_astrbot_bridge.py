"""Validate credential routing and voice request identity."""

from types import SimpleNamespace

import pytest

from discord_gpt_live.astrbot_bridge import resolve_live_provider, voice_rules


def model_config():
    return {
        "provider_sources": [
            {
                "id": "service",
                "api_base": "https://voice.example.test/custom/v1/",
                "key": ["service-secret"],
                "custom_headers": {"X-Project": "voice"},
            }
        ],
        "provider": [
            {
                "id": "service/voice",
                "provider_source_id": "service",
                "model": "configured-live-alias",
                "enable": True,
            }
        ],
    }


def test_selected_model_reuses_its_source_without_official_host_restriction():
    result = resolve_live_provider(model_config(), "service/voice")
    assert result.model == "configured-live-alias"
    assert result.websocket_url == "wss://voice.example.test/custom/v1/live/sessions"
    assert result.api_key == "service-secret"
    assert result.headers == {"X-Project": "voice"}
    assert "service-secret" not in repr(result)


def test_model_overrides_source_and_supports_configured_local_endpoint():
    config = model_config()
    config["provider"][0].update(
        api_base="http://127.0.0.1:9000/v1", key=["", "model-key"]
    )
    result = resolve_live_provider(config, "service/voice")
    assert result.websocket_url == "ws://127.0.0.1:9000/v1/live/sessions"
    assert result.api_key == "model-key"


@pytest.mark.parametrize("provider_id", ["", "missing", "service"])
def test_missing_model_selection_does_not_fall_back_to_another_source(provider_id):
    with pytest.raises(ValueError):
        resolve_live_provider(model_config(), provider_id)


@pytest.mark.parametrize(
    "field,value",
    [("enable", False), ("model", ""), ("key", []), ("provider_source_id", "missing")],
)
def test_invalid_selected_model_is_rejected(field, value):
    config = model_config()
    config["provider"][0][field] = value
    with pytest.raises(ValueError):
        resolve_live_provider(config, "service/voice")


@pytest.mark.parametrize(
    "url",
    [
        "",
        "ftp://voice.example.test/v1",
        "https://user:secret@voice.example.test/v1",
        "https://voice.example.test/v1?key=secret",
        "https://voice.example.test/v1#fragment",
    ],
)
def test_ambiguous_model_endpoint_is_rejected_without_leaking_credentials(url):
    config = model_config()
    config["provider_sources"][0]["api_base"] = url
    with pytest.raises(ValueError) as error:
        resolve_live_provider(config, "service/voice")
    assert "secret" not in str(error.value)


def test_voice_rules_preserve_persona_and_delegate_memory():
    rules = voice_rules("Original group persona", "Speak softly")
    assert "Original group persona" in rules
    assert "Speak softly" in rules
    assert "<|SpawnThinking|>" in rules
    assert "memory" in rules.lower()


@pytest.mark.asyncio
async def test_adapter_binds_original_umo_and_persona(tmp_path):
    from unittest.mock import AsyncMock
    from discord_gpt_live.astrbot_bridge import AstrBotBridge

    event = SimpleNamespace(
        unified_msg_origin="discord:GroupMessage:123",
        get_platform_id=lambda: "discord",
        get_platform_name=lambda: "discord",
    )
    conversation = SimpleNamespace(
        cid="original",
        persona_id="persona1",
        history='[{"role":"user","content":"Remember our project"}]',
    )
    context = SimpleNamespace(
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="original"),
            get_conversation=AsyncMock(return_value=conversation),
        ),
        persona_manager=SimpleNamespace(
            resolve_selected_persona=AsyncMock(
                return_value=("persona1", {"prompt": "Original persona"}, None, False)
            ),
        ),
        get_config=lambda **kwargs: {"provider_settings": {}},
    )
    adapter = AstrBotBridge(context, event, tmp_path)
    instructions, history = await adapter.initialize("")
    assert "Original persona" in instructions
    assert history[0]["content"] == "Remember our project"
    assert adapter.sync.cid == "original"
    assert adapter.sync.umo == event.unified_msg_origin


def test_backend_request_gets_recent_unawakened_text_without_repeating_current(
    tmp_path,
):
    from discord_gpt_live.astrbot_bridge import AstrBotBridge
    from discord_gpt_live.context_sync import ContextSync

    adapter = AstrBotBridge(None, SimpleNamespace(unified_msg_origin="group"), tmp_path)
    adapter.sync = ContextSync("group", "cid", tmp_path / "pending.json")
    adapter.sync.add_text("123", "Alice", "A new fact")
    adapter.sync.add_text("124", "Bob", "Current request")
    request = SimpleNamespace(contexts=[], prompt="Current request\n[discord-text:124]")
    adapter.inject_pending(request)
    assert any("A new fact" in str(message) for message in request.contexts)
    assert not any("Current request" in str(message) for message in request.contexts)


def test_pending_voice_is_not_injected_after_conversation_reset(tmp_path):
    from discord_gpt_live.astrbot_bridge import AstrBotBridge
    from discord_gpt_live.context_sync import ContextSync

    adapter = AstrBotBridge(None, SimpleNamespace(unified_msg_origin="group"), tmp_path)
    adapter.sync = ContextSync("group", "old", tmp_path / "pending.json")
    adapter.sync.add_voice("live", "e1", "user", "Old private context", 0, 100)
    request = SimpleNamespace(
        contexts=[], prompt="New question", conversation=SimpleNamespace(cid="new")
    )
    adapter.inject_pending(request)
    assert request.contexts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_count,content",
    [(200, "Small fact"), (2, "新" * 20000)],
    ids=["many_messages", "long_unicode"],
)
async def test_initial_history_bounds_include_recovery_records(
    tmp_path, message_count, content
):
    import json
    from unittest.mock import AsyncMock
    from discord_gpt_live.astrbot_bridge import AstrBotBridge
    from discord_gpt_live.context_sync import ContextSync

    event = SimpleNamespace(
        unified_msg_origin="group",
        get_platform_name=lambda: "discord",
    )
    conversation = SimpleNamespace(
        cid="cid",
        persona_id=None,
        history=json.dumps([{"role": "user", "content": content}] * message_count),
    )
    context = SimpleNamespace(
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="cid"),
            get_conversation=AsyncMock(return_value=conversation),
        ),
        persona_manager=SimpleNamespace(
            resolve_selected_persona=AsyncMock(return_value=(None, None, None, False))
        ),
        get_config=lambda **kwargs: {},
    )
    adapter = AstrBotBridge(context, event, tmp_path)
    adapter.sync = ContextSync("group", "cid", tmp_path / "pending.json")
    for index in range(200):
        adapter.sync.add_voice(
            "previous", str(index), "user", content, index, index + 1
        )
    _, raw_history = await adapter.initialize("")
    history = adapter.snapshot(raw_history)
    assert 0 < len(history) <= 128
    assert sum(len(item["content"].encode("utf-8")) for item in history) <= 6000
    assert history[-1]["content"].endswith(content[-10:])
    assert len(adapter.sync.pending) == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "late_transcript",
        "missing_transcript",
        "correction",
        "earlier_transcript",
        "text_request",
    ],
)
async def test_backend_uses_available_transcript_and_reports_later_changes(
    tmp_path, monkeypatch, scenario
):
    import asyncio
    from unittest.mock import AsyncMock
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal
    from discord_gpt_live.astrbot_bridge import AstrBotBridge
    from discord_gpt_live.context_sync import ContextSync

    adapter = AstrBotBridge(
        SimpleNamespace(get_config=lambda **kwargs: {}),
        SimpleNamespace(unified_msg_origin="group"),
        tmp_path,
    )
    adapter.sync = ContextSync("group", "cid", tmp_path / "pending.json")
    adapter.binding_changed = AsyncMock(return_value=False)
    adapter.flush = AsyncMock()
    event = SimpleNamespace(
        captured=[], get_result=lambda: None, set_extra=lambda *args: None
    )
    adapter.make_event = lambda prompt: event
    initialized = AsyncMock()

    def transcript(identifier, content, end_ms=1000):
        adapter.transcript(
            "live1",
            {
                "type": "session.input_transcript.delta",
                "event_id": identifier,
                "delta": content,
                "start_ms": 0,
                "end_ms": end_ms,
            },
        )

    class Stage:
        initialize = initialized

        async def process(self, current, prompt):
            if scenario == "correction":
                transcript("correction", "Actually, check the second project.")
            current.captured.append("The first project is Aurora.")
            yield None

    monkeypatch.setattr(internal, "InternalAgentSubStage", Stage)
    if scenario in ("correction", "earlier_transcript"):
        transcript(
            "initial",
            "What is the first project?",
            900 if scenario == "earlier_transcript" else 1000,
        )
    if scenario == "text_request":
        adapter.respond_to_text = True
        adapter.sync.add_text("123", "Alice", "What is the first project?")
    task = asyncio.create_task(
        adapter.run_backend({"offset_ms": 1000, "delegation": {"id": "del1"}})
    )
    await asyncio.sleep(0)
    if scenario != "correction":
        assert not initialized.called
    if scenario == "late_transcript":
        transcript("late", "What is the first project?")
    result = await asyncio.wait_for(task, 3)
    if scenario == "missing_transcript":
        assert "confirm" in result.lower()
        initialized.assert_not_awaited()
    else:
        initialized.assert_awaited_once()
        assert "Aurora" in result
        if scenario == "correction":
            assert "newer" in result.lower()


@pytest.mark.asyncio
async def test_rejoin_seeds_saved_text_without_replaying_it_as_new_activity(tmp_path):
    from discord_gpt_live.astrbot_bridge import AstrBotBridge
    from discord_gpt_live.context_sync import ContextSync

    path = tmp_path / "pending.json"
    old = ContextSync("group", "cid", path)
    old.add_text("123", "Alice", "Earlier question")
    old.add_text("124", "Bot", "Already delivered answer", answered=True)
    old.set_text_task("123", "processing", "attempt")
    old.set_text_task("123", "completed", "attempt")
    await old.checkpoint()
    adapter = AstrBotBridge(None, SimpleNamespace(unified_msg_origin="group"), tmp_path)
    adapter.sync = ContextSync("group", "cid", path)
    history = adapter.snapshot([])
    assert adapter.sync.next_text_chunk() is None
    assert "Earlier question" in str(history)
    assert "Already delivered answer" in str(history)
    assert "completed" in str(history)
    assert adapter.snapshot(history) == history
    adapter.sync.add_text("125", "Alice", "A new question after joining")
    assert "A new question after joining" in adapter.sync.next_text_chunk()[1]


@pytest.mark.asyncio
async def test_initialization_retains_full_history_until_final_startup_snapshot(
    tmp_path,
):
    from unittest.mock import AsyncMock
    from discord_gpt_live.astrbot_bridge import AstrBotBridge
    from discord_gpt_live.context_sync import ContextSync

    adapter = AstrBotBridge(
        SimpleNamespace(
            conversation_manager=SimpleNamespace(
                get_curr_conversation_id=AsyncMock(return_value="cid"),
                get_conversation=AsyncMock(
                    return_value=SimpleNamespace(
                        cid="cid", persona_id=None, history="[]"
                    )
                ),
            ),
            persona_manager=SimpleNamespace(
                resolve_selected_persona=AsyncMock(
                    return_value=(None, None, None, False)
                )
            ),
            get_config=lambda **kwargs: {},
        ),
        SimpleNamespace(
            unified_msg_origin="group", get_platform_name=lambda: "discord"
        ),
        tmp_path,
    )
    adapter.sync = ContextSync("group", "cid", tmp_path / "pending.json")
    for index in range(1, 5):
        adapter.sync.add_text(str(index), "Alice", str(index) * 2900)
    _, raw_history = await adapter.initialize("")
    assert raw_history == []  # Merge the journal and enforce budgets only once.
    final = adapter.snapshot(raw_history)
    ids = [item["content"].split("[discord-text:")[-1].split("]")[0] for item in final]
    assert ids == ["2", "3", "4"]
    assert adapter.sync.next_text_chunk() is None
