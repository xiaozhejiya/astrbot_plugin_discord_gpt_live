"""Exercise the real AstrBot internal agent stage with a controlled provider."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_before_request", [False, True], ids=["normal", "reset"])
async def test_original_agent_runs_memory_hooks_and_saves_history(
    tmp_path, monkeypatch, reset_before_request
):
    from astrbot.api.message_components import Plain
    from astrbot.core.config.default import DEFAULT_CONFIG
    from astrbot.core.db.po import Conversation
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata
    from astrbot.core.platform.sources.discord.discord_platform_event import (
        DiscordPlatformEvent,
    )
    from astrbot.core.provider import Provider
    from astrbot.core.provider.entities import LLMResponse
    from astrbot.core.star.context import Context
    from astrbot.core.star.star_handler import EventType
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal
    from discord_gpt_live.astrbot_bridge import AstrBotBridge

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["provider_settings"]["proactive_capability"] = {"add_cron_tools": False}
    config["provider_ltm_settings"]["group_message_history_enable"] = False
    config["provider_settings"]["streaming_response"] = False
    config["plugin_set"] = ["memory_plugin"]
    config["kb_names"] = []
    config["kb_agentic_mode"] = False
    provider = MagicMock(spec=Provider)
    provider.provider_config = {
        "id": "controlled",
        "type": "openai_chat_completion",
        "modalities": ["text"],
        "max_context_tokens": 8192,
    }
    provider.get_model.return_value = "controlled-model"
    provider.text_chat = AsyncMock(
        return_value=LLMResponse(
            role="assistant",
            completion_text="The shared project is called Aurora.",
        )
    )
    context = MagicMock(spec=Context)
    context.get_config.return_value = config
    context.get_using_provider_async = AsyncMock(return_value=provider)
    context.get_using_provider.return_value = provider
    context.get_provider_by_id.return_value = provider
    context.subagent_orchestrator = None
    context.persona_manager = SimpleNamespace(
        resolve_selected_persona=AsyncMock(
            return_value=(
                "persona",
                {"prompt": "You are the original group bot."},
                None,
                False,
            )
        ),
        personas_v3=[],
    )
    tool_manager = MagicMock()
    tool_manager.get_builtin_tool.side_effect = lambda cls, **kwargs: cls(**kwargs)
    context.get_llm_tool_manager.return_value = tool_manager
    conversation = MagicMock(spec=Conversation)
    conversation.cid = "bound-conversation"
    conversation.persona_id = "persona"
    conversation.history = "[]"
    conversation.token_usage = 0

    async def save(umo, cid, history=None, **kwargs):
        assert cid == "bound-conversation"
        conversation.history = json.dumps(history)

    context.conversation_manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value=conversation.cid),
        get_conversation=AsyncMock(return_value=conversation),
        update_conversation=AsyncMock(side_effect=save),
    )
    message = AstrBotMessage()
    message.type = MessageType.GROUP_MESSAGE
    message.self_id = "900"
    message.session_id = "123"
    message.message_id = "1"
    message.group_id = "123"
    message.sender = MessageMember("42", "Alice")
    message.message_str = "live_join"
    message.message = [Plain("live_join")]
    message.raw_message = None
    origin = DiscordPlatformEvent(
        "live_join",
        message,
        PlatformMetadata(id="discord", name="discord", description="test"),
        "123",
        MagicMock(),
    )
    origin.plugins_name = []
    adapter = AstrBotBridge(context, origin, tmp_path)
    await adapter.initialize("")
    adapter.transcript(
        "live1",
        {
            "type": "session.input_transcript.delta",
            "event_id": "e1",
            "delta": "What is our project called?",
            "start_ms": 0,
            "end_ms": 1000,
        },
    )
    hooks = []
    from discord_gpt_live.main import DiscordGPTLivePlugin

    plugin = object.__new__(DiscordGPTLivePlugin)
    plugin.sessions = {("discord", 1): SimpleNamespace(adapter=adapter, closing=False)}

    async def hook(event, event_type, *args, **kwargs):
        assert event.plugins_name == ["memory_plugin"]
        hooks.append(event_type)
        if event_type == EventType.OnLLMRequestEvent:
            if reset_before_request:
                args[0].conversation = SimpleNamespace(cid="replacement-conversation")
            await plugin.on_request(event, args[0])
            if event.is_stopped():
                return True
            args[0].system_prompt += "\nShared memory: the project is Aurora."
            args[0].func_tool = None
        return False

    monkeypatch.setattr(internal, "call_event_hook", hook)
    monkeypatch.setattr("astrbot.core.astr_agent_hooks.call_event_hook", hook)
    if hasattr(internal, "SessionServiceManager"):
        monkeypatch.setattr(
            internal.SessionServiceManager,
            "should_process_llm_request",
            AsyncMock(return_value=True),
        )
    monkeypatch.setattr(
        internal, "prepare_follow_up_capture", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(internal.Metric, "upload", AsyncMock())
    result = await adapter.run_backend({"offset_ms": 1000})
    if reset_before_request:
        provider.text_chat.assert_not_awaited()
        assert "Aurora" not in conversation.history
        return
    assert "Aurora" in result
    assert EventType.OnLLMRequestEvent in hooks
    assert EventType.OnLLMResponseEvent in hooks
    assert "Shared memory: the project is Aurora." in json.dumps(
        provider.text_chat.call_args.kwargs["contexts"], default=str
    )
    assert "Aurora" in conversation.history
    assert origin.get_result() is None
