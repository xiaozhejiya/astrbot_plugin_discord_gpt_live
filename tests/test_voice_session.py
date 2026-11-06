"""Verify Live event routing and cleanup without external services."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from discord_gpt_live.voice_session import LiveSession


def session():
    bridge = SimpleNamespace(
        close=lambda: None,
        feed_output=lambda pcm: None,
        read_input=lambda: bytes(960),
        error=None,
    )
    transcripts = []
    live = LiveSession(
        api_key="test-key",
        websocket_url="wss://voice.example.test/custom/v1/live/sessions",
        headers={"X-Project": "voice"},
        model="gpt-live-1",
        voice="marin",
        instructions="Inherited persona",
        history=[],
        bridge=bridge,
        on_transcript=transcripts.append,
        on_delegate=AsyncMock(return_value="Answer"),
    )
    live.ws = SimpleNamespace(closed=False, send_json=AsyncMock(), close=AsyncMock())
    return live, transcripts


@pytest.mark.asyncio
async def test_session_start_and_transcript_routing():
    live, transcripts = session()
    await live.handle_event({"type": "session.started", "session": {"id": "live_1"}})
    event = {
        "type": "session.input_transcript.delta",
        "event_id": "e1",
        "delta": "Hello",
    }
    await live.handle_event(event)
    assert live.live_id == "live_1"
    assert live.ready.is_set()
    assert transcripts == [event]


@pytest.mark.asyncio
async def test_context_append_waits_for_matching_ack():
    live, _ = session()
    task = asyncio.create_task(live.append_context("Context", event_id="ctx1"))
    await asyncio.sleep(0)
    assert not task.done()
    await live.handle_event(
        {"type": "session.thinking.appended", "client_event_id": "other"}
    )
    assert not task.done()
    await live.handle_event(
        {"type": "session.thinking.appended", "client_event_id": "ctx1"}
    )
    await task
    message = live.ws.send_json.call_args.args[0]
    assert message["delegation_id"] is None
    assert message["type"] == "session.thinking.append"


@pytest.mark.asyncio
async def test_error_rejects_correlated_append():
    live, _ = session()
    task = asyncio.create_task(live.append_context("Context", event_id="ctx1"))
    await asyncio.sleep(0)
    await live.handle_event(
        {
            "type": "error",
            "error": {"client_event_id": "ctx1", "code": "invalid_request"},
        }
    )
    with pytest.raises(RuntimeError, match="invalid_request"):
        await task


@pytest.mark.asyncio
async def test_close_releases_pending_ack_and_closes_transport():
    live, _ = session()
    task = asyncio.create_task(live.append_context("Context", event_id="ctx1"))
    await asyncio.sleep(0)
    await live.close()
    with pytest.raises(ConnectionError):
        await task
    live.ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_output_audio_decodes_strictly():
    live, _ = session()
    captured = []
    live.bridge.feed_output = captured.append
    await live.handle_event({"type": "session.output_audio.delta", "delta": "AAAAAA=="})
    assert captured == [bytes(4)]
    with pytest.raises(ValueError):
        await live.handle_event({"type": "session.output_audio.delta", "delta": "!!!"})


@pytest.mark.asyncio
async def test_text_policy_update_uses_instruction_event():
    live, _ = session()
    task = asyncio.create_task(live.update_text_policy(True))
    await asyncio.sleep(0)
    message = live.ws.send_json.call_args.args[0]
    assert message["type"] == "session.instructions.append"
    await live.handle_event(
        {
            "type": "session.instructions.appended",
            "client_event_id": message["event_id"],
        }
    )
    await task
    assert live.respond_to_text


@pytest.mark.asyncio
async def test_start_encodes_nonempty_history_for_live(monkeypatch):
    from discord_gpt_live import voice_session

    live, _ = session()
    live.history = [
        {"role": "user", "content": "Our project is Aurora."},
        {"role": "assistant", "content": "I will remember Aurora."},
    ]
    http = SimpleNamespace(
        ws_connect=AsyncMock(return_value=live.ws), close=AsyncMock()
    )
    monkeypatch.setattr(voice_session.aiohttp, "ClientSession", lambda **kwargs: http)
    live._receive = AsyncMock()
    live._send_audio = AsyncMock()
    live.ready.set()
    try:
        await live.start()
        assert (
            http.ws_connect.call_args.args[0]
            == "wss://voice.example.test/custom/v1/live/sessions"
        )
        assert http.ws_connect.call_args.kwargs["headers"] == {
            "X-Project": "voice",
            "Authorization": "Bearer test-key",
        }
        sent = live.ws.send_json.call_args_list[0].args[0]["session"]["input"]
        assert sent == [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Our project is Aurora."}],
            },
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will remember Aurora."}],
            },
        ]
    finally:
        await live.close()


@pytest.mark.asyncio
async def test_duplicate_delegation_runs_backend_once_and_preserves_id():
    live, _ = session()
    live.append_context = AsyncMock()
    event = {
        "type": "session.delegation.created",
        "offset_ms": 1000,
        "delegation": {"id": "del1", "target": "client"},
    }
    await live.handle_event(event)
    await live.handle_event(event)
    await asyncio.gather(*live.tasks)
    live.on_delegate.assert_awaited_once_with(event)
    live.append_context.assert_awaited_once_with(
        "Answer", kind="commentary", delegation_id="del1"
    )
    await live.close()


@pytest.mark.asyncio
async def test_output_transcript_carries_local_playback_snapshot():
    live, transcripts = session()
    live.bridge.stats = {"output_handed_to_discord_ms": 80, "output_buffer_ms": 20}
    await live.handle_event(
        {
            "type": "session.output_transcript.delta",
            "event_id": "out1",
            "delta": "Hello",
        }
    )
    assert transcripts[0]["playback"] == live.bridge.stats
    await live.close()
