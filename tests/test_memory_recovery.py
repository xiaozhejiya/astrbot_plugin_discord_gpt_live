"""Check durable progress when one memory hook fails after another succeeds."""

from types import SimpleNamespace

import pytest

from discord_gpt_live.astrbot_bridge import AstrBotBridge
from discord_gpt_live.context_sync import ContextSync


@pytest.mark.asyncio
async def test_memory_retry_resumes_batch_without_repeating_successful_hook(
    tmp_path, monkeypatch
):
    from astrbot.core.star.star_handler import star_handlers_registry

    calls = []
    attempts = [0]

    async def first(event, response):
        calls.append(("first", event.message_obj.message_id))

    async def second(event, response):
        calls.append(("second", event.message_obj.message_id))
        attempts[0] += 1
        if attempts[0] == 1:
            raise OSError("temporary store failure")

    handlers = [
        SimpleNamespace(
            handler_full_name="first", handler=first, handler_module_path="first"
        ),
        SimpleNamespace(
            handler_full_name="second", handler=second, handler_module_path="second"
        ),
    ]
    monkeypatch.setattr(
        star_handlers_registry, "get_handlers_by_event_type", lambda *a, **k: handlers
    )
    context = SimpleNamespace(get_all_stars=lambda: [])
    origin = SimpleNamespace(unified_msg_origin="group")
    adapter = AstrBotBridge(context, origin, tmp_path)
    path = tmp_path / "pending.json"
    adapter.sync = ContextSync("group", "cid", path)
    adapter.sync.add_voice("live", "e1", "user", "Question", 0, 100)
    adapter.sync.add_voice("live", "e2", "assistant", "Answer", 100, 200)

    def make_event(text, *, message_id=None):
        return SimpleNamespace(
            plugins_name=None,
            message_obj=SimpleNamespace(message_id=message_id),
            set_extra=lambda *args: None,
            is_stopped=lambda: False,
        )

    adapter.make_event = make_event
    with pytest.raises(OSError):
        await adapter.reflect_voice()
    recovered = AstrBotBridge(context, origin, tmp_path)
    recovered.sync = ContextSync("group", "cid", path)
    recovered.make_event = make_event
    await recovered.reflect_voice()
    assert [name for name, _ in calls] == ["first", "second", "second"]
    assert len({message_id for _, message_id in calls}) == 1
    assert calls[0][1]
    assert not recovered.sync.memory_pending
