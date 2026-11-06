"""Check shared-history journaling and incremental text synchronization."""

import asyncio
import json

import pytest

from discord_gpt_live.context_sync import ContextSync


class History:
    def __init__(self):
        self.messages = [{"role": "user", "content": "Existing group message"}]
        self.lock = asyncio.Lock()
        self.fail = False

    async def read(self):
        return list(self.messages)

    async def save(self, messages):
        if self.fail:
            raise OSError("database unavailable")
        self.messages = messages


@pytest.mark.asyncio
async def test_flush_preserves_existing_history_and_deduplicates(tmp_path):
    history = History()
    sync = ContextSync("umo", "cid", tmp_path / "pending.json")
    assert sync.add_voice("live1", "e1", "user", "Hello", 0, 200)
    assert not sync.add_voice("live1", "e1", "user", "Hello", 0, 200)
    await sync.flush(history.read, history.save, history.lock)
    await sync.flush(history.read, history.save, history.lock)
    assert len(history.messages) == 2
    assert "Hello" in history.messages[-1]["content"]
    assert not sync.pending_messages()


@pytest.mark.asyncio
async def test_failed_save_is_recoverable(tmp_path):
    path = tmp_path / "pending.json"
    sync = ContextSync("umo", "cid", path)
    history = History()
    sync.add_voice("live1", "e1", "assistant", "Result", 0, 200)
    history.fail = True
    with pytest.raises(OSError):
        await sync.flush(history.read, history.save, history.lock)
    recovered = ContextSync("umo", "cid", path)
    assert len(recovered.pending_messages()) == 1
    history.fail = False
    await recovered.flush(history.read, history.save, history.lock)
    assert len(history.messages) == 2


@pytest.mark.asyncio
async def test_new_audio_during_database_write_is_retained(tmp_path):
    sync = ContextSync("umo", "cid", tmp_path / "pending.json")
    history = History()
    sync.add_voice("live1", "e1", "user", "First", 0, 100)

    async def save(messages):
        sync.add_voice("live1", "e2", "user", "Second", 100, 200)
        await history.save(messages)

    await sync.flush(history.read, save, history.lock)
    assert "Second" in sync.pending_messages()[0]["content"]


def test_text_updates_are_attributed_and_acknowledged(tmp_path):
    sync = ContextSync("umo", "cid", tmp_path / "pending.json")
    sync.add_text("123", "Alice", "hello", answered=False)
    sync.add_text("123", "Alice", "hello", answered=False)
    first = sync.next_text_chunk()
    assert first is not None
    event_id, text = first
    assert "Alice" in text and "hello" in text
    assert sync.next_text_chunk() == first
    sync.ack_text(event_id)
    assert sync.next_text_chunk() is None
    sync.add_text("124", "Bot", "answer", answered=True)
    assert "already_answered" in sync.next_text_chunk()[1]


def test_large_unicode_text_is_bounded_per_append(tmp_path):
    sync = ContextSync("umo", "cid", tmp_path / "pending.json")
    sync.add_text("1", "Alice", "长" * 3000)
    chunks = []
    while packet := sync.next_text_chunk():
        chunks.append(packet[1])
        assert len(packet[1].encode("utf-8")) <= 1500
        sync.ack_text(packet[0])
    assert len(chunks) > 1


def test_recovery_keeps_original_conversation_binding(tmp_path):
    path = tmp_path / "pending.json"
    path.write_text(json.dumps({"umo": "old", "cid": "oldcid", "pending": []}))
    with pytest.raises(ValueError, match="binding"):
        ContextSync("new", "newcid", path)


@pytest.mark.asyncio
async def test_memory_records_survive_history_commit_and_restart(tmp_path):
    path = tmp_path / "pending.json"
    sync = ContextSync("umo", "cid", path)
    history = History()
    sync.add_voice("live1", "e1", "user", "Remember this fact", 0, 200)
    await sync.flush(history.read, history.save, history.lock)
    recovered = ContextSync("umo", "cid", path)
    assert recovered.memory_pending[0]["text"] == "Remember this fact"
    assert not recovered.add_voice("live1", "e1", "user", "Remember this fact", 0, 200)


def test_edit_replaces_queued_chunks_and_replays_latest_version(tmp_path):
    sync = ContextSync("umo", "cid", tmp_path / "pending.json")
    sync.add_text("123", "Alice", "old value")
    old_event_id = sync.next_text_chunk()[0]
    assert sync.revise_text("123", "new value")
    assert "new value" in sync.next_text_chunk()[1]
    assert "old value" not in sync.next_text_chunk()[1]
    sync.ack_text(old_event_id)
    assert sync.next_text_chunk() is not None
    sync.ack_text(sync.next_text_chunk()[0])
    sync.replay_text()
    assert "new value" in sync.next_text_chunk()[1]


def test_delete_filters_tagged_history_and_queued_text(tmp_path):
    sync = ContextSync("umo", "cid", tmp_path / "pending.json")
    sync.add_text("123", "Alice", "old value")
    history = [
        {"role": "user", "content": "old value\n[discord-text:123]"},
        {"role": "assistant", "content": "Another topic"},
    ]
    sync.revise_text("123", deleted=True)
    assert sync.next_text_chunk() is None
    assert sync.project_history(history) == [history[1]]
    sync.replay_text()
    assert sync.next_text_chunk() is None


@pytest.mark.asyncio
async def test_text_revision_survives_restart_without_voice(tmp_path):
    path = tmp_path / "pending.json"
    sync = ContextSync("umo", "cid", path)
    sync.add_text("123", "Alice", "old value")
    sync.revise_text("123", "new value")
    history = History()
    await sync.flush(history.read, history.save, history.lock)
    recovered = ContextSync("umo", "cid", path)
    assert recovered.text_records["123"]["text"] == "new value"
    result = recovered.project_history(
        [{"role": "user", "content": "old value\n[discord-text:123]"}]
    )
    assert "new value" in result[0]["content"]
    assert "old value" not in result[0]["content"]


def test_history_overlay_includes_unawakened_text_once(tmp_path):
    sync = ContextSync("umo", "cid", tmp_path / "pending.json")
    sync.add_text("123", "Alice", "New group fact")
    result = sync.project_history([], include_observed=True)
    assert len(result) == 1 and "New group fact" in result[0]["content"]
    assert sync.project_history(result, include_observed=True) == result


def test_replay_is_bounded_and_skips_snapshot_ids(tmp_path):
    sync = ContextSync("umo", "cid", tmp_path / "pending.json")
    for index in range(100):
        sync.add_text(str(index), "Alice", "字" * 4000)
        while packet := sync.next_text_chunk():
            sync.ack_text(packet[0])
    sync.replay_text(exclude_ids={"99"})
    packets = list(sync.text_queue)
    assert len(packets) < 1024
    assert all(json.loads(packet[1])["message"] != "99" for packet in packets)


@pytest.mark.asyncio
async def test_text_edit_is_committed_when_no_voice_is_pending(tmp_path):
    history = History()
    history.messages = [{"role": "user", "content": "old fact\n[discord-text:123]"}]
    sync = ContextSync("umo", "cid", tmp_path / "pending.json")
    sync.add_text("123", "Alice", "old fact")
    sync.revise_text("123", "corrected fact")
    await sync.flush(history.read, history.save, history.lock)
    assert "corrected fact" in history.messages[0]["content"]
    sync.revise_text("123", deleted=True)
    await sync.flush(history.read, history.save, history.lock)
    assert history.messages == []


def test_buffered_message_already_in_snapshot_is_not_sent_twice(tmp_path):
    sync = ContextSync("umo", "cid", tmp_path / "pending.json")
    sync.replay_text(exclude_ids={"123"})
    sync.add_text("123", "Alice", "Already in startup history")
    assert sync.next_text_chunk() is None
    assert sync.text_records["123"]["text"] == "Already in startup history"
    sync.revise_text("123", "Updated after startup")
    assert "Updated after startup" in sync.next_text_chunk()[1]


@pytest.mark.asyncio
async def test_text_task_updates_are_coalesced_and_recovered(tmp_path):
    path = tmp_path / "pending.json"
    sync = ContextSync("umo", "cid", path)
    sync.set_text_task("123", "processing", "attempt1")
    sync.set_text_task("123", "completed", "attempt1")
    sync.set_text_task("123", "ended_without_result", "attempt1")
    packets = [json.loads(packet[1]) for packet in sync.text_queue]
    assert len(packets) == 1
    assert packets[0]["origin"] == "discord_text_task"
    assert packets[0]["state"] == "completed"
    sync.set_text_task("123", "processing", "attempt2")
    sync.set_text_task("123", "completed", "attempt1")
    assert sync.text_tasks["123"]["state"] == "processing"
    await sync.checkpoint()
    recovered = ContextSync("umo", "cid", path)
    assert recovered.text_tasks["123"]["state"] == "unknown_after_restart"
    recovered.replay_text()
    assert (
        json.loads(recovered.next_text_chunk()[1])["state"] == "unknown_after_restart"
    )


@pytest.mark.asyncio
async def test_pending_output_preserves_local_playback_evidence(tmp_path):
    path = tmp_path / "pending.json"
    sync = ContextSync("umo", "cid", path)
    sync.add_voice("live", "output", "assistant", "A spoken answer", 0, 100)
    snapshot = {
        "output_generated_ms": 100,
        "output_handed_to_discord_ms": 60,
        "output_discarded_ms": 40,
    }
    sync.update_playback("live", snapshot)
    assert sync.pending[0]["playback"] == snapshot
    assert sync.memory_pending[0]["playback"] == snapshot
    assert "delivery unconfirmed" in sync.pending_messages()[0]["content"].lower()
    assert "60" in sync.pending_messages()[0]["content"]
    await sync.checkpoint()
    recovered = ContextSync("umo", "cid", path)
    assert recovered.memory_pending[0]["playback"] == snapshot


def test_text_edit_keeps_task_status_and_original_source_revision(tmp_path):
    sync = ContextSync("umo", "cid", tmp_path / "pending.json")
    sync.add_text("123", "Alice", "old request")
    sync.set_text_task("123", "processing", "attempt")
    sync.revise_text("123", "new request")
    sync.set_text_task("123", "completed", "attempt")
    packets = [json.loads(packet[1]) for packet in sync.text_queue]
    task = next(item for item in packets if item["origin"] == "discord_text_task")
    assert task["source_revision"] == 0
    assert sync.text_records["123"]["revision"] == 1


@pytest.mark.parametrize("state", ["processing", "ended_without_result"])
def test_task_replay_preserves_attempt_revision_and_terminal_state(tmp_path, state):
    sync = ContextSync("umo", "cid", tmp_path / "pending.json")
    sync.add_text("123", "Alice", "old request")
    sync.set_text_task("123", "processing", "attempt")
    if state != "processing":
        sync.set_text_task("123", state, "attempt")
    sync.revise_text("123", "new request")
    sync.replay_text()
    packets = [json.loads(packet[1]) for packet in sync.text_queue]
    task = next(item for item in packets if item["origin"] == "discord_text_task")
    assert task["source_revision"] == 0
    assert task["state"] == state
