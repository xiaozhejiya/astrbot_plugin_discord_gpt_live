"""Exercise continuous audio with real sample buffers."""

import numpy as np

from discord_gpt_live.audio_bridge import DiscordAudioBridge


def pcm(value, frames=960):
    return np.full((frames, 2), value, dtype="<i2").tobytes()


def test_empty_input_is_exact_twenty_ms():
    bridge = DiscordAudioBridge(99)
    assert bridge.read_input() == bytes(960)
    assert bridge.source.read() == bytes(3840)
    bridge.close()
    assert bridge.source.read() == b""


def test_members_mix_without_overflow():
    now = [0.0]
    bridge = DiscordAudioBridge(99, clock=lambda: now[0])
    frames = []
    for index in range(100):
        now[0] = index * 0.02
        bridge.write_member(1, pcm(22000))
        bridge.write_member(2, pcm(22000))
        frames.append(bridge.read_input())
    samples = np.frombuffer(b"".join(frames), dtype="<i2")
    assert samples.max() > 30000
    # The low-pass filter rings at a step edge; steady samples must not wrap.
    assert samples[-960:].min() > 30000
    assert len(samples) == 48000
    bridge.close()


def test_bot_and_disallowed_members_are_excluded():
    bridge = DiscordAudioBridge(99, is_user_allowed=lambda uid: uid == 1)
    for _ in range(20):
        bridge.write_member(99, pcm(10000))
        bridge.write_member(2, pcm(10000))
        assert bridge.read_input() == bytes(960)


def test_output_split_bytes_are_preserved():
    wave = (np.sin(np.arange(4800) * 0.06) * 12000).astype("<i2").tobytes()
    whole = DiscordAudioBridge(99)
    split = DiscordAudioBridge(99)
    whole.feed_output(wave)
    for offset in range(0, len(wave), 137):
        split.feed_output(wave[offset : offset + 137])
    first = b"".join(whole.source.read() for _ in range(15))
    second = b"".join(split.source.read() for _ in range(15))
    assert first == second
    assert any(first)


def test_output_backlog_is_bounded_and_reported():
    bridge = DiscordAudioBridge(99)
    bridge.feed_output(np.ones(48000, dtype="<i2").tobytes())
    assert bridge.error
    assert bridge.stats["output_buffer_ms"] <= 1000


def test_input_backlog_is_bounded():
    bridge = DiscordAudioBridge(99)
    for _ in range(1000):
        bridge.write_member(1, pcm(1000))
    assert bridge.stats["input_dropped_frames"] > 0
    bridge.remove_user(1)
    assert bridge.read_input() == bytes(960)


def test_closed_bridge_discards_late_callbacks():
    bridge = DiscordAudioBridge(99)
    bridge.close()
    bridge.write_member(1, pcm(10000))
    bridge.feed_output(bytes(960))
    assert bridge.source.read() == b""
    assert bridge.read_input() == bytes(960)


def test_rtp_timeline_absorbs_arrival_jitter_and_duplicate_packets():
    from types import SimpleNamespace

    now = [0.0]
    bridge = DiscordAudioBridge(99, clock=lambda: now[0])
    user = SimpleNamespace(id=1, bot=False)
    for timestamp in (1000, 1960, 1960):
        packet = SimpleNamespace(
            packet=SimpleNamespace(timestamp=timestamp), pcm=pcm(1000)
        )
        bridge.sink.write(packet, user)
        now[0] += 0.005
    assert bridge.stats["received_frames"] == 2
    assert bridge.stats["input_dropped_frames"] == 1


def test_output_metrics_distinguish_generated_dequeued_and_discarded_audio():
    bridge = DiscordAudioBridge(99)
    bridge.feed_output(np.full(4800, 1000, dtype="<i2").tobytes())
    assert bridge.stats["output_generated_ms"] > 100
    assert bridge.stats["output_handed_to_discord_ms"] == 0
    assert any(bridge.source.read())
    assert bridge.stats["output_handed_to_discord_ms"] == 20
    queued = bridge.stats["output_buffer_ms"]
    bridge.close()
    assert bridge.stats["output_discarded_ms"] == queued
    bridge.close()
    assert bridge.stats["output_discarded_ms"] == queued


def test_input_clock_recovers_after_sender_catches_up_from_a_short_stall():
    now = [0.0]
    bridge = DiscordAudioBridge(99, clock=lambda: now[0])
    events = []
    for index in range(100):
        events.append((index * 0.02, 0, index))
        read_time = max(index * 0.02, 0.48) if 20 <= index <= 24 else index * 0.02
        events.append((read_time, 1, index))
    for instant, kind, index in sorted(events):
        now[0] = instant
        if kind == 0:
            bridge.write_member(1, pcm(1000), timestamp=index * 960)
        else:
            bridge.read_input()
    assert bridge.stats["received_frames"] == 100
    assert bridge.stats["input_dropped_frames"] == 0
    assert bridge.read_index == 100


def test_first_input_read_aligns_after_live_connection_startup():
    now = [0.0]
    bridge = DiscordAudioBridge(99, clock=lambda: now[0])
    now[0] = 5.0
    bridge.read_input()
    assert bridge.read_index == 251


def test_input_recovers_when_network_delay_outgrows_initial_jitter_window():
    now = [0.0]
    bridge = DiscordAudioBridge(99, clock=lambda: now[0])
    output = []
    for tick in range(160):
        now[0] = tick * 0.02
        if tick < 20:
            bridge.write_member(1, pcm(5000), timestamp=tick * 960)
        elif tick >= 30:
            # A 200 ms network-delay increase must not mute all later speech.
            bridge.write_member(1, pcm(5000), timestamp=(tick - 10) * 960)
        output.append(bridge.read_input())
    samples = np.frombuffer(b"".join(output[-50:]), dtype="<i2")
    assert np.mean(np.abs(samples)) > 4000
    assert bridge.stats["received_frames"] >= 140


def test_old_duplicate_after_playout_does_not_restart_member_timeline():
    now = [0.0]
    bridge = DiscordAudioBridge(99, clock=lambda: now[0])
    for tick in range(100):
        now[0] = tick * 0.02
        bridge.write_member(1, pcm(1000), timestamp=tick * 960)
        bridge.read_input()
    count = bridge.stats["received_frames"]
    bridge.write_member(1, pcm(1000), timestamp=0)
    assert bridge.stats["received_frames"] == count


def test_reanchoring_does_not_replay_a_previously_played_neighbor_packet():
    now = [0.0]
    bridge = DiscordAudioBridge(99, clock=lambda: now[0])
    for tick in range(31):
        now[0] = tick * 0.02
        if tick < 20:
            bridge.write_member(1, pcm(1000), timestamp=tick * 960)
        bridge.read_input()
    bridge.write_member(1, pcm(1000), timestamp=20 * 960)
    count = bridge.stats["received_frames"]
    bridge.write_member(1, pcm(1000), timestamp=19 * 960)
    assert bridge.stats["received_frames"] == count
