"""Continuous, bounded PCM bridges for Discord and GPT-Live."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import soxr
from discord import AudioSource
from discord.sinks import Sink


class StreamingSink(Sink):
    """Forward decoded member audio to the bridge without retaining recordings."""

    def __init__(self, bridge: DiscordAudioBridge) -> None:
        super().__init__()
        self.bridge = bridge

    def write(self, data, user) -> None:
        """Accept Pycord voice packets.

        Args:
            data: VoiceData containing decoded PCM, or PCM bytes.
            user: Member or user associated with the packet.
        """
        if user is None or getattr(user, "bot", False):
            return
        user_id = int(getattr(user, "id", user))
        self.bridge.write_member(
            user_id,
            getattr(data, "pcm", data),
            timestamp=getattr(getattr(data, "packet", None), "timestamp", None),
        )

    def cleanup(self) -> None:
        self.finished = True


class PlaybackSource(AudioSource):
    """Expose fixed-size Discord frames from the bounded output buffer."""

    def __init__(self, bridge: DiscordAudioBridge) -> None:
        self.bridge = bridge
        self.started = False

    def read(self) -> bytes:
        """Return a frame, silence during starvation, or EOF after shutdown."""
        bridge = self.bridge
        with bridge.lock:
            if bridge.closed:
                return b""
            if not self.started:
                if len(bridge.output) < 3840 * 5:
                    return bytes(3840)
                self.started = True
            if len(bridge.output) < 3840:
                bridge.counters["output_underruns"] += 1
                return bytes(3840)
            frame = bytes(bridge.output[:3840])
            del bridge.output[:3840]
            bridge.counters["played_frames"] += 1
            return frame

    def cleanup(self) -> None:
        self.bridge.close()


class DiscordAudioBridge:
    """Mix members on a local timeline and resample both continuous streams.

    Args:
        bot_user_id: Bot account excluded from audio input.
        clock: Monotonic time source.
        is_user_allowed: Optional member filter, called from the receive thread.
    """

    def __init__(
        self,
        bot_user_id: int,
        clock: Callable[[], float] = time.monotonic,
        is_user_allowed: Callable[[int], bool] | None = None,
    ) -> None:
        self.bot_user_id = bot_user_id
        self.clock = clock
        self.origin = clock()
        self.is_user_allowed = is_user_allowed
        self.lock = threading.RLock()
        self.closed = False
        self.error: str | None = None
        self.members: dict[int, dict[int, np.ndarray]] = {}
        self.member_timebase: dict[int, tuple[int, int]] = {}
        self.member_last_timestamp: dict[int, int] = {}
        self.member_pending: dict[int, bytearray] = {}
        self.member_next: dict[int, int] = {}
        self.read_index = 0
        self.output = bytearray()
        self.output_pending = bytearray()
        # Reserve two frames for the streaming resampler's filter delay.
        self.input_resampled = bytearray(960 * 2)
        self.input_resampler = soxr.ResampleStream(
            48000, 24000, 1, dtype="float32", quality="LQ"
        )
        self.output_resampler = soxr.ResampleStream(
            24000, 48000, 1, dtype="float32", quality="LQ"
        )
        self.counters = {
            "received_frames": 0,
            "input_dropped_frames": 0,
            "input_late_frames": 0,
            "input_future_frames": 0,
            "input_duplicate_frames": 0,
            "input_resyncs": 0,
            "input_mixed_frames": 0,
            "output_underruns": 0,
            "played_frames": 0,
            "output_generated_bytes": 0,
            "output_discarded_bytes": 0,
        }
        self.sink = StreamingSink(self)
        self.source = PlaybackSource(self)

    @property
    def stats(self) -> dict[str, int | float]:
        """Return an atomic metrics snapshot."""
        with self.lock:
            return {
                **self.counters,
                "output_buffer_ms": len(self.output) / 192,
                "output_generated_ms": self.counters["output_generated_bytes"] / 192,
                "output_handed_to_discord_ms": self.counters["played_frames"] * 20,
                "output_discarded_ms": self.counters["output_discarded_bytes"] / 192,
                "input_members": len(self.members),
            }

    def write_member(
        self, user_id: int, pcm: bytes, timestamp: int | None = None
    ) -> None:
        """Queue complete stereo frames with bounded per-member jitter.

        Args:
            user_id: Discord member ID.
            pcm: Signed PCM16LE, stereo, 48 kHz.
            timestamp: Optional RTP sample timestamp for jitter and duplicate handling.
        """
        if user_id == self.bot_user_id:
            return
        if self.is_user_allowed and not self.is_user_allowed(user_id):
            return
        with self.lock:
            if self.closed:
                return
            pending = self.member_pending.setdefault(user_id, bytearray())
            pending.extend(pcm)
            queue = self.members.setdefault(user_id, {})
            # Three frames absorb arrival jitter before the shared playout clock.
            arrival = round((self.clock() - self.origin) / 0.02) + 3
            if timestamp is None:
                index = max(arrival, self.member_next.get(user_id, arrival))
            else:
                base_stamp, base_index = self.member_timebase.setdefault(
                    user_id, (timestamp, arrival)
                )
                delta = (timestamp - base_stamp + (1 << 31)) % (1 << 32) - (1 << 31)
                # Once reanchored, packets preceding that RTP boundary belong to
                # the old playout timeline and must never be scheduled again.
                if delta < 0:
                    dropped = len(pending) // 3840
                    self.counters["input_dropped_frames"] += dropped
                    self.counters["input_late_frames"] += dropped
                    pending.clear()
                    return
                index = base_index + delta // 960
                previous = self.member_last_timestamp.get(user_id)
                forward = (
                    previous is None
                    or ((timestamp - previous + (1 << 31)) % (1 << 32) - (1 << 31)) > 0
                )
                # A route-delay change or resumed talkspurt can shift all later RTP
                # packets behind playout. Reanchor fresh packets instead of muting
                # the speaker indefinitely; old/duplicate packets cannot move it.
                if forward:
                    self.member_last_timestamp[user_id] = timestamp
                    if index < self.read_index or abs(index - arrival) > 50:
                        index = max(arrival, self.read_index + 3)
                        self.member_timebase[user_id] = (timestamp, index)
                        self.counters["input_resyncs"] += 1
            while len(pending) >= 3840:
                frame = bytes(pending[:3840])
                del pending[:3840]
                if (
                    index < self.read_index
                    or index >= self.read_index + 50
                    or len(queue) >= 50
                    or index in queue
                ):
                    self.counters["input_dropped_frames"] += 1
                    if index < self.read_index:
                        self.counters["input_late_frames"] += 1
                    elif index >= self.read_index + 50:
                        self.counters["input_future_frames"] += 1
                    elif index in queue:
                        self.counters["input_duplicate_frames"] += 1
                    index += 1
                    continue
                samples = np.frombuffer(frame, dtype="<i2").reshape(-1, 2)
                queue[index] = samples.astype(np.float32).mean(axis=1)
                self.counters["received_frames"] += 1
                index += 1
            self.member_next[user_id] = index

    def read_input(self) -> bytes:
        """Mix one twenty-millisecond frame for a real-time-paced sender."""
        with self.lock:
            if self.closed:
                return bytes(960)
            # Align once after startup; catch-up reads must consume consecutive frames.
            if self.read_index == 0:
                self.read_index = round((self.clock() - self.origin) / 0.02)
            index = self.read_index
            mixed = np.zeros(960, dtype=np.float32)
            for queue in self.members.values():
                for expired in [position for position in queue if position < index]:
                    del queue[expired]
                    self.counters["input_dropped_frames"] += 1
                    self.counters["input_late_frames"] += 1
                samples = queue.pop(index, None)
                if samples is not None:
                    mixed += samples
                    self.counters["input_mixed_frames"] += 1
            self.read_index = index + 1
            # Leave filter headroom; clip again after resampling.
            np.clip(mixed, -32768, 32767, out=mixed)
            converted = self.input_resampler.resample_chunk(mixed)
            self.input_resampled.extend(
                np.clip(converted, -32768, 32767).astype("<i2").tobytes()
            )
            frame = bytes(self.input_resampled[:960])
            del self.input_resampled[:960]
            return frame.ljust(960, b"\0")

    def feed_output(self, pcm: bytes) -> None:
        """Resample arbitrary Live chunks, preserving partial samples.

        Args:
            pcm: Signed PCM16LE, mono, 24 kHz.
        """
        with self.lock:
            if self.closed or self.error:
                return
            self.output_pending.extend(pcm)
            complete = len(self.output_pending) // 2 * 2
            if not complete:
                return
            samples = np.frombuffer(
                bytes(self.output_pending[:complete]), dtype="<i2"
            ).astype(np.float32)
            del self.output_pending[:complete]
            converted = self.output_resampler.resample_chunk(samples)
            mono = np.clip(converted, -32768, 32767).astype("<i2")
            stereo = np.repeat(mono[:, None], 2, axis=1).tobytes()
            self.counters["output_generated_bytes"] += len(stereo)
            available = 192000 - len(self.output)
            self.output.extend(stereo[:available])
            if len(stereo) > available:
                self.counters["output_discarded_bytes"] += len(stereo) - available
                self.error = "Live playback backlog exceeded one second."

    def remove_user(self, user_id: int) -> None:
        """Release buffers when a member leaves the voice channel."""
        with self.lock:
            self.members.pop(user_id, None)
            self.member_pending.pop(user_id, None)
            self.member_next.pop(user_id, None)
            self.member_timebase.pop(user_id, None)
            self.member_last_timestamp.pop(user_id, None)

    def close(self) -> None:
        """Release buffers and make late callbacks harmless."""
        with self.lock:
            self.closed = True
            self.members.clear()
            self.member_pending.clear()
            self.member_next.clear()
            self.member_timebase.clear()
            self.member_last_timestamp.clear()
            self.counters["output_discarded_bytes"] += len(self.output)
            self.output.clear()
            self.output_pending.clear()
            self.input_resampled.clear()
