"""Check real Live startup with synthetic group history and existing credentials."""

import argparse
import asyncio
import json
import sys
import types
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("discord_gpt_live")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("discord_gpt_live", package)

from discord_gpt_live.astrbot_bridge import resolve_live_provider  # noqa: E402
from discord_gpt_live.audio_bridge import DiscordAudioBridge  # noqa: E402
from discord_gpt_live.voice_session import LiveSession  # noqa: E402


async def probe(config_path: Path, provider_id: str, input_wav: Path | None) -> None:
    """Open a short synthetic session and validate history, policy and cleanup.

    Args:
        config_path: Local AstrBot configuration containing the selected voice model.
        provider_id: Selected AstrBot voice model configuration ID.
        input_wav: Optional PCM48k stereo fixture asking for the synthetic project code.
    """
    fixture = b""
    if input_wav is not None:
        with wave.open(str(input_wav), "rb") as recording:
            if (
                recording.getnchannels() != 2
                or recording.getframerate() != 48000
                or recording.getsampwidth() != 2
                or recording.getnframes() > 30 * 48000
            ):
                raise ValueError(
                    "Use a PCM16 stereo 48 kHz fixture of at most 30 seconds."
                )
            fixture = recording.readframes(recording.getnframes())
    transcripts = []
    delegations = []
    audible_frames = 0
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    credentials = resolve_live_provider(config, provider_id)
    bridge = DiscordAudioBridge(0)

    async def delegate(event):
        input_end = max(
            (
                entry["end_ms"]
                for entry in transcripts
                if entry["type"] == "session.input_transcript.delta"
            ),
            default=0,
        )
        delegations.append(
            {"offset_ms": event["offset_ms"], "latest_input_end_ms": input_end}
        )
        return "The verified synthetic shared-memory project code is Aurora seventeen."

    async def consume():
        nonlocal audible_frames
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        while not bridge.closed:
            if any(bridge.source.read()):
                audible_frames += 1
            deadline += 0.02
            await asyncio.sleep(max(0, deadline - loop.time()))

    live = LiveSession(
        api_key=credentials.api_key,
        model=credentials.model,
        websocket_url=credentials.websocket_url,
        headers=credentials.headers,
        voice="marin",
        instructions=(
            "This is a brief connection test. Wait for voice input. "
            "When asked for the project code in shared memory, Generate <|SpawnThinking|>. "
            "Speak the exact verified project code returned by the backend."
        ),
        history=[
            {"role": "user", "content": "Synthetic group project name: Aurora."},
            {"role": "assistant", "content": "I remember the name Aurora."},
            {"role": "user", "content": "This test uses synthetic context only."},
        ],
        bridge=bridge,
        on_transcript=transcripts.append,
        on_delegate=delegate,
    )
    playback = asyncio.create_task(consume())
    try:
        try:
            await asyncio.wait_for(live.start(), 45)
        except Exception:
            print(f"STARTUP_STATUS {live.error or 'connection failed'}")
            raise
        print("LIVE_STARTED", flush=True)
        await live.append_context(
            "Synthetic background context: the test is in progress."
        )
        print("CONTEXT_ACKNOWLEDGED", flush=True)
        await live.update_text_policy(True)
        await live.update_text_policy(False)
        if fixture:
            loop = asyncio.get_running_loop()
            deadline = loop.time()
            for offset in range(0, len(fixture), 3840):
                bridge.write_member(
                    1, fixture[offset : offset + 3840].ljust(3840, b"\0"), offset // 4
                )
                deadline += 0.02
                await asyncio.sleep(max(0, deadline - loop.time()))
            await asyncio.sleep(8)
        else:
            await asyncio.sleep(3)
        if live.error or bridge.error:
            print(
                f"TRANSPORT_STATUS live={live.error} audio={bridge.error} stats={bridge.stats}",
                flush=True,
            )
            raise RuntimeError("Live transport or audio processing failed.")
    finally:
        await live.close()
        playback.cancel()
        await asyncio.gather(playback, return_exceptions=True)
    print(
        f"FINAL_STATUS error={live.error} usage_received={live.usage is not None}",
        flush=True,
    )
    if live.error or live.usage is None:
        raise RuntimeError("Live final usage was not confirmed.")
    if fixture:
        inputs = [
            entry
            for entry in transcripts
            if entry["type"] == "session.input_transcript.delta"
        ]
        spoken = "".join(
            entry["delta"]
            for entry in transcripts
            if entry["type"] == "session.output_transcript.delta"
        ).lower()
        code_received = "aurora" in spoken and ("seventeen" in spoken or "17" in spoken)
        print(
            json.dumps(
                {
                    "input_fragments": len(inputs),
                    "delegations": delegations,
                    "audible_output_frames": audible_frames,
                    "verified_code_spoken": code_received,
                }
            )
        )
        if not (inputs and delegations and audible_frames and code_received):
            raise RuntimeError("Synthetic spoken delegation round trip was incomplete.")
        print("LIVE_SPOKEN_INPUT_DELEGATION_AND_AUDIO_OUTPUT_PASSED")
    print("LIVE_NONEMPTY_HISTORY_CONTEXT_POLICY_AND_CLOSE_PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provider-id", required=True)
    parser.add_argument("--input-wav", type=Path)
    args = parser.parse_args()
    try:
        asyncio.run(probe(args.config, args.provider_id, args.input_wav))
    except Exception as exc:
        # Exclude credential-bearing network exceptions and service bodies.
        print(f"LIVE_PROBE_FAILED ({type(exc).__name__})", file=sys.stderr)
        sys.exit(1)
