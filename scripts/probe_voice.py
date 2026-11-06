"""Probe DAVE receive with two configured test bots and a quiet reference tone."""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import discord
import numpy as np
from discord.sinks import Sink
from discord.voice import VoiceClient


class ProbeSink(Sink):
    def __init__(self):
        super().__init__()
        self.frames = 0
        self.audible = 0
        self.users = set()

    def write(self, data, user):
        raw = getattr(data, "pcm", data)
        self.frames += 1
        if (
            raw
            and np.max(np.abs(np.frombuffer(raw, dtype="<i2").astype(np.int32))) > 100
        ):
            self.audible += 1
        if user:
            self.users.add(getattr(user, "id", user))

    def cleanup(self):
        self.finished = True


class Tone(discord.AudioSource):
    def __init__(self):
        self.index = 0

    def read(self):
        if self.index >= 250:
            return b""
        phase = np.arange(960) + self.index * 960
        self.index += 1
        samples = (np.sin(phase * 2 * np.pi * 440 / 48000) * 1500).astype("<i2")
        return np.repeat(samples[:, None], 2, axis=1).tobytes()


async def probe(config_path, guild_id, channel_id):
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    platforms = [
        p for p in config["platform"] if p.get("type") == "discord" and p.get("enable")
    ][:2]
    if len(platforms) != 2:
        raise ValueError("Two configured test bots are required.")
    clients, tasks, voices, sinks = [], [], [], []
    try:
        for platform in platforms:
            client = discord.Client(
                intents=discord.Intents.default(),
                proxy=platform.get("discord_proxy") or None,
            )
            clients.append(client)
            await client.login(platform["discord_token"])
            tasks.append(asyncio.create_task(client.connect(reconnect=False)))
            await asyncio.wait_for(client.wait_until_ready(), 30)
            guild = client.get_guild(guild_id)
            if guild is None or guild.me.voice:
                raise RuntimeError("Test guild unavailable or bot already connected.")
            voice = await guild.get_channel(channel_id).connect(
                cls=VoiceClient, timeout=30
            )
            voices.append(voice)
            print("transport_connected", platform["id"], flush=True)
        # An empty channel may not establish group keys until a peer joins.
        for platform, voice in zip(platforms, voices):
            deadline = asyncio.get_running_loop().time() + 15
            state = voice._connection
            while state.dave_protocol_version > 0 and (
                state.dave_session is None or not state.dave_session.ready
            ):
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("DAVE encryption keys did not become ready.")
                await asyncio.sleep(0.05)
            sink = ProbeSink()
            sinks.append(sink)
            voice.start_recording(sink)
            print(
                "connected",
                platform["id"],
                "dave_ready",
                bool(state.dave_session and state.dave_session.ready),
                flush=True,
            )
        for voice in voices:
            voice.play(Tone())
            await asyncio.sleep(6)
        results = [
            {"frames": s.frames, "audible_frames": s.audible, "speakers": len(s.users)}
            for s in sinks
        ]
        print(json.dumps(results), flush=True)
        if not all(s.audible > 50 for s in sinks):
            raise RuntimeError("DAVE receive probe did not capture sufficient audio.")
    finally:
        for voice in voices:
            try:
                if voice.is_recording():
                    voice.stop_recording()
                await voice.disconnect(force=True)
            except Exception as exc:
                print("cleanup", type(exc).__name__, flush=True)
        for client in clients:
            await client.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--guild", type=int, required=True)
    parser.add_argument("--channel", type=int, required=True)
    args = parser.parse_args()
    logging.getLogger("discord").setLevel(logging.ERROR)
    asyncio.run(probe(args.config, args.guild, args.channel))
