"""Verify the installed LivingMemory capture and storage against a temporary database."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
import tempfile
import types
from pathlib import Path


async def probe(plugin_root: Path, memory_root: Path, data_root: Path) -> None:
    """Run real storage and reflection components without modifying user memory.

    Args:
        plugin_root: This plugin's checkout.
        memory_root: Installed LivingMemory plugin source.
        data_root: Isolated temporary data directory.
    """
    from astrbot.api.message_components import Plain
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata
    from astrbot.core.platform.sources.discord.discord_platform_event import (
        DiscordPlatformEvent,
    )
    from astrbot.core.star.star_handler import star_handlers_registry

    package = types.ModuleType("discord_live_probe")
    package.__path__ = [str(plugin_root)]
    sys.modules[package.__name__] = package
    memory_package = types.ModuleType("livingmemory_probe")
    memory_package.__path__ = [str(memory_root)]
    sys.modules[memory_package.__name__] = memory_package

    bridge_module = importlib.import_module("discord_live_probe.astrbot_bridge")
    sync_module = importlib.import_module("discord_live_probe.context_sync")
    importlib.import_module("livingmemory_probe.core")
    store_class = importlib.import_module(
        "livingmemory_probe.storage.conversation_store"
    ).ConversationStore
    manager_class = importlib.import_module(
        "livingmemory_probe.core.managers.conversation_manager"
    ).ConversationManager
    config_class = importlib.import_module(
        "livingmemory_probe.core.base.config_manager"
    ).ConfigManager
    utils_class = importlib.import_module(
        "livingmemory_probe.core.event_handler_modules.message_utils"
    ).MessageUtils
    capture_class = importlib.import_module(
        "livingmemory_probe.core.event_handler_modules.group_capture"
    ).GroupCapture
    reflection_class = importlib.import_module(
        "livingmemory_probe.core.event_handler_modules.memory_reflection"
    ).MemoryReflection

    store = store_class(str(data_root / "memory.db"))
    await store.initialize()
    previous_registry = star_handlers_registry.get_handlers_by_event_type
    tasks = set()
    try:
        manager = manager_class(store)
        config = config_class()
        utils = utils_class(config, manager)
        capture = capture_class(config, manager, utils)
        reflection = reflection_class(
            None, config, None, None, manager, utils, tasks, set(), asyncio.Lock()
        )
        memory_plugin = types.SimpleNamespace(
            _run_passive_group_capture=capture.handle_all_group_messages
        )
        context = types.SimpleNamespace(
            get_config=lambda **kwargs: {"plugin_set": ["*"]},
            get_all_stars=lambda: [
                types.SimpleNamespace(
                    activated=True,
                    name="astrbot_plugin_livingmemory",
                    star_cls=memory_plugin,
                )
            ],
        )
        message = AstrBotMessage()
        message.type = MessageType.GROUP_MESSAGE
        message.self_id = "900"
        message.session_id = "123"
        message.message_id = "1"
        message.group_id = "123"
        message.sender = MessageMember("42", "Test participant")
        message.message = [Plain("Start test")]
        message.message_str = "Start test"
        message.raw_message = None
        origin = DiscordPlatformEvent(
            "Start test",
            message,
            PlatformMetadata(id="discord_probe", name="discord", description="test"),
            "123",
            types.SimpleNamespace(
                user=types.SimpleNamespace(id=900, display_name="Probe bot")
            ),
        )
        origin.plugins_name = None
        bridge = bridge_module.AstrBotBridge(context, origin, data_root)
        bridge.sync = sync_module.ContextSync(
            origin.unified_msg_origin, "test_conversation", data_root / "pending.json"
        )
        bridge.sync.add_voice(
            "test_live", "input", "user", "The project name is Aurora.", 0, 500
        )
        bridge.sync.add_voice(
            "test_live", "output", "assistant", "I remember Aurora.", 500, 900
        )
        star_handlers_registry.get_handlers_by_event_type = lambda *args, **kwargs: [
            types.SimpleNamespace(
                handler_full_name="livingmemory_reflection",
                handler=reflection.handle_memory_reflection,
            )
        ]
        await bridge.reflect_voice()
        saved = await store.get_messages(origin.unified_msg_origin)
        assert [item.role for item in saved] == ["user", "assistant"]
        assert all("Aurora" in item.content for item in saved)
        assert all(item.metadata.get("source_message_id") for item in saved)
        assert len({item.metadata["source_message_id"] for item in saved}) == 1
        assert saved[1].sender_id == "900"
        assert "delivery unconfirmed" in saved[1].content
        await store.close()
        await store.initialize()
        assert len(await store.get_messages(origin.unified_msg_origin)) == 2
        await bridge.reflect_voice()
        assert len(await store.get_messages(origin.unified_msg_origin)) == 2
        print("LIVINGMEMORY_CAPTURE_STORAGE_REOPEN_AND_RETRY_PASSED", flush=True)
    finally:
        star_handlers_registry.get_handlers_by_event_type = previous_registry
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-root", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="discord-live-memory-") as directory:
        os.environ["ASTRBOT_ROOT"] = directory
        asyncio.run(
            probe(
                Path(__file__).resolve().parents[1], args.memory_root, Path(directory)
            )
        )
