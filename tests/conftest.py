"""Load the plugin package directly from its checkout."""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("discord_gpt_live")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("discord_gpt_live", package)
