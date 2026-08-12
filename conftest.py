"""pytest configuration — make the repository root importable.

The ``scripts/`` directory contains standalone CLI scripts (not an installed
package), and several tests import from it (e.g. ``from scripts.build_sft_v2
import ...``).  ``uv run pytest`` does not add the current directory to
``sys.path`` the way ``python -m pytest`` does, so this conftest inserts the
repository root so those imports resolve regardless of invocation style.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
