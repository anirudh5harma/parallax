from __future__ import annotations

import sys
from pathlib import Path

package_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(package_root))

from deep_research.cli import worker_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(worker_main())
