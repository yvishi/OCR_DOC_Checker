import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

# Keep pytest output clean: tests that want spans inject an in-memory exporter.
os.environ.setdefault("TELEMETRY_EXPORTER", "none")
