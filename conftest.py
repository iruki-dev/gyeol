"""Make the reference service importable in the test-suite without installing it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "reference_service"))
