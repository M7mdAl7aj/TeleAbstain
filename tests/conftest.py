"""Skip the tests that read the 3GPP schema through git when it has not been fetched (README, step 2)."""
from pathlib import Path

import pytest

SCHEMA = Path(__file__).resolve().parent.parent / "vendor" / "MnS" / ".git"


def pytest_collection_modifyitems(config, items):
    if SCHEMA.exists():
        return
    skip = pytest.mark.skip(reason="the 3GPP schema of README step 2 is not present")
    for item in items:
        if "r19" in getattr(item, "fixturenames", ()):
            item.add_marker(skip)
