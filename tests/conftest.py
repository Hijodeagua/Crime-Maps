"""Shared fixtures for all tests."""

import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd
import pytest

from crimemaps.config import CHARLOTTE


@pytest.fixture
def city():
    return CHARLOTTE


@pytest.fixture
def date_range():
    start = pd.Timestamp("2024-01-01", tz="America/New_York")
    end = pd.Timestamp("2024-01-31", tz="America/New_York")
    return start, end
