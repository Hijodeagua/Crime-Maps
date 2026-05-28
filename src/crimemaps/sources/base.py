"""Abstract base class for all measure sources."""

from abc import ABC, abstractmethod

import pandas as pd

from crimemaps import schema
from crimemaps.config import CityConfig


class MeasureSource(ABC):
    """
    Every measure source subclasses this and implements fetch().
    fetch() must return a DataFrame conforming to schema.COLUMNS.
    """

    def __init__(self, city: CityConfig):
        self.city = city

    @abstractmethod
    def fetch(
        self,
        start: "pd.Timestamp",
        end: "pd.Timestamp",
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch data for the given date range and return a canonical DataFrame."""
        ...

    @property
    @abstractmethod
    def source_slug(self) -> str:
        """Unique slug for this source, used as measure_source column value."""
        ...

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate schema and ensure correct dtypes. Call at end of fetch()."""
        for col in schema.COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[schema.COLUMNS]
        schema.validate(df)
        return df
