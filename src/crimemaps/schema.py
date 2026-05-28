"""
Canonical data contract for the crime-maps project.

Every measure source — regardless of origin — normalizes into a single tidy DataFrame
conforming to this schema. CMPD incidents are the first source; future sources
(victimization surveys, ED admissions, 911/311 calls, etc.) drop in via the same contract.

Column semantics
----------------
measure_source   : slug identifying the data source (e.g. "cmpd_incidents")
datetime         : timezone-aware event timestamp (America/New_York); authoritative time
geography_id     : 11-digit census-tract GEOID, or UNASSIGNED for unmatched points
lat, lon         : WGS-84 point coordinates (nullable for pre-aggregated sources)
category         : normalized crime group — "Violent", "Property", "Other", or "Unknown"
nibrs_code       : source-native NIBRS code where applicable (nullable)
value            : 1.0 for every point-level incident; pre-aggregated sources may carry
                   other values — do NOT mix semantics within a single measure_source pull
source_record_id : original record identifier for deduplication and traceback (nullable)
retrieved_at     : tz-aware timestamp of the data pull (provenance)
"""

import pandas as pd

# Sentinel used when a point cannot be joined to any census tract.
UNASSIGNED = "UNASSIGNED"

COLUMNS = [
    "measure_source",
    "datetime",
    "geography_id",
    "lat",
    "lon",
    "category",
    "nibrs_code",
    "value",
    "source_record_id",
    "retrieved_at",
]

DTYPES = {
    "measure_source": "string",
    "geography_id": "string",
    "lat": "float64",
    "lon": "float64",
    "category": "string",
    "nibrs_code": "string",
    "value": "float64",
    "source_record_id": "string",
}

CATEGORIES = ("Violent", "Property", "Other", "Unknown")


def empty() -> pd.DataFrame:
    """Return an empty DataFrame conforming to the canonical schema."""
    df = pd.DataFrame(columns=COLUMNS)
    return df.astype({k: v for k, v in DTYPES.items() if k in df.columns})


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Assert required columns are present and value is non-negative. Returns df."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Schema violation — missing columns: {missing}")
    if (df["value"] < 0).any():
        raise ValueError("Schema violation — value column contains negative entries")
    return df
