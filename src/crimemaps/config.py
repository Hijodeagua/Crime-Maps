"""
City configuration registry.

All city-specific constants live here as CityConfig instances. Downstream code always
reads from a selected CityConfig — never from globals — so adding a new city is a single
registry entry without touching any other module.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FieldMapping:
    """Maps ArcGIS layer field names to canonical names for a given city/source."""
    occurrence_date: str          # preferred event-time field
    report_date: str              # fallback if occurrence_date is null per-record
    lat: str
    lon: str
    offense: str                  # human-readable offense description
    nibrs_code: str               # NIBRS code field
    division: Optional[str]       # police division/district (nullable)
    record_id: Optional[str]      # source unique identifier (nullable)


@dataclass(frozen=True)
class CityConfig:
    """All city-specific parameters. One instance per city in CITIES."""
    slug: str
    name: str

    # ArcGIS REST endpoints
    query_url: str
    layer_meta_url: str

    # ArcGIS field mapping
    field_mapping: FieldMapping

    # Census geography
    census_state_fips: str        # e.g. "37" for NC
    census_county_fips: str       # e.g. "119" for Mecklenburg
    tract_vintage: int            # e.g. 2020 — must match boundary file and ACS vintage
    acs_release: str              # e.g. "2018-2022" — ACS 5-year release label
    # Census cartographic boundary (TIGER) URL for tract polygons (GeoJSON)
    # Format: will be fetched and cached; can be None to force use of bundled fixture
    boundary_url: Optional[str]

    # Coordinate reference systems
    wgs84_crs: str = "EPSG:4326"
    planar_crs: str = "EPSG:32617"  # UTM Zone 17N — covers Charlotte; set per city

    # Folium map defaults
    map_center: Tuple[float, float] = (35.2271, -80.8431)
    map_zoom: int = 11

    # NIBRS classification
    exclude_800_series: bool = True   # exclude non-criminal 800-series codes
    nibrs_groups: Dict[str, List[str]] = field(default_factory=dict)
    # nibrs_groups maps category label -> list of NIBRS code prefixes/values

    # Per-capita rate suppression
    # Tracts with residential population below this threshold get their rate
    # suppressed (shown as flagged) to prevent extreme rates from tiny tracts
    # (e.g. airport parcels, industrial zones). Daytime vs. residential population
    # is NOT corrected in v1 — see README for denominator caveats.
    min_population_for_rate: int = 100

    # Date field preference: "occurrence" or "report"
    date_field_preference: str = "occurrence"


# ---------------------------------------------------------------------------
# Charlotte, NC — CMPD Incidents
# ---------------------------------------------------------------------------
_CHARLOTTE_NIBRS_GROUPS: Dict[str, List[str]] = {
    "Violent": [
        "09A", "09B", "09C",  # Murder/Homicide
        "11A", "11B", "11C", "11D",  # Sex offenses
        "120",  # Robbery
        "13A", "13B", "13C",  # Assault
        "100",  # Kidnapping
    ],
    "Property": [
        "200",  # Arson
        "210",  # Extortion
        "220",  # Burglary
        "23A", "23B", "23C", "23D", "23E", "23F", "23G", "23H",  # Larceny
        "240",  # Motor vehicle theft
        "250",  # Embezzlement
        "26A", "26B", "26C", "26D", "26E",  # Fraud
        "270",  # Bribery
        "280",  # Receiving stolen property
        "290",  # Destruction/damage/vandalism
    ],
    "Other": [],  # populated at query time as "not Violent and not Property"
}

CHARLOTTE = CityConfig(
    slug="charlotte",
    name="Charlotte, NC",
    query_url=(
        "https://gis.charlottenc.gov/arcgis/rest/services/CMPD/"
        "CMPDIncidents/MapServer/0/query"
    ),
    layer_meta_url=(
        "https://gis.charlottenc.gov/arcgis/rest/services/CMPD/"
        "CMPDIncidents/MapServer/0?f=json"
    ),
    field_mapping=FieldMapping(
        occurrence_date="DATE_INCIDENT_BEGAN",
        report_date="DATE_REPORTED",
        lat="Latitude",
        lon="Longitude",
        offense="HIGHEST_NIBRS_DESCRIPTION",
        nibrs_code="HIGHEST_NIBRS_CODE",
        division="DIVISION_ID",
        record_id="OBJECTID",
    ),
    census_state_fips="37",
    census_county_fips="119",
    tract_vintage=2020,
    acs_release="2018-2022",
    # Census Bureau cartographic boundary — 2020 tracts, Mecklenburg County
    boundary_url=(
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
        "tigerWMS_Census2020/MapServer/8/query"
        "?where=STATE%3D%2237%27+AND+COUNTY%3D%27119%27"
        "&outFields=GEOID%2CNAME&f=geojson&returnGeometry=true"
    ),
    planar_crs="EPSG:32617",
    map_center=(35.2271, -80.8431),
    map_zoom=11,
    exclude_800_series=True,
    nibrs_groups=_CHARLOTTE_NIBRS_GROUPS,
    min_population_for_rate=100,
    date_field_preference="occurrence",
)

# Registry — keyed by slug
CITIES: Dict[str, CityConfig] = {
    CHARLOTTE.slug: CHARLOTTE,
}
