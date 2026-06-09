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
class CFSFieldMapping:
    """Maps calls-for-service layer fields to canonical names.

    Any field other than event_datetime/call_type may be None when the layer
    doesn't publish it (e.g. CMPD CFS publishes block-level addresses, not
    coordinates). The CFS source also resolves fields heuristically against
    live layer metadata, so these are best-effort defaults.
    """
    event_datetime: str
    call_type: str
    address: Optional[str] = None
    division: Optional[str] = None
    record_id: Optional[str] = None
    lat: Optional[str] = None
    lon: Optional[str] = None


@dataclass(frozen=True)
class CFSConfig:
    """Calls-for-service (911 dispatch) ArcGIS endpoint for a city."""
    query_url: str
    layer_meta_url: str
    field_mapping: CFSFieldMapping


@dataclass(frozen=True)
class ScannerFeed:
    """A link to live scanner audio for the city (not ingested — audio streams
    have no structured API; calls-for-service data is the machine-readable
    counterpart)."""
    name: str
    url: str
    note: str = ""


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

    # Cache slug for the incident source (kept as "cmpd_incidents" for Charlotte
    # so existing snapshots remain readable; new cities use "police_incidents")
    incident_source_slug: str = "police_incidents"

    # How far back the UI allows date-range selection. Three years balances
    # usefulness against ArcGIS fetch size (~100k+ records for Charlotte) and
    # the growing effect of upstream reclassification on older records.
    max_history_days: int = 365 * 3

    # Calls-for-service (911 dispatch) endpoint — the structured counterpart to
    # scanner traffic. None when the city doesn't publish one.
    cfs: Optional[CFSConfig] = None

    # Live scanner-audio links shown in the UI (audio is not ingested)
    scanner_feeds: Tuple[ScannerFeed, ...] = ()

    # Demo-source cluster centers: (lat, lon, weight). Empty → demo source
    # synthesizes clusters around map_center.
    demo_clusters: Tuple[Tuple[float, float, float], ...] = ()


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

# Rough cluster centers for the synthetic demo source: (lat, lon, weight)
_CHARLOTTE_DEMO_CLUSTERS: Tuple[Tuple[float, float, float], ...] = (
    (35.2271, -80.8431, 0.15),   # Uptown
    (35.1988, -80.8273, 0.10),   # South End / Dilworth
    (35.2560, -80.8070, 0.08),   # NoDa / Plaza Midwood
    (35.1720, -80.8700, 0.08),   # South Park
    (35.3140, -80.7510, 0.07),   # University City
    (35.2090, -80.9150, 0.07),   # Steele Creek / Berewick
    (35.2750, -80.8950, 0.07),   # Toringdon / Ballantyne W
    (35.1540, -80.8020, 0.07),   # Ballantyne
    (35.2840, -80.7940, 0.06),   # Concord Mills area
    (35.2430, -80.7640, 0.06),   # Mint Hill
    (35.3500, -80.8600, 0.06),   # Huntersville
    (35.1050, -80.8700, 0.06),   # Pineville
    (35.2300, -80.9400, 0.07),   # Airport / Berewick
)

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
    incident_source_slug="cmpd_incidents",
    # CMPD Calls for Service — hub page:
    # https://data.charlottenc.gov/datasets/cmpd-calls-for-service
    # Endpoint and field names are best-effort and validated against live layer
    # metadata at fetch time (the CFS source resolves fields heuristically and
    # falls back to demo data if the layer is unreachable).
    cfs=CFSConfig(
        query_url=(
            "https://gis.charlottenc.gov/arcgis/rest/services/CMPD/"
            "CMPDCallsForService/MapServer/0/query"
        ),
        layer_meta_url=(
            "https://gis.charlottenc.gov/arcgis/rest/services/CMPD/"
            "CMPDCallsForService/MapServer/0?f=json"
        ),
        field_mapping=CFSFieldMapping(
            event_datetime="CALENDAR_DATE",
            call_type="CALL_TYPE",
            address="ADDRESS",
            division="DIVISION",
            record_id="OBJECTID",
        ),
    ),
    scanner_feeds=(
        ScannerFeed(
            name="Broadcastify — North Carolina directory",
            url="https://www.broadcastify.com/listen/stid/34",
            note="Find Mecklenburg County / CMPD dispatch audio feeds.",
        ),
        ScannerFeed(
            name="OpenMHz — system search",
            url="https://openmhz.com/systems",
            note="Trunked-radio archives; search for Charlotte/Mecklenburg.",
        ),
    ),
    demo_clusters=_CHARLOTTE_DEMO_CLUSTERS,
)

# ---------------------------------------------------------------------------
# Raleigh, NC — Raleigh Police Incidents (NIBRS)
#
# EXPERIMENTAL: endpoint and field names are best-effort from the Raleigh open
# data portal (https://data.raleighnc.gov) and are validated against live layer
# metadata at fetch time. If the live layer disagrees, the app falls back to
# snapshot/demo tiers and the data-source banner says so.
# ---------------------------------------------------------------------------
RALEIGH = CityConfig(
    slug="raleigh",
    name="Raleigh, NC",
    query_url=(
        "https://services.arcgis.com/v400IkDOw1ad7Yad/arcgis/rest/services/"
        "Police_Incidents/FeatureServer/0/query"
    ),
    layer_meta_url=(
        "https://services.arcgis.com/v400IkDOw1ad7Yad/arcgis/rest/services/"
        "Police_Incidents/FeatureServer/0?f=json"
    ),
    field_mapping=FieldMapping(
        occurrence_date="reported_date",
        report_date="reported_date",
        lat="latitude",
        lon="longitude",
        offense="crime_description",
        nibrs_code="crime_code",
        division="district",
        record_id="OBJECTID",
    ),
    census_state_fips="37",
    census_county_fips="183",   # Wake County
    tract_vintage=2020,
    acs_release="2018-2022",
    boundary_url=(
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
        "tigerWMS_Census2020/MapServer/8/query"
        "?where=STATE%3D%2237%27+AND+COUNTY%3D%27183%27"
        "&outFields=GEOID%2CNAME&f=geojson&returnGeometry=true"
    ),
    planar_crs="EPSG:32617",    # UTM 17N also covers Raleigh (-78.64°)
    map_center=(35.7796, -78.6382),
    map_zoom=11,
    exclude_800_series=True,
    nibrs_groups=_CHARLOTTE_NIBRS_GROUPS,  # Raleigh crime_code is NIBRS-based
    min_population_for_rate=100,
    date_field_preference="report",  # layer publishes reported_date only
    incident_source_slug="police_incidents",
    scanner_feeds=(
        ScannerFeed(
            name="Broadcastify — North Carolina directory",
            url="https://www.broadcastify.com/listen/stid/34",
            note="Find Wake County / Raleigh PD dispatch audio feeds.",
        ),
    ),
    demo_clusters=(
        (35.7796, -78.6382, 0.18),  # Downtown
        (35.7872, -78.6705, 0.10),  # NC State / Hillsborough St
        (35.8324, -78.6429, 0.09),  # North Hills
        (35.7956, -78.7050, 0.08),  # Crabtree
        (35.7327, -78.6280, 0.08),  # Garner border / South Raleigh
        (35.8585, -78.5790, 0.07),  # Triangle Town Center
        (35.7680, -78.5510, 0.07),  # New Bern Ave corridor / East Raleigh
        (35.8230, -78.7290, 0.06),  # Northwest Raleigh
    ),
)

# Registry — keyed by slug
CITIES: Dict[str, CityConfig] = {
    CHARLOTTE.slug: CHARLOTTE,
    RALEIGH.slug: RALEIGH,
}
