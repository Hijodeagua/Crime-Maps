"""
Crime Maps — Charlotte NC interactive dashboard.

Run with:  streamlit run src/crimemaps/app.py
"""

import logging
import random
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Make project importable when run from repo root or src/
sys.path.insert(0, str(Path(__file__).parents[1]))

import crimemaps.hotspots as hotspots
import crimemaps.intensity as intensity
from crimemaps import aggregate, cache, streetmap
from crimemaps.config import CITIES, CityConfig
from crimemaps.geography import assign_geography, load_boundaries
from crimemaps.loader import DataSourceInfo, load, load_cfs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Crime Maps",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🗺️ Crime Maps")
    st.caption("Incident data as one measure source among many.")

    # City selector
    city_slug = st.selectbox(
        "City",
        options=list(CITIES.keys()),
        format_func=lambda s: CITIES[s].name,
    )
    city: CityConfig = CITIES[city_slug]

    st.divider()

    # Date range — quick presets over a rolling window capped at max_history_days
    st.subheader("Date range")
    today = pd.Timestamp.now().normalize()
    earliest = today - pd.Timedelta(days=city.max_history_days)

    PRESETS = {
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 90 days": 90,
        "Last 6 months": 180,
        "Past year": 365,
        "Past 3 years": city.max_history_days,
        "Custom range": None,
    }
    preset = st.selectbox("Period", options=list(PRESETS.keys()), index=3)

    if PRESETS[preset] is not None:
        end_date = today.date()
        start_date = (today - pd.Timedelta(days=PRESETS[preset])).date()
        st.caption(f"{start_date} → {end_date}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From",
                value=(today - pd.Timedelta(days=180)).date(),
                min_value=earliest.date(),
                max_value=today.date(),
            )
        with col2:
            end_date = st.date_input(
                "To",
                value=today.date(),
                min_value=earliest.date(),
                max_value=today.date(),
            )

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        st.error("Start date must be before end date.")
        st.stop()

    st.caption(
        f"History is capped at {city.max_history_days // 365} years: older records "
        "are increasingly affected by upstream reclassification, and multi-year "
        "pulls are large. Wide ranges are fetched once, snapshotted, then served "
        "from disk."
    )

    prefer_cache = st.toggle(
        "Prefer cached data (faster)",
        value=True,
        help=(
            "Serve the request from an existing snapshot that covers the selected "
            "range instead of re-querying the live API. Snapshots are refreshed "
            "nightly by a scheduled job. Turn off to force a live API pull."
        ),
    )

    st.divider()

    # Crime category filter
    st.subheader("Category filter")
    all_categories = ("Violent", "Property", "Other", "Unknown")
    selected_categories = st.multiselect(
        "Crime categories",
        options=all_categories,
        default=list(all_categories),
    )

    st.divider()

    # Rate vs count toggle
    st.subheader("Map metric")
    show_rate = st.toggle(
        "Per-capita rate (per 1,000 residents)",
        value=True,
        help=(
            "On: color by incidents per 1,000 residents (quantile bins). "
            "Off: raw incident count. Tracts with fewer than "
            f"{city.min_population_for_rate} residents are suppressed on the rate map."
        ),
    )
    st.caption(
        "Choropleth uses **quantile binning** (each color represents equal numbers of "
        "tracts). Rates use residential population — daytime vs. residential denominator "
        "mismatch is a known caveat; see README."
    )

    st.divider()

    # Snapshot pin
    st.subheader("Snapshot / reproducibility")
    snaps = cache.list_snapshots(city_slug, city.incident_source_slug)
    snap_options = ["Latest (auto)"] + [s["retrieved_at"][:19] for s in snaps]
    pinned = st.selectbox("Pin to pull timestamp", options=snap_options)
    pinned_snapshot = None if pinned == "Latest (auto)" else pinned

    st.divider()

    # KDE controls (shown on projection tab)
    st.subheader("Projection settings")
    bandwidth_m = st.slider(
        "KDE bandwidth (meters)",
        min_value=200, max_value=3000, value=800, step=100,
        help="Spatial smoothing radius for the recent-intensity projection.",
    )
    halflife_days = st.slider(
        "Recency half-life (days)",
        min_value=7, max_value=365, value=90, step=7,
        help="Incidents older than this contribute half as much weight.",
    )

# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load(
    city_slug: str,
    start_iso: str,
    end_iso: str,
    categories: tuple,
    pinned_snapshot,
    prefer_cache: bool = False,
):
    city = CITIES[city_slug]
    return load(
        city,
        pd.Timestamp(start_iso),
        pd.Timestamp(end_iso),
        categories=categories,
        pinned_snapshot=pinned_snapshot,
        prefer_cache=prefer_cache,
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_load_cfs(city_slug: str, start_iso: str, end_iso: str):
    city = CITIES[city_slug]
    return load_cfs(city, pd.Timestamp(start_iso), pd.Timestamp(end_iso))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_boundaries(city_slug: str):
    return load_boundaries(CITIES[city_slug])


# A little personality while data loads (a multi-year pull can take a beat).
_LOADING_QUIPS = (
    "Dusting for prints…",
    "Canvassing the neighborhood…",
    "Pulling the case files…",
    "Connecting the red string…",
    "Following the data trail…",
    "Triangulating hotspots…",
    "Checking the precinct logs…",
)

with st.spinner(random.choice(_LOADING_QUIPS)):
    df, info = cached_load(
        city_slug,
        start.isoformat(),
        end.isoformat(),
        tuple(selected_categories),
        pinned_snapshot,
        prefer_cache,
    )

try:
    boundaries = cached_boundaries(city_slug)
except Exception as e:
    boundaries = None
    st.warning(f"Could not load tract boundaries: {e}")

# ---------------------------------------------------------------------------
# Data quality banner
# ---------------------------------------------------------------------------

source_labels = {"live": "🟢 Live CMPD API", "snapshot": "🟡 Cached snapshot", "demo": "🔴 Synthetic demo"}
source_label = source_labels.get(info.tier, info.tier)
snap_note = f" (pinned: {info.retrieved_at[:19]})" if info.retrieved_at else ""

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Data source", source_labels.get(info.tier, info.tier))
col_b.metric("Incidents loaded", f"{info.row_count:,}")
col_c.metric(
    "Unassigned (⚠️ excluded from rate map)",
    f"{info.unassigned_count:,}",
    delta=f"{info.unassigned_pct:.1f}%",
    delta_color="inverse",
)
col_d.metric("Date range", f"{start.date()} → {end.date()}")

if info.tier == "demo":
    st.warning(
        "⚠️ **Synthetic data only.** The live incidents API is unreachable and no cached "
        "snapshot was found. All counts and rates are fabricated for UI testing."
    )
    if info.live_error:
        with st.expander("Why did the live fetch fail?"):
            st.code(info.live_error, language=None)
elif info.tier == "snapshot" and info.live_error:
    st.info(
        "ℹ️ Serving cached snapshot — the live API was unreachable on this request."
    )
    with st.expander("Live fetch error"):
        st.code(info.live_error, language=None)

# ---------------------------------------------------------------------------
# Compute tract aggregates
# ---------------------------------------------------------------------------

with st.spinner("Aggregating to census tracts…"):
    tract_data = aggregate.tract_counts(df, city, categories=tuple(selected_categories))
    temporal = aggregate.temporal_profile(df)
    trend = aggregate.trend_by_tract(df, city)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab_street, tab2, tab3, tab_live = st.tabs([
    "📊 Per-Capita Hotspots",
    "🔥 Street-Level Heat Map",
    "📈 Trends",
    "🌡️ Recent-Intensity Projection",
    "📡 Live Activity",
])

# ------ Tab 1: Choropleth ------
with tab1:
    st.subheader("Per-Capita Incident Concentration by Census Tract")
    st.caption(
        f"ACS {city.acs_release} population · Census {city.tract_vintage} tract boundaries · "
        f"Mecklenburg County · Quantile bins · "
        f"Tracts < {city.min_population_for_rate} residents suppressed (grey)"
    )

    if tract_data.empty or boundaries is None:
        st.info("No data to display. Try adjusting the date range or category filters.")
    else:
        fmap = hotspots.base_map(city.map_center, city.map_zoom)
        fmap = hotspots.choropleth_layer(fmap, tract_data, boundaries, show_rate=show_rate)
        st_folium(fmap, width=None, height=560, returned_objects=[])

    if not tract_data.empty:
        n_suppressed = tract_data["rate_suppressed"].sum()
        n_total_tracts = len(tract_data)
        n_geo_only = n_total_tracts - len(df[df["geography_id"] != "UNASSIGNED"]["geography_id"].unique())
        st.caption(
            f"{n_suppressed} of {n_total_tracts} tracts suppressed (low pop.) · "
            f"{info.unassigned_count:,} incidents unassigned (outside tract boundaries or null coords)"
        )

# ------ Street-Level Heat Map ------
with tab_street:
    st.subheader("Street-Level Heat Map")
    st.caption(
        "Raw incident points — **not population-adjusted**: busy areas glow partly "
        "because more people are there. Agencies typically geocode to the block "
        "midpoint, so street-level precision is approximate. Use the time slider "
        "to scrub through the selected period."
    )

    if df.empty:
        st.info("No data to display. Try adjusting the date range or category filters.")
    else:
        range_days = (end - start).days
        mode = st.radio(
            "View",
            options=["Static heat map", "Animated over time", "Individual incidents"],
            horizontal=True,
            key="street_mode",
        )

        smap = hotspots.base_map(city.map_center, city.map_zoom)

        if mode == "Static heat map":
            layer, note = streetmap.heatmap_layer(df)
            if layer is None:
                st.info(note)
            else:
                layer.add_to(smap)
                st_folium(smap, width=None, height=560, returned_objects=[])
                if note:
                    st.caption(note)

        elif mode == "Animated over time":
            # Weekly frames stay legible up to ~6 months; monthly beyond that
            freq = "W" if range_days <= 190 else "MS"
            freq_label = "weekly" if freq == "W" else "monthly"
            smap, err = streetmap.animated_heatmap(smap, df, freq=freq)
            if err:
                st.info(err)
            else:
                st_folium(smap, width=None, height=560, returned_objects=[])
                st.caption(
                    f"{freq_label.capitalize()} frames · use ▶ or drag the slider "
                    "to move through time."
                )

        else:
            layer, note = streetmap.marker_cluster_layer(df)
            if layer is None:
                st.info(note)
            else:
                layer.add_to(smap)
                st_folium(smap, width=None, height=560, returned_objects=[])
                if note:
                    st.caption(note)
                st.caption("Marker colors: 🔴 Violent · 🟠 Property · 🔵 Other · ⚪ Unknown")

# ------ Tab 2: Trends ------
with tab2:
    st.subheader("Incident Patterns Over Time")

    if df.empty:
        st.info("No data available for the selected filters.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            # Monthly trend line
            if not temporal["month"].empty:
                st.markdown("**Monthly incident count**")
                chart = (
                    alt.Chart(temporal["month"])
                    .mark_line(point=True, color="#d62728")
                    .encode(
                        x=alt.X("month:O", title="Month"),
                        y=alt.Y("count:Q", title="Incidents"),
                        tooltip=["month", "count"],
                    )
                    .properties(height=220)
                )
                st.altair_chart(chart, width='stretch')

            # Hour of day
            if not temporal["hour"].empty:
                st.markdown("**By hour of day**")
                chart_h = (
                    alt.Chart(temporal["hour"])
                    .mark_bar(color="#1f77b4")
                    .encode(
                        x=alt.X("hour:O", title="Hour"),
                        y=alt.Y("count:Q", title="Incidents"),
                        tooltip=["hour", "count"],
                    )
                    .properties(height=180)
                )
                st.altair_chart(chart_h, width='stretch')

        with col_right:
            # Day of week
            if not temporal["dow"].empty:
                st.markdown("**By day of week**")
                chart_d = (
                    alt.Chart(temporal["dow"])
                    .mark_bar(color="#2ca02c")
                    .encode(
                        x=alt.X("dow:O", title="Day", sort=list(temporal["dow"]["dow"])),
                        y=alt.Y("count:Q", title="Incidents"),
                        tooltip=["dow", "count"],
                    )
                    .properties(height=200)
                )
                st.altair_chart(chart_d, width='stretch')

            # Tract trend table
            if not trend.empty:
                st.markdown("**Top heating/cooling tracts (recent vs. earlier half)**")
                top = (
                    trend.dropna(subset=["pct_change"])
                    .sort_values("pct_change", ascending=False)
                    .head(10)[["geography_id", "count_recent", "count_early", "pct_change"]]
                )
                top["pct_change"] = top["pct_change"].map("{:+.1f}%".format)
                st.dataframe(top, width='stretch', hide_index=True)

# ------ Tab 3: Recent-Intensity Projection ------
with tab3:
    st.subheader("Recent-Intensity Projection")
    st.info(
        "**This is not a forecast.** The projection below uses a recency-weighted "
        "kernel density estimator to visualize *where recent incidents have been "
        "concentrated*. Older incidents receive exponentially lower weight. "
        "This reflects inertia in the data, not a predictive model. "
        "Treat it as a smoothed summary of recent activity, not a prediction of future crime."
    )

    if df.empty:
        st.warning("No data available for projection.")
    else:
        with st.spinner("Computing intensity surface…"):
            fmap2 = hotspots.base_map(city.map_center, city.map_zoom)
            layer, err = intensity.intensity_layer(
                df, city,
                bandwidth_m=bandwidth_m,
                halflife_days=halflife_days,
            )
            if err:
                st.warning(err)
            else:
                layer.add_to(fmap2)
                st_folium(fmap2, width=None, height=520, returned_objects=[])
                st.caption(
                    f"Bandwidth: {bandwidth_m} m · Recency half-life: {halflife_days} days · "
                    f"CRS: {city.planar_crs} (projected) → EPSG:4326 (display)"
                )

# ------ Live Activity (calls for service + scanner links) ------
with tab_live:
    st.subheader("Live Activity — Calls for Service")
    st.caption(
        "Recent 911 / officer-initiated dispatch calls — the structured counterpart "
        "to scanner traffic. **Calls ≠ crimes**: many calls are unfounded, duplicated, "
        "or reclassified after investigation. Use this as a what-is-happening-now view, "
        "not as evidence of crime levels."
    )

    col_lb, col_rf = st.columns([3, 1])
    with col_lb:
        lookback_label = st.select_slider(
            "Lookback window",
            options=["6 hours", "24 hours", "3 days", "7 days"],
            value="24 hours",
        )
    with col_rf:
        st.write("")
        if st.button("🔄 Refresh feed"):
            cached_load_cfs.clear()

    _LOOKBACK_HOURS = {"6 hours": 6, "24 hours": 24, "3 days": 72, "7 days": 168}
    cfs_end = pd.Timestamp.now()
    cfs_start = cfs_end - pd.Timedelta(hours=_LOOKBACK_HOURS[lookback_label])

    with st.spinner("Loading calls for service…"):
        cfs_df, cfs_info = cached_load_cfs(
            city_slug, cfs_start.isoformat(), cfs_end.isoformat()
        )

    cfs_tier_labels = {
        "live": "🟢 Live dispatch feed",
        "snapshot": "🟡 Cached snapshot",
        "demo": "🔴 Synthetic demo feed",
    }
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Feed source", cfs_tier_labels.get(cfs_info.tier, cfs_info.tier))
    mcol2.metric("Calls in window", f"{cfs_info.row_count:,}")
    mcol3.metric("Without coordinates", f"{cfs_info.unassigned_count:,}")

    if cfs_info.tier == "demo":
        st.warning(
            "⚠️ **Synthetic feed.** The live calls-for-service endpoint is unreachable "
            "(or not configured for this city) and no snapshot exists. Calls below are "
            "fabricated for UI testing."
        )

    if cfs_df.empty:
        st.info("No calls in the selected window.")
    else:
        map_col, table_col = st.columns([3, 2])

        with map_col:
            live_map = hotspots.base_map(city.map_center, city.map_zoom)
            layer, note = streetmap.cfs_map_layer(cfs_df)
            if layer is not None:
                layer.add_to(live_map)
                st_folium(live_map, width=None, height=480, returned_objects=[])
            if note:
                st.info(note)

        with table_col:
            st.markdown("**Most recent calls**")
            display = cfs_df.head(200).copy()
            display["time"] = display["datetime"].dt.strftime("%m-%d %H:%M")
            st.dataframe(
                display[["time", "call_type", "address", "division"]],
                width='stretch',
                hide_index=True,
                height=420,
            )

            st.markdown("**Top call types in window**")
            top_types = (
                cfs_df["call_type"].value_counts().head(8)
                .rename_axis("call_type").reset_index(name="count")
            )
            st.dataframe(top_types, width='stretch', hide_index=True)

    if city.scanner_feeds:
        st.divider()
        st.markdown("**🎧 Live scanner audio**")
        st.caption(
            "Scanner audio is a live stream with no structured API, so it isn't "
            "ingested into the maps — listen alongside the dispatch feed instead."
        )
        for feed in city.scanner_feeds:
            st.markdown(f"- [{feed.name}]({feed.url}) — {feed.note}")
