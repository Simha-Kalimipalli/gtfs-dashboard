"""
YRT GTFS Service Allocation Dashboard
Estimates where additional bus service reduces passenger waiting time most.
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import zipfile
import io
import os
import requests
from datetime import datetime, time, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── YRT brand colours ─────────────────────────────────────────────────────────
YRT_DARK  = "#006ec7"
YRT_LIGHT = "#00a3e0"
YRT_WHITE = "#ffffff"
YRT_BG    = "#03101f"
YRT_CARD  = "#071828"
YRT_BORDER= "#0a2540"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YRT Service Allocator",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,600;1,400&display=swap');

  html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {YRT_BG};
  }}
  h1, h2, h3 {{ font-family: 'Space Mono', monospace; color: {YRT_WHITE}; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #041525 0%, #071e35 100%);
    border-right: 1px solid {YRT_BORDER};
  }}

  .metric-card {{
    background: linear-gradient(135deg, {YRT_CARD} 0%, #0a1e33 100%);
    border: 1px solid {YRT_BORDER};
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    color: #e0e0e0;
    margin-bottom: 0.5rem;
  }}
  .metric-card .label {{
    font-size: 0.72rem; color: {YRT_LIGHT};
    letter-spacing: 0.1em; text-transform: uppercase;
  }}
  .metric-card .value {{
    font-size: 2rem; font-family: 'Space Mono', monospace; color: {YRT_WHITE};
  }}
  .metric-card .delta {{ font-size: 0.8rem; color: #54d4a0; }}

  .rank-1 {{ border-left: 4px solid #ffd700; }}
  .rank-2 {{ border-left: 4px solid #c0c0c0; }}
  .rank-3 {{ border-left: 4px solid #cd7f32; }}

  .warning-box {{
    background: #1e1200;
    border: 1px solid #ff8c00;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    color: #ffb347;
    font-size: 0.88rem;
  }}
  .info-box {{
    background: #021528;
    border: 1px solid {YRT_DARK};
    border-radius: 8px;
    padding: 0.85rem 1rem;
    color: {YRT_LIGHT};
    font-size: 0.88rem;
  }}
  .freq-pill {{
    display: inline-block;
    background: {YRT_DARK};
    color: white;
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin: 2px;
  }}
  .section-rule {{
    border: none;
    border-top: 1px solid {YRT_BORDER};
    margin: 1.5rem 0;
  }}

  /* YRT header bar */
  .yrt-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.5rem 0 1.2rem 0;
    border-bottom: 2px solid {YRT_DARK};
    margin-bottom: 1.5rem;
  }}
  .yrt-header .title-block h1 {{
    margin: 0; font-size: 1.6rem; color: {YRT_WHITE};
  }}
  .yrt-header .title-block p {{
    margin: 0; font-size: 0.85rem; color: {YRT_LIGHT}; font-family: 'DM Sans', sans-serif;
  }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — hardcoded YRT GTFS URL
# ═══════════════════════════════════════════════════════════════════════════════
YRT_GTFS_URL = "https://www.yrt.ca/google/google_transit.zip"

# Frequency scenario options: label → target headway in minutes
FREQ_OPTIONS = {
    "Current frequency (as scheduled)":   None,   # None = use actual GTFS headway
    "30-min headway  (2 buses/hr)":        30,
    "15-min headway  (4 buses/hr)":        15,
    "10-min headway  (6 buses/hr)":        10,
    "5-min headway   (12 buses/hr)":        5,
}

# ═══════════════════════════════════════════════════════════════════════════════
# GTFS LOADING & PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def fetch_yrt_gtfs() -> bytes:
    """Download the live YRT GTFS zip and return its bytes."""
    r = requests.get(YRT_GTFS_URL, timeout=60)
    r.raise_for_status()
    return r.content


@st.cache_data(show_spinner=False)
def load_gtfs_bytes(gtfs_bytes: bytes) -> dict:
    """Load all relevant GTFS tables from zip bytes."""
    tables = {}
    required = ["routes.txt", "trips.txt", "stop_times.txt", "calendar.txt", "stops.txt"]
    optional = ["calendar_dates.txt", "shapes.txt"]

    with zipfile.ZipFile(io.BytesIO(gtfs_bytes)) as zf:
        names = zf.namelist()
        prefix = ""
        if "routes.txt" not in names:
            for n in names:
                if n.endswith("routes.txt"):
                    prefix = n.replace("routes.txt", "")
                    break

        for fname in required + optional:
            path = prefix + fname
            if path in names:
                with zf.open(path) as f:
                    tables[fname.replace(".txt", "")] = pd.read_csv(f, dtype=str)
            elif fname in required:
                st.error(f"Missing required GTFS file: {fname}")
                st.stop()
    return tables


def parse_gtfs_time(s: str) -> int:
    """Parse HH:MM:SS GTFS time (allows hours > 23) → total seconds."""
    try:
        parts = str(s).strip().split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        return -1


def get_active_service_ids(tables: dict, analysis_date) -> set:
    """Return service_ids active on the given date."""
    dow = analysis_date.strftime("%A").lower()
    active = set()

    if "calendar" in tables:
        cal = tables["calendar"].copy()
        cal["start_date"] = pd.to_datetime(cal["start_date"], format="%Y%m%d", errors="coerce")
        cal["end_date"]   = pd.to_datetime(cal["end_date"],   format="%Y%m%d", errors="coerce")
        dt = pd.Timestamp(analysis_date)
        mask = (cal["start_date"] <= dt) & (cal["end_date"] >= dt) & (cal[dow] == "1")
        active.update(cal.loc[mask, "service_id"].tolist())

    if "calendar_dates" in tables:
        cd = tables["calendar_dates"].copy()
        date_str = analysis_date.strftime("%Y%m%d")
        added   = cd[(cd["date"] == date_str) & (cd["exception_type"] == "1")]["service_id"].tolist()
        removed = cd[(cd["date"] == date_str) & (cd["exception_type"] == "2")]["service_id"].tolist()
        active.update(added)
        active -= set(removed)

    return active


def compute_route_headways(tables: dict, service_ids: set,
                            window_start_s: int, window_end_s: int) -> pd.DataFrame:
    """For each route/direction, compute headway metrics in the time window."""
    trips  = tables["trips"]
    routes = tables["routes"]
    st_df  = tables["stop_times"]

    active_trips = trips[trips["service_id"].isin(service_ids)].copy()

    st_sorted = st_df.copy()
    st_sorted["stop_sequence"] = pd.to_numeric(st_sorted["stop_sequence"], errors="coerce")
    st_sorted["dep_sec"] = st_sorted["departure_time"].apply(parse_gtfs_time)

    first_stops = (st_sorted.sort_values("stop_sequence")
                             .groupby("trip_id").first().reset_index()
                             [["trip_id", "dep_sec"]])
    last_stops  = (st_sorted.sort_values("stop_sequence")
                             .groupby("trip_id").last().reset_index()
                             [["trip_id", "dep_sec"]].rename(columns={"dep_sec": "arr_sec"}))

    trip_times = active_trips.merge(first_stops, on="trip_id", how="left")
    trip_times = trip_times.merge(last_stops, on="trip_id", how="left")

    in_window = trip_times[
        (trip_times["dep_sec"] >= window_start_s) &
        (trip_times["dep_sec"] <= window_end_s)
    ].copy()

    if in_window.empty:
        return pd.DataFrame()

    in_window["one_way_min"] = (in_window["arr_sec"] - in_window["dep_sec"]) / 60

    results = []
    group_cols = ["route_id"] + (["direction_id"] if "direction_id" in in_window.columns else [])
    window_min = (window_end_s - window_start_s) / 60

    for keys, grp in in_window.groupby(group_cols):
        if isinstance(keys, str):
            keys = (keys,)
        route_id  = keys[0]
        direction = keys[1] if len(keys) > 1 else "0"

        deps    = sorted(grp["dep_sec"].dropna().tolist())
        n_trips = len(deps)
        if n_trips < 1:
            continue

        if n_trips > 1:
            gaps = [deps[i+1] - deps[i] for i in range(len(deps)-1)]
            avg_headway_min = np.mean(gaps) / 60
        else:
            avg_headway_min = window_min

        ow_times   = grp["one_way_min"].dropna()
        avg_ow_min = ow_times.mean() if not ow_times.empty else np.nan
        cycle_min  = 2 * avg_ow_min * 1.15 if not np.isnan(avg_ow_min) else np.nan

        if not np.isnan(cycle_min) and avg_headway_min > 0:
            n_vehicles = math.ceil(cycle_min / avg_headway_min)
        else:
            n_vehicles = np.nan

        results.append({
            "route_id":          route_id,
            "direction_id":      direction,
            "n_trips_in_window": n_trips,
            "avg_headway_min":   round(avg_headway_min, 1),
            "avg_ow_trip_min":   round(avg_ow_min, 1) if not np.isnan(avg_ow_min) else None,
            "cycle_min":         round(cycle_min, 1)  if not np.isnan(cycle_min)  else None,
            "n_vehicles_est":    int(n_vehicles) if not np.isnan(n_vehicles) else None,
            "expected_wait_min": round(avg_headway_min / 2, 1),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    route_names = tables["routes"][["route_id", "route_short_name", "route_long_name"]].drop_duplicates()
    df = df.merge(route_names, on="route_id", how="left")
    df["route_label"] = df["route_short_name"].fillna(df["route_id"])
    return df


def compute_savings_for_scenario(row: pd.Series, target_headway_min: float | None) -> dict:
    """
    Given a route row and a target headway (or None = current), compute savings.
    If target_headway_min is None, returns zero savings (baseline scenario).
    """
    current_h = row["avg_headway_min"]
    current_w = row["expected_wait_min"]

    if target_headway_min is None:
        # "Current frequency" — no change
        return {
            "scenario_headway_min": round(current_h, 1),
            "scenario_wait_min":    round(current_w, 1),
            "wait_reduction_min":   0.0,
            "buses_added":          0,
        }

    # Only improve if target is better (lower) than current
    if target_headway_min >= current_h:
        new_h = current_h
        new_w = current_w
        buses_added = 0
    else:
        new_h = target_headway_min
        new_w = new_h / 2
        # Estimate buses needed
        if row["cycle_min"] and not np.isnan(row["cycle_min"]):
            n_new = math.ceil(row["cycle_min"] / new_h)
            buses_added = max(0, n_new - (row["n_vehicles_est"] or 0))
        else:
            buses_added = None

    return {
        "scenario_headway_min": round(new_h, 1),
        "scenario_wait_min":    round(new_w, 1),
        "wait_reduction_min":   round(max(0, current_w - new_w), 1),
        "buses_added":          buses_added,
    }


def rank_routes(route_df: pd.DataFrame, ridership: dict,
                target_headway_min: float | None) -> pd.DataFrame:
    """Compute passenger-hour savings and rank routes."""
    rows = []
    for _, r in route_df.iterrows():
        pax = ridership.get(r["route_id"], 0)
        sim = compute_savings_for_scenario(r, target_headway_min)
        pws = sim["wait_reduction_min"] * pax   # passenger-minutes
        phs = pws / 60                           # passenger-hours

        rows.append({
            "route_id":           r["route_id"],
            "route_label":        r["route_label"],
            "direction_id":       r["direction_id"],
            "current_headway":    r["avg_headway_min"],
            "current_wait":       r["expected_wait_min"],
            "n_vehicles_est":     r["n_vehicles_est"],
            "scenario_headway":   sim["scenario_headway_min"],
            "scenario_wait":      sim["scenario_wait_min"],
            "wait_reduction_min": sim["wait_reduction_min"],
            "buses_added":        sim["buses_added"],
            "assumed_ridership":  pax,
            "pax_hr_savings":     round(phs, 2),
        })

    out = pd.DataFrame(rows).sort_values("pax_hr_savings", ascending=False).reset_index(drop=True)
    out.index += 1
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def metric_card(label, value, delta=""):
    delta_html = f'<div class="delta">▲ {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


def plotly_yrt(fig):
    """Apply YRT dark theme to a plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#d0d8e4"),
        margin=dict(t=40, b=40, l=20, r=20),
        coloraxis_colorbar=dict(tickfont=dict(color="#d0d8e4")),
    )
    fig.update_xaxes(gridcolor=YRT_BORDER, zerolinecolor=YRT_BORDER)
    fig.update_yaxes(gridcolor=YRT_BORDER, zerolinecolor=YRT_BORDER)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

logo_col, title_col = st.columns([1, 8])
with logo_col:
    # Try to show logo if it exists in the repo; fall back gracefully
    logo_path = "yrt-logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        st.markdown(f'<div style="font-family:Space Mono;font-size:1.4rem;color:{YRT_DARK};font-weight:700;padding-top:8px">YRT</div>', unsafe_allow_html=True)
with title_col:
    st.markdown(f"""
    <div style="padding-top:4px">
      <h1 style="margin:0;font-size:1.7rem;color:{YRT_WHITE}">Service Allocation Tool</h1>
      <p style="margin:0;font-size:0.85rem;color:{YRT_LIGHT};font-family:'DM Sans',sans-serif">
        York Region Transit · GTFS Headway &amp; Passenger-Hour Savings Analysis
      </p>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    if os.path.exists("yrt-logo.png"):
        st.image("yrt-logo.png", width=110)

    st.markdown(f"### Analysis Settings")

    analysis_date = st.date_input(
        "Service date",
        value=datetime.today().date(),
        help="Active service IDs are resolved for this date.",
    )

    col_s, col_e = st.columns(2)
    with col_s:
        window_start = st.time_input("From", value=time(7, 0))
    with col_e:
        window_end   = st.time_input("To",   value=time(9, 0))

    st.markdown("---")
    st.markdown("### Frequency Scenario")
    freq_label = st.radio(
        "Target headway",
        options=list(FREQ_OPTIONS.keys()),
        index=0,
        help=(
            "Select the service frequency you want to model. "
            "Routes already meeting or exceeding the target show zero savings."
        ),
    )
    target_hw = FREQ_OPTIONS[freq_label]

    if target_hw is not None:
        buses_hr = 60 / target_hw
        st.markdown(
            f'<span class="freq-pill">{target_hw} min headway</span>'
            f'<span class="freq-pill">{buses_hr:.0f} buses/hr</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(f"""
    <div class="info-box">
      <b>Data source</b><br>
      Live YRT GTFS feed loaded automatically.<br>
      Ridership values are <em>synthetic</em> — not actual YRT counts.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD GTFS (hardcoded YRT)
# ═══════════════════════════════════════════════════════════════════════════════

with st.spinner("⬇ Fetching YRT GTFS feed…"):
    try:
        gtfs_bytes = fetch_yrt_gtfs()
    except Exception as e:
        st.error(
            f"Could not download the YRT GTFS feed automatically ({e}). "
            "Please upload it manually below."
        )
        manual_upload = st.file_uploader("Upload YRT GTFS .zip", type="zip")
        if manual_upload is None:
            st.stop()
        gtfs_bytes = manual_upload.read()

with st.spinner("Parsing GTFS tables…"):
    tables = load_gtfs_bytes(gtfs_bytes)

routes_df = tables["routes"]
st.success(
    f"✅  Loaded **{routes_df['route_id'].nunique()} routes** · "
    f"**{len(tables['trips'])} trips** · "
    f"**{len(tables['stop_times'])} stop-times**"
)

# ── Route filter ──────────────────────────────────────────────────────────────
st.markdown("### Route Filter *(optional — leave blank for all routes)*")

all_route_opts = (
    routes_df
    .assign(label=lambda d:
        d["route_short_name"].fillna("") + " — " + d["route_long_name"].fillna(""))
    [["route_id", "label"]].drop_duplicates().sort_values("label")
)
label_to_id = dict(zip(all_route_opts["label"], all_route_opts["route_id"]))

selected_labels = st.multiselect(
    "Candidate routes",
    options=all_route_opts["label"].tolist(),
    default=[],
    placeholder="All routes",
)
selected_ids = [label_to_id[l] for l in selected_labels] if selected_labels else None

# ── Compute headways ──────────────────────────────────────────────────────────
ws = window_start.hour * 3600 + window_start.minute * 60
we = window_end.hour   * 3600 + window_end.minute   * 60
if we <= ws:
    st.error("Window end must be after window start.")
    st.stop()

window_min = (we - ws) / 60

with st.spinner("Computing baseline headways…"):
    service_ids = get_active_service_ids(tables, analysis_date)
    if not service_ids:
        st.warning(
            f"No service IDs found for {analysis_date}. "
            "Try a different date or check the GTFS calendar."
        )
        st.stop()

    work_tables = dict(tables)
    if selected_ids:
        ft = tables["trips"][tables["trips"]["route_id"].isin(selected_ids)]
        fs = tables["stop_times"][tables["stop_times"]["trip_id"].isin(ft["trip_id"])]
        work_tables["trips"]      = ft
        work_tables["stop_times"] = fs

    headway_df = compute_route_headways(work_tables, service_ids, ws, we)

if headway_df.empty:
    st.warning("No trips found in the selected window. Try a different time range or date.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Baseline Service Summary")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Routes analysed", str(headway_df["route_id"].nunique()))
with c2:
    metric_card("Active service IDs", str(len(service_ids)))
with c3:
    metric_card("Median headway", f"{headway_df['avg_headway_min'].median():.0f} min")
with c4:
    metric_card("Routes ≥ 30 min headway",
                str(len(headway_df[headway_df["avg_headway_min"] >= 30])))

# Headway histogram
fig_hist = px.histogram(
    headway_df, x="avg_headway_min", nbins=30,
    labels={"avg_headway_min": "Average Headway (min)"},
    color_discrete_sequence=[YRT_DARK],
    template="plotly_dark",
    title="Headway Distribution across Routes",
)
fig_hist.add_vline(x=30, line_dash="dash", line_color="#ff8c00",
                   annotation_text="30-min threshold",
                   annotation_font_color="#ff8c00")
if target_hw:
    fig_hist.add_vline(x=target_hw, line_dash="dot", line_color=YRT_LIGHT,
                       annotation_text=f"Target: {target_hw} min",
                       annotation_font_color=YRT_LIGHT)
st.plotly_chart(plotly_yrt(fig_hist), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RIDERSHIP ASSUMPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Ridership Assumptions")

st.markdown("""
<div class="warning-box">
⚠️  <b>Synthetic data only.</b>  Route-level ridership is not included in the public GTFS feed.
The values below are for demonstration purposes and do <em>not</em> represent actual YRT ridership.
Agencies with APC data can replace these values with observed counts.
</div>
""", unsafe_allow_html=True)

st.markdown("**Quick-fill — set all routes to a uniform ridership**")

qc1, qc2, qc3, qc4 = st.columns(4)
fill_val = None
with qc1:
    if st.button("250 pax / route"):  fill_val = 250
with qc2:
    if st.button("500 pax / route"):  fill_val = 500
with qc3:
    if st.button("1 000 pax / route"): fill_val = 1000
with qc4:
    if st.button("2 000 pax / route"): fill_val = 2000

unique_routes = headway_df[["route_id", "route_label"]].drop_duplicates()

if "ridership_values" not in st.session_state:
    st.session_state.ridership_values = {r: 500 for r in unique_routes["route_id"]}
if fill_val:
    for r in unique_routes["route_id"]:
        st.session_state.ridership_values[r] = fill_val

with st.expander("✏️ Edit per-route ridership", expanded=False):
    st.caption(
        "Enter the assumed number of passengers affected during the selected time window. "
        "These are NOT actual YRT figures."
    )
    ridership_inputs = {}
    cols_per_row = 4
    route_list = unique_routes.to_dict("records")
    for i in range(0, len(route_list), cols_per_row):
        chunk = route_list[i:i+cols_per_row]
        cols  = st.columns(cols_per_row)
        for col, rr in zip(cols, chunk):
            with col:
                v = st.number_input(
                    f"Rte {rr['route_label']}",
                    min_value=0, max_value=100_000,
                    value=st.session_state.ridership_values.get(rr["route_id"], 500),
                    step=50,
                    key=f"rid_{rr['route_id']}",
                )
                ridership_inputs[rr["route_id"]] = v

final_ridership = {**st.session_state.ridership_values, **ridership_inputs}

# ═══════════════════════════════════════════════════════════════════════════════
# RANKING
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Route Ranking · Scenario Results")

scenario_desc = freq_label
st.caption(f"**Scenario:** {scenario_desc}")

ranked = rank_routes(headway_df, final_ridership, target_hw)

if ranked.empty or ranked["pax_hr_savings"].sum() == 0:
    st.info("No routes show positive savings under the current scenario. "
            "All routes may already meet or exceed the target frequency.")
    st.stop()

# Top 3
st.markdown("### 🏆 Top Candidates")
top3   = ranked[ranked["pax_hr_savings"] > 0].head(3)
medals = ["🥇", "🥈", "🥉"]
cols   = st.columns(max(len(top3), 1))
for i, (_, row) in enumerate(top3.iterrows()):
    with cols[i]:
        buses_str = (f"+{int(row['buses_added'])} bus(es)"
                     if row["buses_added"] not in (None, 0, np.nan) else "")
        st.markdown(f"""
        <div class="metric-card rank-{i+1}">
          <div class="label">{medals[i]} Rank {i+1} {buses_str}</div>
          <div class="value">{row['route_label']}</div>
          <div style="color:{YRT_LIGHT};font-size:0.9rem;margin-top:4px">
            {row['pax_hr_savings']:.1f} pax-hrs saved
          </div>
          <div style="color:#a0a0a0;font-size:0.8rem;margin-top:4px">
            {row['current_headway']} min → {row['scenario_headway']} min headway<br>
            Wait: {row['current_wait']} → {row['scenario_wait']} min
          </div>
        </div>""", unsafe_allow_html=True)

# Full table — no background_gradient (no matplotlib dependency)
st.markdown("### Full Route Ranking")
display_cols = {
    "route_label":        "Route",
    "direction_id":       "Dir",
    "current_headway":    "Current Hdwy (min)",
    "scenario_headway":   "Scenario Hdwy (min)",
    "current_wait":       "Current Wait (min)",
    "scenario_wait":      "Scenario Wait (min)",
    "wait_reduction_min": "Wait Saved (min/pax)",
    "buses_added":        "Est. Buses Added",
    "assumed_ridership":  "Ridership (assumed)",
    "pax_hr_savings":     "Pax-Hr Savings",
}
display_df = ranked[list(display_cols.keys())].rename(columns=display_cols)
st.dataframe(
    display_df.style.format({
        "Pax-Hr Savings":       "{:.2f}",
        "Wait Saved (min/pax)": "{:.1f}",
        "Current Hdwy (min)":   "{:.1f}",
        "Scenario Hdwy (min)":  "{:.1f}",
    }),
    use_container_width=True,
    height=420,
)

# Bar chart
st.markdown("### Passenger-Hour Savings by Route (Top 20)")
fig_bar = px.bar(
    ranked[ranked["pax_hr_savings"] > 0].head(20),
    x="route_label", y="pax_hr_savings",
    color="pax_hr_savings",
    color_continuous_scale=[[0, YRT_DARK], [1, YRT_LIGHT]],
    labels={"route_label": "Route", "pax_hr_savings": "Pax-Hr Savings"},
    template="plotly_dark",
    text="pax_hr_savings",
    title="Top Routes by Estimated Passenger-Hour Savings",
)
fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig_bar.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_tickangle=-35)
st.plotly_chart(plotly_yrt(fig_bar), use_container_width=True)

# Bubble chart
st.markdown("### Headway vs. Ridership")
st.caption("Bubble size = passenger-hour savings. Top-right = high headway + high ridership = greatest benefit.")
fig_sc = px.scatter(
    ranked[ranked["pax_hr_savings"] > 0],
    x="current_headway", y="assumed_ridership",
    size="pax_hr_savings", color="pax_hr_savings",
    hover_name="route_label",
    hover_data={"pax_hr_savings": ":.2f"},
    color_continuous_scale=[[0, YRT_DARK], [1, YRT_LIGHT]],
    labels={"current_headway": "Current Headway (min)", "assumed_ridership": "Assumed Ridership"},
    template="plotly_dark", size_max=50,
    title="Route Priority Map",
)
st.plotly_chart(plotly_yrt(fig_sc), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY — compare all 5 scenarios for top route
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
st.markdown("## Sensitivity: All Frequency Scenarios")

top_route  = ranked.iloc[0]
sens_rows  = []
for lbl, hw in FREQ_OPTIONS.items():
    sim = compute_savings_for_scenario(top_route, hw)
    phs = sim["wait_reduction_min"] * final_ridership.get(top_route["route_id"], 0) / 60
    sens_rows.append({
        "Scenario":            lbl,
        "Target Hdwy (min)":   hw if hw else top_route["current_headway"],
        "New Wait (min)":      sim["scenario_wait_min"],
        "Pax-Hr Savings":      round(phs, 2),
    })

sens_df = pd.DataFrame(sens_rows)

fig_sens = make_subplots(specs=[[{"secondary_y": True}]])
fig_sens.add_trace(go.Bar(
    x=sens_df["Scenario"], y=sens_df["Pax-Hr Savings"],
    name="Pax-Hr Savings", marker_color=YRT_DARK,
), secondary_y=False)
fig_sens.add_trace(go.Scatter(
    x=sens_df["Scenario"], y=sens_df["New Wait (min)"],
    name="Wait Time (min)", mode="lines+markers",
    line=dict(color=YRT_LIGHT, width=2),
    marker=dict(size=8, color=YRT_LIGHT),
), secondary_y=True)
fig_sens.update_layout(
    title=f"Scenario Comparison — Route {top_route['route_label']}",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#d0d8e4"),
    legend=dict(orientation="h", y=1.12),
    xaxis_tickangle=-20,
    margin=dict(t=60, b=60),
)
fig_sens.update_yaxes(title_text="Pax-Hr Savings",  secondary_y=False,
                       gridcolor=YRT_BORDER, zerolinecolor=YRT_BORDER)
fig_sens.update_yaxes(title_text="Wait Time (min)", secondary_y=True,
                       gridcolor=YRT_BORDER)
st.plotly_chart(fig_sens, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT & EXTRAS
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("🗂 Raw Baseline Headway Data"):
    st.dataframe(headway_df, use_container_width=True)

st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
col_dl, col_meth = st.columns([1, 2])
with col_dl:
    csv_bytes = ranked.to_csv(index=True).encode()
    st.download_button(
        "⬇ Download ranking as CSV",
        data=csv_bytes, file_name="yrt_route_ranking.csv", mime="text/csv",
    )

with col_meth:
    with st.expander("📐 Methodology"):
        st.markdown(r"""
**Baseline headway** `H_r` — average gap between consecutive departures.

**Expected wait** (half-headway): `W_r = H_r / 2`

**Cycle time**: `C_r = 2 × one-way trip time × 1.15`

**Scenario headway**: set directly by the chosen frequency tier (e.g. 15, 10, 5 min).
Routes already meeting the target show zero savings.

**Passenger-hour savings**: `PHS_r = (W_r − W_new,r) × R_r / 60`

Ridership `R_r` is **synthetic / user-defined** — not observed YRT data.
        """)

st.caption("YRT Service Allocation Tool · Proof-of-concept screening aid · Not a replacement for operational service planning.")
