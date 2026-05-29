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
from datetime import datetime, time, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GTFS Service Allocator",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Space Mono', monospace; }

  .metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    color: #e0e0e0;
    margin-bottom: 0.5rem;
  }
  .metric-card .label { font-size: 0.75rem; color: #7ecfff; letter-spacing: 0.1em; text-transform: uppercase; }
  .metric-card .value { font-size: 2rem; font-family: 'Space Mono', monospace; color: #ffffff; }
  .metric-card .delta { font-size: 0.8rem; color: #54d4a0; }

  .route-badge {
    display: inline-block;
    background: #0f3460;
    color: #7ecfff;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    margin-right: 4px;
  }
  .rank-1 { border-left: 4px solid #ffd700; }
  .rank-2 { border-left: 4px solid #c0c0c0; }
  .rank-3 { border-left: 4px solid #cd7f32; }
  .warning-box {
    background: #2d1b00;
    border: 1px solid #ff8c00;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    color: #ffb347;
    font-size: 0.88rem;
  }
  .info-box {
    background: #001a2d;
    border: 1px solid #0077b6;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    color: #90e0ef;
    font-size: 0.88rem;
  }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# GTFS LOADING & PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_gtfs(uploaded_file_bytes: bytes) -> dict[str, pd.DataFrame]:
    """Load all relevant GTFS tables from a zip file."""
    tables = {}
    required = ["routes.txt", "trips.txt", "stop_times.txt",
                 "calendar.txt", "stops.txt"]
    optional = ["calendar_dates.txt", "shapes.txt"]

    with zipfile.ZipFile(io.BytesIO(uploaded_file_bytes)) as zf:
        names = zf.namelist()
        # Handle nested folder inside zip
        prefix = ""
        if not "routes.txt" in names:
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


def get_active_service_ids(tables: dict, analysis_date: datetime.date) -> set:
    """Return service_ids active on the given date."""
    dow = analysis_date.strftime("%A").lower()  # e.g. "monday"
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
    """
    For each route, compute headway metrics during the analysis window.
    Returns a DataFrame with one row per route direction.
    """
    trips  = tables["trips"]
    routes = tables["routes"]
    st_df  = tables["stop_times"]

    # Filter to active trips
    active_trips = trips[trips["service_id"].isin(service_ids)].copy()

    # Get first stop departure for each trip (proxy for departure time)
    st_sorted = st_df.copy()
    st_sorted["stop_sequence"] = pd.to_numeric(st_sorted["stop_sequence"], errors="coerce")
    st_sorted["dep_sec"] = st_sorted["departure_time"].apply(parse_gtfs_time)

    # First stop of each trip
    first_stops = (st_sorted.sort_values("stop_sequence")
                             .groupby("trip_id")
                             .first()
                             .reset_index()[["trip_id", "dep_sec", "departure_time"]])

    # Last stop of each trip (for one-way trip duration)
    last_stops = (st_sorted.sort_values("stop_sequence")
                            .groupby("trip_id")
                            .last()
                            .reset_index()[["trip_id", "dep_sec"]]
                            .rename(columns={"dep_sec": "arr_sec"}))

    # Merge
    trip_times = active_trips.merge(first_stops, on="trip_id", how="left")
    trip_times = trip_times.merge(last_stops, on="trip_id", how="left")

    # Filter to window
    in_window = trip_times[
        (trip_times["dep_sec"] >= window_start_s) &
        (trip_times["dep_sec"] <= window_end_s)
    ].copy()

    if in_window.empty:
        return pd.DataFrame()

    in_window["one_way_min"] = (in_window["arr_sec"] - in_window["dep_sec"]) / 60

    # Group by route + direction
    results = []
    group_cols = ["route_id"]
    if "direction_id" in in_window.columns:
        group_cols.append("direction_id")

    for keys, grp in in_window.groupby(group_cols):
        if isinstance(keys, str):
            keys = (keys,)
        route_id = keys[0]
        direction = keys[1] if len(keys) > 1 else "0"

        deps = sorted(grp["dep_sec"].dropna().tolist())
        n_trips = len(deps)
        if n_trips < 1:
            continue

        window_min = (window_end_s - window_start_s) / 60

        # Average headway
        if n_trips > 1:
            gaps = [deps[i+1] - deps[i] for i in range(len(deps)-1)]
            avg_headway_min = np.mean(gaps) / 60
        else:
            avg_headway_min = window_min  # only 1 trip → headway = whole window

        # One-way trip time
        ow_times = grp["one_way_min"].dropna()
        avg_ow_min = ow_times.mean() if not ow_times.empty else np.nan

        # Estimated cycle time (2 × one-way + 15% recovery)
        if not np.isnan(avg_ow_min):
            cycle_min = 2 * avg_ow_min * 1.15
        else:
            cycle_min = np.nan

        # Current vehicle count estimate
        if not np.isnan(cycle_min) and avg_headway_min > 0:
            n_vehicles = math.ceil(cycle_min / avg_headway_min)
        else:
            n_vehicles = np.nan

        # Expected wait (half-headway)
        expected_wait_min = avg_headway_min / 2

        results.append({
            "route_id": route_id,
            "direction_id": direction,
            "n_trips_in_window": n_trips,
            "avg_headway_min": round(avg_headway_min, 1),
            "avg_ow_trip_min": round(avg_ow_min, 1) if not np.isnan(avg_ow_min) else None,
            "cycle_min": round(cycle_min, 1) if not np.isnan(cycle_min) else None,
            "n_vehicles_est": int(n_vehicles) if not np.isnan(n_vehicles) else None,
            "expected_wait_min": round(expected_wait_min, 1),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Join route short name
    route_names = routes[["route_id", "route_short_name", "route_long_name"]].drop_duplicates()
    df = df.merge(route_names, on="route_id", how="left")
    df["route_label"] = df["route_short_name"].fillna(df["route_id"])
    return df


def simulate_added_service(row: pd.Series, added_buses: int) -> dict:
    """Given a route row, simulate adding `added_buses` and return new metrics."""
    if pd.isna(row["cycle_min"]) or pd.isna(row["n_vehicles_est"]):
        return {}
    n_new    = row["n_vehicles_est"] + added_buses
    h_new    = row["cycle_min"] / n_new
    wait_new = h_new / 2
    delta_w  = row["expected_wait_min"] - wait_new
    return {
        "new_headway_min": round(h_new, 1),
        "new_wait_min":    round(wait_new, 1),
        "wait_reduction_min": round(delta_w, 1),
    }


def rank_routes(route_df: pd.DataFrame, ridership: dict[str, float],
                added_buses: int) -> pd.DataFrame:
    """Compute passenger-hour savings and rank routes."""
    rows = []
    for _, r in route_df.iterrows():
        key = r["route_id"]
        pax = ridership.get(key, 0)
        sim = simulate_added_service(r, added_buses)
        if not sim or sim["wait_reduction_min"] <= 0:
            continue
        pws = sim["wait_reduction_min"] * pax          # passenger-minutes saved
        phs = pws / 60                                  # passenger-hours saved
        rows.append({
            "route_id":           r["route_id"],
            "route_label":        r["route_label"],
            "direction_id":       r["direction_id"],
            "current_headway":    r["avg_headway_min"],
            "current_wait":       r["expected_wait_min"],
            "n_vehicles_est":     r["n_vehicles_est"],
            "new_headway":        sim["new_headway_min"],
            "new_wait":           sim["new_wait_min"],
            "wait_reduction_min": sim["wait_reduction_min"],
            "assumed_ridership":  pax,
            "pax_hr_savings":     round(phs, 2),
        })

    out = pd.DataFrame(rows).sort_values("pax_hr_savings", ascending=False).reset_index(drop=True)
    out.index += 1
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, delta: str = ""):
    delta_html = f'<div class="delta">▲ {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🚌 GTFS Service Allocation Tool")
st.caption("Estimate where adding buses saves the most passenger time · Based on YRT methodology")

# ── Sidebar: Data Upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("1 · Load GTFS Feed")
    uploaded = st.file_uploader("Upload GTFS .zip", type="zip")

    st.markdown("---")
    st.header("2 · Analysis Settings")

    analysis_date = st.date_input(
        "Service date",
        value=datetime.today().date(),
        help="The tool will find service_ids active on this date."
    )

    col_s, col_e = st.columns(2)
    with col_s:
        window_start = st.time_input("Window start", value=time(7, 0))
    with col_e:
        window_end   = st.time_input("Window end",   value=time(9, 0))

    st.markdown("---")
    st.header("3 · Scenario")
    added_buses = st.slider("Buses to add per route", 1, 5, 1)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <b>About this tool</b><br>
    Implements the half-headway methodology to rank routes by passenger-hour savings.
    Ridership is synthetic/user-defined — not actual YRT data.
    </div>
    """, unsafe_allow_html=True)

# ── Main panel ────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="info-box" style="margin-top:2rem; font-size:1rem; padding:1.5rem;">
    👈 &nbsp;Upload a <b>GTFS .zip</b> file in the sidebar to begin.<br><br>
    You can download York Region Transit's GTFS feed from
    <a href="https://www.yrt.ca/en/about-us/developer-resources.aspx" target="_blank" style="color:#7ecfff">yrt.ca developer resources</a>
    or use any standard GTFS feed.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Reading GTFS files…"):
    tables = load_gtfs(uploaded.read())

routes_df = tables["routes"]

st.success(f"Loaded {len(routes_df)} routes, {len(tables['trips'])} trips, {len(tables['stop_times'])} stop-times.")

# ── Route filter ──────────────────────────────────────────────────────────────
st.markdown("### Route Filter *(optional)*")
st.caption("Leave blank to analyse all routes. For large feeds, selecting specific routes is faster.")

all_route_options = (
    routes_df
    .assign(label=lambda d: d["route_short_name"].fillna("") + " — " + d["route_long_name"].fillna(""))
    [["route_id", "label"]]
    .drop_duplicates()
    .sort_values("label")
)
label_to_id = dict(zip(all_route_options["label"], all_route_options["route_id"]))

selected_labels = st.multiselect(
    "Select candidate routes",
    options=all_route_options["label"].tolist(),
    default=[]
)
selected_route_ids = [label_to_id[l] for l in selected_labels] if selected_labels else None

# ── Compute headways ──────────────────────────────────────────────────────────
ws = window_start.hour * 3600 + window_start.minute * 60
we = window_end.hour   * 3600 + window_end.minute   * 60
if we <= ws:
    st.error("Window end must be after window start.")
    st.stop()

window_min = (we - ws) / 60

with st.spinner("Computing headways…"):
    service_ids = get_active_service_ids(tables, analysis_date)
    if not service_ids:
        st.warning(
            f"No service IDs found for {analysis_date}. "
            "Try a different date or check the calendar."
        )
        st.stop()

    # Filter tables if user selected specific routes
    if selected_route_ids:
        filtered_trips = tables["trips"][tables["trips"]["route_id"].isin(selected_route_ids)]
        filtered_st    = tables["stop_times"][tables["stop_times"]["trip_id"].isin(filtered_trips["trip_id"])]
        filtered_tables = dict(tables)
        filtered_tables["trips"] = filtered_trips
        filtered_tables["stop_times"] = filtered_st
    else:
        filtered_tables = tables

    headway_df = compute_route_headways(filtered_tables, service_ids, ws, we)

if headway_df.empty:
    st.warning("No trips found in the selected window. Try a different time range or date.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Baseline Service Summary")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Routes analysed", str(headway_df["route_id"].nunique()))
with c2:
    metric_card("Active service IDs", str(len(service_ids)))
with c3:
    metric_card("Median headway", f"{headway_df['avg_headway_min'].median():.0f} min")
with c4:
    long_hw = headway_df[headway_df["avg_headway_min"] >= 30]
    metric_card("Routes ≥ 30 min headway", str(len(long_hw)))

# ── Headway distribution chart ────────────────────────────────────────────────
st.markdown("### Headway Distribution")

fig_hist = px.histogram(
    headway_df,
    x="avg_headway_min",
    nbins=30,
    labels={"avg_headway_min": "Average Headway (min)"},
    color_discrete_sequence=["#0077b6"],
    template="plotly_dark",
)
fig_hist.add_vline(x=30, line_dash="dash", line_color="#ff8c00",
                   annotation_text="30-min threshold", annotation_position="top right")
fig_hist.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_family="DM Sans",
    margin=dict(t=30, b=30),
)
st.plotly_chart(fig_hist, use_container_width=True)

# ── Ridership input ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Ridership Assumptions")
st.markdown("""
<div class="warning-box">
⚠️ Detailed route-level ridership is not in the public GTFS feed.
Enter <b>synthetic or assumed</b> daily passenger counts below.
These are for demonstration only and do not represent actual ridership.
</div>
""", unsafe_allow_html=True)

st.markdown("**Quick-fill options**")
fill_col1, fill_col2, fill_col3 = st.columns(3)
ridership_mode = None
with fill_col1:
    if st.button("Fill all with 500 pax"):
        ridership_mode = 500
with fill_col2:
    if st.button("Fill all with 1 000 pax"):
        ridership_mode = 1000
with fill_col3:
    if st.button("Fill all with 2 000 pax"):
        ridership_mode = 2000

# Show a table with editable ridership
unique_routes = headway_df[["route_id", "route_label"]].drop_duplicates()

if "ridership_values" not in st.session_state:
    st.session_state.ridership_values = {r: 500 for r in unique_routes["route_id"]}

if ridership_mode:
    for r in unique_routes["route_id"]:
        st.session_state.ridership_values[r] = ridership_mode

with st.expander("Edit per-route ridership", expanded=False):
    ridership_inputs = {}
    cols_per_row = 3
    route_list = unique_routes.to_dict("records")
    for i in range(0, len(route_list), cols_per_row):
        row_routes = route_list[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for col, rr in zip(cols, row_routes):
            with col:
                val = st.number_input(
                    f"Rte {rr['route_label']}",
                    min_value=0,
                    max_value=100000,
                    value=st.session_state.ridership_values.get(rr["route_id"], 500),
                    step=100,
                    key=f"rid_{rr['route_id']}"
                )
                ridership_inputs[rr["route_id"]] = val

# Merge session state with inputs
final_ridership = {**st.session_state.ridership_values, **ridership_inputs}

# ── Run ranking ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Route Ranking · Added-Service Scenario")
st.caption(f"Simulating adding **{added_buses} bus(es)** to each candidate route")

ranked = rank_routes(headway_df, final_ridership, added_buses)

if ranked.empty:
    st.warning("No routes produced positive savings. All routes may have missing cycle-time data.")
    st.stop()

# ── Top 3 highlight ───────────────────────────────────────────────────────────
st.markdown("### 🏆 Top Candidates")
top3 = ranked.head(3)
medals = ["🥇", "🥈", "🥉"]
cols = st.columns(len(top3))
for i, (_, row) in enumerate(top3.iterrows()):
    with cols[i]:
        st.markdown(f"""
        <div class="metric-card rank-{i+1}">
          <div class="label">{medals[i]} Rank {i+1}</div>
          <div class="value">{row['route_label']}</div>
          <div style="color:#7ecfff; font-size:0.9rem; margin-top:4px">
            {row['pax_hr_savings']:.1f} pax-hrs saved
          </div>
          <div style="color:#a0a0a0; font-size:0.8rem; margin-top:4px">
            {row['current_headway']} min → {row['new_headway']} min headway<br>
            Wait: {row['current_wait']} → {row['new_wait']} min
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Full ranking table ────────────────────────────────────────────────────────
st.markdown("### Full Route Ranking")
display_cols = {
    "route_label":        "Route",
    "direction_id":       "Dir",
    "current_headway":    "Current Hdwy (min)",
    "new_headway":        "New Hdwy (min)",
    "current_wait":       "Current Wait (min)",
    "new_wait":           "New Wait (min)",
    "wait_reduction_min": "Wait Saved (min/pax)",
    "assumed_ridership":  "Ridership (assumed)",
    "pax_hr_savings":     "Pax-Hr Savings",
}
display_df = ranked[list(display_cols.keys())].rename(columns=display_cols)

st.dataframe(
    display_df.style
    .background_gradient(subset=["Pax-Hr Savings"], cmap="YlOrRd")
    .format({"Pax-Hr Savings": "{:.2f}", "Wait Saved (min/pax)": "{:.1f}"}),
    use_container_width=True,
    height=400,
)

# ── Bar chart: passenger-hour savings ────────────────────────────────────────
st.markdown("### Passenger-Hour Savings by Route")

fig_bar = px.bar(
    ranked.head(20),
    x="route_label",
    y="pax_hr_savings",
    color="pax_hr_savings",
    color_continuous_scale="Teal",
    labels={"route_label": "Route", "pax_hr_savings": "Pax-Hr Savings"},
    template="plotly_dark",
    text="pax_hr_savings",
)
fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig_bar.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_family="DM Sans",
    showlegend=False,
    coloraxis_showscale=False,
    margin=dict(t=30, b=30),
    xaxis_tickangle=-35,
)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Scatter: headway vs ridership (bubble = savings) ─────────────────────────
st.markdown("### Headway vs. Ridership")
st.caption("Bubble size = passenger-hour savings. Routes in the top-right (high headway, high ridership) benefit most.")

fig_scatter = px.scatter(
    ranked,
    x="current_headway",
    y="assumed_ridership",
    size="pax_hr_savings",
    color="pax_hr_savings",
    hover_name="route_label",
    hover_data={"pax_hr_savings": ":.2f", "current_headway": True, "assumed_ridership": True},
    labels={
        "current_headway": "Current Headway (min)",
        "assumed_ridership": "Assumed Ridership",
        "pax_hr_savings": "Pax-Hr Savings"
    },
    color_continuous_scale="Teal",
    template="plotly_dark",
    size_max=50,
)
fig_scatter.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_family="DM Sans",
    margin=dict(t=30, b=30),
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Multi-bus sensitivity ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## Sensitivity: Incremental Bus Addition")
st.caption("How do savings change as more buses are added to the top route?")

top_route = ranked.iloc[0]
sensitivity_rows = []
for b in range(1, 6):
    sim = simulate_added_service(top_route, b)
    if sim:
        phs = sim["wait_reduction_min"] * final_ridership.get(top_route["route_id"], 0) / 60
        sensitivity_rows.append({"Buses Added": b, "New Headway (min)": sim["new_headway_min"],
                                  "Pax-Hr Savings": round(phs, 2)})

if sensitivity_rows:
    sens_df = pd.DataFrame(sensitivity_rows)
    fig_sens = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sens.add_trace(go.Bar(x=sens_df["Buses Added"], y=sens_df["Pax-Hr Savings"],
                               name="Pax-Hr Savings", marker_color="#0077b6"), secondary_y=False)
    fig_sens.add_trace(go.Scatter(x=sens_df["Buses Added"], y=sens_df["New Headway (min)"],
                                   name="New Headway (min)", mode="lines+markers",
                                   marker_color="#ff8c00"), secondary_y=True)
    fig_sens.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="DM Sans",
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", y=1.1),
        xaxis_title="Buses Added",
    )
    fig_sens.update_yaxes(title_text="Pax-Hr Savings", secondary_y=False)
    fig_sens.update_yaxes(title_text="New Headway (min)", secondary_y=True)
    st.plotly_chart(fig_sens, use_container_width=True)

# ── Raw baseline table ────────────────────────────────────────────────────────
with st.expander("Raw Baseline Headway Data"):
    st.dataframe(headway_df, use_container_width=True)

# ── CSV download ──────────────────────────────────────────────────────────────
st.markdown("---")
csv = ranked.to_csv(index=True).encode()
st.download_button(
    label="⬇ Download ranking as CSV",
    data=csv,
    file_name="route_ranking.csv",
    mime="text/csv",
)

# ── Methodology note ──────────────────────────────────────────────────────────
with st.expander("📐 Methodology"):
    st.markdown(r"""
**Baseline headway** (`H_r`) — average gap between consecutive departures within the window.

**Expected wait** (half-headway assumption):
$$W_r = \frac{H_r}{2}$$

**Cycle time** (estimated):
$$C_r = 2 \times \text{one-way trip time} \times 1.15$$

**Vehicles required**:
$$N_r = \left\lceil \frac{C_r}{H_r} \right\rceil$$

**New headway after adding buses**:
$$H_{new,r} = \frac{C_r}{N_r + \Delta}$$

**Passenger-hour savings**:
$$PHS_r = \frac{(W_r - W_{new,r}) \times R_r}{60}$$

Routes are ranked by $PHS_r$ descending.  
Ridership $R_r$ is **synthetic / user-defined** — not observed YRT data.
    """)

st.caption("Tool developed as a proof-of-concept screening aid. Not a replacement for operational service planning.")
