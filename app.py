"""
YRT GTFS Service Allocation Dashboard
======================================
Gravity-model ridership (pure Python, no geopandas) +
GTFS headway analysis + r5py accessibility stub.
"""

import streamlit as st
import pandas as pd
import numpy as np
import math, zipfile, io, os, json
import requests
from datetime import datetime, time as dtime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ridership import (
    compute_gravity_ridership,
    ridership_for_all_routes,
    hourly_profile_chart_data,
    HOURLY_NORM,
)

# ── Brand ────────────────────────────────────────────────────────────────────
YRT_DARK   = "#006ec7"
YRT_LIGHT  = "#00a3e0"
YRT_WHITE  = "#ffffff"
YRT_BG     = "#03101f"
YRT_CARD   = "#071828"
YRT_BORDER = "#0a2540"
YRT_GTFS_URL = "https://www.yrt.ca/google/google_transit.zip"

FREQ_OPTIONS = {
    "Current frequency (as scheduled)": None,
    "30-min headway  (2 buses/hr)":      30,
    "15-min headway  (4 buses/hr)":      15,
    "10-min headway  (6 buses/hr)":      10,
    "5-min headway   (12 buses/hr)":      5,
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="YRT Service Allocator", page_icon="🚌",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,600;1,400&display=swap');
html,body,[class*="css"]{{font-family:'DM Sans',sans-serif;background-color:{YRT_BG};}}
h1,h2,h3{{font-family:'Space Mono',monospace;color:{YRT_WHITE};}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,#041525 0%,#071e35 100%);border-right:1px solid {YRT_BORDER};}}
.metric-card{{background:linear-gradient(135deg,{YRT_CARD} 0%,#0a1e33 100%);border:1px solid {YRT_BORDER};border-radius:10px;padding:1.2rem 1.5rem;color:#e0e0e0;margin-bottom:0.5rem;}}
.metric-card .label{{font-size:.72rem;color:{YRT_LIGHT};letter-spacing:.1em;text-transform:uppercase;}}
.metric-card .value{{font-size:2rem;font-family:'Space Mono',monospace;color:{YRT_WHITE};}}
.metric-card .delta{{font-size:.8rem;color:#54d4a0;}}
.rank-1{{border-left:4px solid #ffd700;}}.rank-2{{border-left:4px solid #c0c0c0;}}.rank-3{{border-left:4px solid #cd7f32;}}
.warning-box{{background:#1e1200;border:1px solid #ff8c00;border-radius:8px;padding:.85rem 1rem;color:#ffb347;font-size:.88rem;}}
.info-box{{background:#021528;border:1px solid {YRT_DARK};border-radius:8px;padding:.85rem 1rem;color:{YRT_LIGHT};font-size:.88rem;}}
.success-box{{background:#001a10;border:1px solid #00c97a;border-radius:8px;padding:.85rem 1rem;color:#54d4a0;font-size:.88rem;}}
.freq-pill{{display:inline-block;background:{YRT_DARK};color:white;border-radius:20px;padding:3px 12px;font-family:'Space Mono',monospace;font-size:.8rem;margin:2px;}}
.section-rule{{border:none;border-top:1px solid {YRT_BORDER};margin:1.5rem 0;}}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# GTFS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def fetch_yrt_gtfs() -> bytes:
    r = requests.get(YRT_GTFS_URL, timeout=60)
    r.raise_for_status()
    return r.content

@st.cache_data(show_spinner=False)
def load_gtfs_bytes(gtfs_bytes: bytes) -> dict:
    tables = {}
    required = ["routes.txt","trips.txt","stop_times.txt","calendar.txt","stops.txt"]
    optional = ["calendar_dates.txt","shapes.txt"]
    with zipfile.ZipFile(io.BytesIO(gtfs_bytes)) as zf:
        names = zf.namelist()
        prefix = ""
        if "routes.txt" not in names:
            for n in names:
                if n.endswith("routes.txt"):
                    prefix = n.replace("routes.txt",""); break
        for fname in required + optional:
            path = prefix + fname
            if path in names:
                with zf.open(path) as f:
                    tables[fname.replace(".txt","")] = pd.read_csv(f, dtype=str)
            elif fname in required:
                st.error(f"Missing required GTFS file: {fname}"); st.stop()
    return tables

def parse_gtfs_time(s):
    try:
        p = str(s).strip().split(":")
        return int(p[0])*3600 + int(p[1])*60 + int(p[2])
    except: return -1

def get_active_service_ids(tables, analysis_date):
    dow = analysis_date.strftime("%A").lower()
    active = set()
    if "calendar" in tables:
        cal = tables["calendar"].copy()
        cal["start_date"] = pd.to_datetime(cal["start_date"], format="%Y%m%d", errors="coerce")
        cal["end_date"]   = pd.to_datetime(cal["end_date"],   format="%Y%m%d", errors="coerce")
        dt = pd.Timestamp(analysis_date)
        mask = (cal["start_date"]<=dt)&(cal["end_date"]>=dt)&(cal[dow]=="1")
        active.update(cal.loc[mask,"service_id"].tolist())
    if "calendar_dates" in tables:
        cd = tables["calendar_dates"].copy()
        ds = analysis_date.strftime("%Y%m%d")
        active.update(cd[(cd["date"]==ds)&(cd["exception_type"]=="1")]["service_id"].tolist())
        active -= set(cd[(cd["date"]==ds)&(cd["exception_type"]=="2")]["service_id"].tolist())
    return active

def compute_route_headways(tables, service_ids, ws, we):
    trips  = tables["trips"]
    st_df  = tables["stop_times"]
    active = trips[trips["service_id"].isin(service_ids)].copy()
    ss = st_df.copy()
    ss["stop_sequence"] = pd.to_numeric(ss["stop_sequence"], errors="coerce")
    ss["dep_sec"] = ss["departure_time"].apply(parse_gtfs_time)
    first = ss.sort_values("stop_sequence").groupby("trip_id").first().reset_index()[["trip_id","dep_sec"]]
    last  = ss.sort_values("stop_sequence").groupby("trip_id").last().reset_index()[["trip_id","dep_sec"]].rename(columns={"dep_sec":"arr_sec"})
    tt = active.merge(first, on="trip_id", how="left").merge(last, on="trip_id", how="left")
    inw = tt[(tt["dep_sec"]>=ws)&(tt["dep_sec"]<=we)].copy()
    if inw.empty: return pd.DataFrame()
    inw["one_way_min"] = (inw["arr_sec"]-inw["dep_sec"])/60
    results = []
    gcols = ["route_id"]+(["direction_id"] if "direction_id" in inw.columns else [])
    window_min = (we-ws)/60
    for keys, grp in inw.groupby(gcols):
        if isinstance(keys,str): keys=(keys,)
        rid = keys[0]; direction = keys[1] if len(keys)>1 else "0"
        deps = sorted(grp["dep_sec"].dropna().tolist())
        n = len(deps)
        if n < 1: continue
        avg_hw = (np.mean([deps[i+1]-deps[i] for i in range(n-1)])/60) if n>1 else window_min
        ow = grp["one_way_min"].dropna()
        avg_ow = ow.mean() if not ow.empty else np.nan
        cycle  = 2*avg_ow*1.15 if not np.isnan(avg_ow) else np.nan
        nveh   = math.ceil(cycle/avg_hw) if (not np.isnan(cycle) and avg_hw>0) else np.nan
        results.append({
            "route_id": rid, "direction_id": direction,
            "n_trips_in_window": n,
            "avg_headway_min":   round(avg_hw, 1),
            "avg_ow_trip_min":   round(avg_ow, 1) if not np.isnan(avg_ow) else None,
            "cycle_min":         round(cycle, 1)  if not np.isnan(cycle)  else None,
            "n_vehicles_est":    int(nveh) if not np.isnan(nveh) else None,
            "expected_wait_min": round(avg_hw/2, 1),
        })
    df = pd.DataFrame(results)
    if df.empty: return df
    rn = tables["routes"][["route_id","route_short_name","route_long_name"]].drop_duplicates()
    df = df.merge(rn, on="route_id", how="left")
    df["route_label"] = df["route_short_name"].fillna(df["route_id"])
    return df

def compute_savings_for_scenario(row, target_hw):
    # Support mapping from both baseline layout (avg_headway_min) and processed layout (current_headway)
    ch = float(row["current_headway"] if "current_headway" in row else row["avg_headway_min"])
    cw = float(row["current_wait"] if "current_wait" in row else row["expected_wait_min"])
    
    if target_hw is None:
        return {"scenario_headway_min":round(ch,1),"scenario_wait_min":round(cw,1),"wait_reduction_min":0.0,"buses_added":0}
    if target_hw >= ch:
        return {"scenario_headway_min":round(ch,1),"scenario_wait_min":round(cw,1),"wait_reduction_min":0.0,"buses_added":0}
    new_w = target_hw / 2
    buses = None
    cyc = row.get("cycle_min")
    if cyc is not None and not (isinstance(cyc, float) and np.isnan(cyc)):
        n_new = math.ceil(float(cyc) / target_hw)
        buses = max(0, n_new - (int(row["n_vehicles_est"]) if row["n_vehicles_est"] is not None else 0))
    return {"scenario_headway_min":round(target_hw,1),"scenario_wait_min":round(new_w,1),
            "wait_reduction_min":round(max(0.0, cw-new_w),1),"buses_added":buses}

def rank_routes(route_df, ridership_dict, target_hw):
    rows = []
    for _, r in route_df.iterrows():
        pax = float(ridership_dict.get(r["route_id"], 0))
        sim = compute_savings_for_scenario(r, target_hw)
        phs = sim["wait_reduction_min"] * pax / 60
        rows.append({
            "route_id":r["route_id"],"route_label":r["route_label"],"direction_id":r["direction_id"],
            "current_headway":r["avg_headway_min"],"current_wait":r["expected_wait_min"],
            "n_vehicles_est":r["n_vehicles_est"],
            "cycle_min":r["cycle_min"],
            "scenario_headway":sim["scenario_headway_min"],"scenario_wait":sim["scenario_wait_min"],
            "wait_reduction_min":sim["wait_reduction_min"],"buses_added":sim["buses_added"],
            "assumed_ridership":pax,"pax_hr_savings":round(phs,2),
        })
    out = pd.DataFrame(rows).sort_values("pax_hr_savings",ascending=False).reset_index(drop=True)
    out.index += 1
    return out

def plotly_yrt(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans",color="#d0d8e4"), margin=dict(t=40,b=40,l=20,r=20))
    fig.update_xaxes(gridcolor=YRT_BORDER, zerolinecolor=YRT_BORDER)
    fig.update_yaxes(gridcolor=YRT_BORDER, zerolinecolor=YRT_BORDER)
    return fig

def metric_card(label, value, delta=""):
    dh = f'<div class="delta">▲ {delta}</div>' if delta else ""
    st.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div>{dh}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# GRAVITY MODEL — cached, runs once per session
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cached_gravity_ridership(gtfs_bytes: bytes, n_samples: int):
    """Runs gravity O-D model and returns (route_ridership, od_samples, zones)."""
    return compute_gravity_ridership(gtfs_bytes, n_samples)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
logo_col, title_col = st.columns([1, 8])
with logo_col:
    if os.path.exists("yrt-logo.png"):
        st.image("yrt-logo.png", width=90)
    else:
        st.markdown(f'<div style="font-family:Space Mono;font-size:1.4rem;color:{YRT_DARK};font-weight:700;padding-top:8px">YRT</div>', unsafe_allow_html=True)
with title_col:
    st.markdown(f'<div style="padding-top:4px"><h1 style="margin:0;font-size:1.7rem;color:{YRT_WHITE}">Service Allocation Tool</h1><p style="margin:0;font-size:.85rem;color:{YRT_LIGHT};font-family:DM Sans,sans-serif">York Region Transit · GTFS Headway &amp; Gravity-Model Ridership Analysis</p></div>', unsafe_allow_html=True)
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if os.path.exists("yrt-logo.png"): st.image("yrt-logo.png", width=110)
    st.markdown("### Analysis Settings")
    analysis_date = st.date_input("Service date", value=datetime.today().date())
    c1, c2 = st.columns(2)
    with c1: window_start = st.time_input("From", value=dtime(7, 0))
    with c2: window_end   = st.time_input("To",   value=dtime(9, 0))
    st.markdown("---")
    st.markdown("### Frequency Scenario")
    freq_label = st.radio("Target headway", options=list(FREQ_OPTIONS.keys()), index=0)
    target_hw  = FREQ_OPTIONS[freq_label]
    if target_hw:
        st.markdown(f'<span class="freq-pill">{target_hw} min</span><span class="freq-pill">{60/target_hw:.0f} buses/hr</span>', unsafe_allow_html=True)
    st.markdown("---")
    n_od_samples = st.slider("O-D samples (gravity model)", 500, 5000, 2000, step=500,
        help="More samples = more accurate ridership estimates, slower first load. Cached after first run.")
    ridership_fallback = st.number_input("Fallback ridership (unsampled routes)",
        min_value=0, max_value=5000, value=200, step=50,
        help="Ridership for routes not reached by any sampled O-D pair.")
    st.markdown(f'<div class="info-box"><b>Ridership method</b><br>Gravity model: pop₁×pop₂/d² from 198 York Region census tracts. O-D pairs assigned to nearby GTFS stops/routes. Hourly profile applied to selected window.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD GTFS
# ═══════════════════════════════════════════════════════════════════════════════
ws = window_start.hour*3600 + window_start.minute*60
we = window_end.hour  *3600 + window_end.minute  *60
if we <= ws: st.error("Window end must be after window start."); st.stop()

with st.spinner("⬇ Fetching YRT GTFS feed…"):
    try:
        gtfs_bytes = fetch_yrt_gtfs()
    except Exception as e:
        st.error(f"Could not download YRT GTFS automatically ({e}). Upload manually:")
        up = st.file_uploader("Upload GTFS .zip", type="zip")
        if up is None: st.stop()
        gtfs_bytes = up.read()

with st.spinner("Parsing GTFS tables…"):
    tables = load_gtfs_bytes(gtfs_bytes)

# ── Run gravity model (cached) ────────────────────────────────────────────────
with st.spinner(f"🧮 Running gravity model ({n_od_samples} O-D samples)… cached after first run"):
    route_ridership_peak, od_samples, zones = cached_gravity_ridership(gtfs_bytes, n_od_samples)

gravity_available = bool(route_ridership_peak)

routes_df = tables["routes"]
st.success(
    f"✅  GTFS: **{routes_df['route_id'].nunique()} routes** · **{len(tables['trips'])} trips** "
    f"| Gravity model: **{len(route_ridership_peak)} routes** estimated from **{len(od_samples)} O-D pairs**"
)

# ── Route filter ──────────────────────────────────────────────────────────────
st.markdown("### Route Filter *(optional — leave blank for all)*")
all_opts = (routes_df.assign(label=lambda d: d["route_short_name"].fillna("")+" — "+d["route_long_name"].fillna(""))
            [["route_id","label"]].drop_duplicates().sort_values("label"))
label_to_id = dict(zip(all_opts["label"], all_opts["route_id"]))
sel_labels  = st.multiselect("Candidate routes", options=all_opts["label"].tolist(), default=[], placeholder="All routes")
sel_ids     = [label_to_id[l] for l in sel_labels] if sel_labels else None

# ── Compute headways ──────────────────────────────────────────────────────────
with st.spinner("Computing baseline headways…"):
    service_ids = get_active_service_ids(tables, analysis_date)
    if not service_ids:
        st.warning(f"No service IDs for {analysis_date}. Try a different date."); st.stop()
    wt = dict(tables)
    if sel_ids:
        ft = tables["trips"][tables["trips"]["route_id"].isin(sel_ids)]
        fs = tables["stop_times"][tables["stop_times"]["trip_id"].isin(ft["trip_id"])]
        wt["trips"] = ft; wt["stop_times"] = fs
    headway_df = compute_route_headways(wt, service_ids, ws, we)

if headway_df.empty:
    st.warning("No trips found in the selected window."); st.stop()

# ── Build window ridership from gravity peak × hourly norm ────────────────────
window_start_h = window_start.hour
window_end_h   = window_end.hour + (1 if window_end.minute > 0 else 0)

all_route_ids  = headway_df["route_id"].unique().tolist()
final_ridership = ridership_for_all_routes(
    all_route_ids, window_start_h, window_end_h,
    route_ridership_peak, fallback=float(ridership_fallback)
)

# ── Compute ranking once (used in multiple tabs) ───────────────────────────────
ranked = rank_routes(headway_df, final_ridership, target_hw)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_baseline, tab_ridership, tab_od, tab_ranking, tab_sensitivity, tab_accessibility = st.tabs([
    "📊 Baseline Service",
    "👥 Ridership Model",
    "🗺 O-D Zones",
    "🏆 Route Ranking",
    "🔁 Sensitivity",
    "♿ Accessibility",
])

# ───────────────────────────────────────────────────────
# TAB 1 — BASELINE SERVICE
# ───────────────────────────────────────────────────────
with tab_baseline:
    st.markdown("## Baseline Service Summary")
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Routes analysed",    str(headway_df["route_id"].nunique()))
    with c2: metric_card("Active service IDs", str(len(service_ids)))
    with c3: metric_card("Median headway",     f"{headway_df['avg_headway_min'].median():.0f} min")
    with c4: metric_card("Routes ≥ 30 min",    str(len(headway_df[headway_df["avg_headway_min"]>=30])))

    fig_hist = px.histogram(headway_df, x="avg_headway_min", nbins=30,
        labels={"avg_headway_min":"Average Headway (min)"},
        color_discrete_sequence=[YRT_DARK], template="plotly_dark",
        title="Headway Distribution")
    fig_hist.add_vline(x=30, line_dash="dash", line_color="#ff8c00",
                       annotation_text="30-min threshold", annotation_font_color="#ff8c00")
    if target_hw:
        fig_hist.add_vline(x=target_hw, line_dash="dot", line_color=YRT_LIGHT,
                           annotation_text=f"Target {target_hw} min", annotation_font_color=YRT_LIGHT)
    st.plotly_chart(plotly_yrt(fig_hist), use_container_width=True)

    # Stop map coloured by headway
    if "stops" in tables:
        stops = tables["stops"].copy()
        stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
        stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
        stops = stops.dropna(subset=["stop_lat","stop_lon"])
        stop_route = tables["stop_times"][["trip_id","stop_id"]].drop_duplicates()
        trip_route = tables["trips"][["trip_id","route_id"]].drop_duplicates()
        sr = stop_route.merge(trip_route, on="trip_id").merge(
             headway_df[["route_id","avg_headway_min"]].drop_duplicates(), on="route_id", how="left")
        stop_hw = sr.groupby("stop_id")["avg_headway_min"].mean().reset_index()
        stops = stops.merge(stop_hw, on="stop_id", how="left")
        fig_map = px.scatter_mapbox(
            stops.dropna(subset=["avg_headway_min"]).sample(min(3000,len(stops)), random_state=42),
            lat="stop_lat", lon="stop_lon", color="avg_headway_min",
            color_continuous_scale=[[0,YRT_LIGHT],[0.5,YRT_DARK],[1,"#ff4444"]],
            range_color=[0,60],
            hover_data={"stop_id":True,"avg_headway_min":":.0f"},
            zoom=9, height=480, mapbox_style="carto-darkmatter",
            title="Stop-Level Average Headway (min)",
            labels={"avg_headway_min":"Avg Headway (min)"},
        )
        st.plotly_chart(plotly_yrt(fig_map), use_container_width=True)

    with st.expander("🗂 Raw Headway Data"):
        st.dataframe(headway_df, use_container_width=True)

# ───────────────────────────────────────────────────────
# TAB 2 — RIDERSHIP MODEL
# ───────────────────────────────────────────────────────
with tab_ridership:
    st.markdown("## Gravity-Model Ridership Estimates")
    st.markdown(f"""
**How it works:**
1. **198 York Region census tracts** loaded from bundled GeoJSON. Population from the `pop` field.
2. Random O-D pairs sampled proportional to population (prob ∝ pop). Each pair uses a random interior point within the census tract polygon.
3. Gravity score: $G_{{ij}} = P_i \\times P_j \\ /\\ d_{{ij}}^2$ where $d$ is haversine distance.
4. Nearest GTFS stops (within 1.5 km) are found for each O-D endpoint; their routes accumulate the gravity score.
5. Route scores normalised so the maximum maps to **{3950} daily pax** (YRT peak-hour reference).
6. Window ridership = peak estimate × sum of hourly normalisation factors for the selected hours.
""")

    # Hourly profile
    st.markdown("### YRT Hourly Demand Profile")
    hp = hourly_profile_chart_data()
    fig_hp = px.bar(hp, x="Hour", y="AvgDaily_Total",
        color="Normalized_Ridership",
        color_continuous_scale=[[0,YRT_DARK],[1,YRT_LIGHT]],
        labels={"AvgDaily_Total":"Avg Daily Boardings","Normalized_Ridership":"Normalised"},
        template="plotly_dark", title="Hourly Demand Profile (peak = hour 17)")
    if window_start_h < window_end_h:
        fig_hp.add_vrect(x0=window_start_h-0.5, x1=window_end_h-0.5,
                         fillcolor=YRT_LIGHT, opacity=0.12, line_width=0,
                         annotation_text="Selected window", annotation_position="top left",
                         annotation_font_color=YRT_LIGHT)
    st.plotly_chart(plotly_yrt(fig_hp), use_container_width=True)

    # Gravity ridership bar — top 30
    st.markdown("### Estimated Window Ridership by Route (Top 30)")
    rid_rows = []
    for r in all_route_ids:
        lbl = headway_df.loc[headway_df["route_id"]==r, "route_label"]
        lbl = lbl.iloc[0] if not lbl.empty else r
        rid_rows.append({"route_id":r, "route_label":lbl, "window_ridership": final_ridership.get(r, 0)})
    rid_df = pd.DataFrame(rid_rows).sort_values("window_ridership", ascending=False).head(30)

    fig_rid = px.bar(rid_df, x="route_label", y="window_ridership",
        color="window_ridership",
        color_continuous_scale=[[0,YRT_DARK],[1,YRT_LIGHT]],
        labels={"route_label":"Route","window_ridership":"Estimated Pax (window)"},
        template="plotly_dark", text="window_ridership",
        title=f"Gravity-Model Ridership · {window_start.strftime('%H:%M')}–{window_end.strftime('%H:%M')}")
    fig_rid.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_rid.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_tickangle=-35)
    st.plotly_chart(plotly_yrt(fig_rid), use_container_width=True)

    st.markdown(f"""<div class="info-box">
<b>Coverage:</b> {len(route_ridership_peak)} / {len(all_route_ids)} routes reached by at least one O-D pair.
Routes not reached use the fallback value ({ridership_fallback} pax).
Increase O-D samples in the sidebar for better coverage.
</div>""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────
# TAB 3 — O-D ZONES
# ───────────────────────────────────────────────────────
with tab_od:
    st.markdown("## Origin–Destination Sampling Map")
    st.caption("Gravity-weighted O-D pairs from York Region census tracts. Lines connect 80 sample pairs.")

    if od_samples:
        od_df = pd.DataFrame(od_samples)
        od_show = od_df.sample(min(300, len(od_df)), random_state=42)

        fig_od = go.Figure()
        fig_od.add_trace(go.Scattermapbox(
            lat=od_show["o_lat"], lon=od_show["o_lon"], mode="markers",
            marker=dict(size=6, color=YRT_LIGHT, opacity=0.65),
            name="Origins",
            hovertemplate="Origin · pop: %{customdata[0]:,}<extra></extra>",
            customdata=od_show[["o_pop"]].values,
        ))
        fig_od.add_trace(go.Scattermapbox(
            lat=od_show["d_lat"], lon=od_show["d_lon"], mode="markers",
            marker=dict(size=6, color="#ff8c00", opacity=0.65),
            name="Destinations",
            hovertemplate="Destination · pop: %{customdata[0]:,}<extra></extra>",
            customdata=od_show[["d_pop"]].values,
        ))
        for _, row in od_show.head(80).iterrows():
            fig_od.add_trace(go.Scattermapbox(
                lat=[row["o_lat"], row["d_lat"], None],
                lon=[row["o_lon"], row["d_lon"], None],
                mode="lines",
                line=dict(width=0.7, color="rgba(0,163,224,0.2)"),
                showlegend=False, hoverinfo="skip",
            ))

        # Census tract centroids sized by population
        if zones:
            zone_df = pd.DataFrame(zones)
            fig_od.add_trace(go.Scattermapbox(
                lat=zone_df["lat"], lon=zone_df["lon"], mode="markers",
                marker=dict(
                    size=np.clip(zone_df["pop"] / 500, 3, 14).tolist(),
                    color=YRT_DARK, opacity=0.4,
                ),
                name="Census tracts (size ∝ pop)",
                hovertemplate="%{customdata[0]}<br>pop: %{customdata[1]:,}<extra></extra>",
                customdata=zone_df[["id","pop"]].values,
            ))

        fig_od.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox_zoom=9,
            mapbox_center={"lat": 43.95, "lon": -79.45},
            height=540,
            margin=dict(t=0,b=0,l=0,r=0),
            legend=dict(orientation="h", y=0.01, x=0.01,
                        bgcolor="rgba(3,16,31,0.8)", font=dict(color=YRT_WHITE)),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_od, use_container_width=True)

        st.markdown(f"""
**O-D stats** · {len(od_df)} pairs sampled · {len(zones)} census tracts · showing {len(od_show)} on map

| Metric | Value |
|---|---|
| Median trip distance | {od_df['dist_km'].median():.1f} km |
| Max gravity score | {od_df['gravity'].max():,.0f} |
| Median routes per pair | {od_df['routes'].apply(len).median():.0f} |
""")
    else:
        st.warning("No O-D samples available — check that `data/york_census.geojson` is present.")

# ───────────────────────────────────────────────────────
# TAB 4 — ROUTE RANKING
# ───────────────────────────────────────────────────────
with tab_ranking:
    st.markdown(f"## Route Ranking · {freq_label}")

    if ranked.empty or ranked["pax_hr_savings"].sum() == 0:
        st.info("No routes show positive savings under this scenario."); st.stop()

    # Top 3 cards
    st.markdown("### 🏆 Top Candidates")
    top3   = ranked[ranked["pax_hr_savings"]>0].head(3)
    medals = ["🥇","🥈","🥉"]
    cols   = st.columns(max(len(top3),1))
    for i, (_,row) in enumerate(top3.iterrows()):
        with cols[i]:
            ba = row["buses_added"]
            buses_str = f"+{int(ba)} bus(es)" if (ba is not None and ba not in (0, np.nan)) else ""
            st.markdown(f"""
            <div class="metric-card rank-{i+1}">
              <div class="label">{medals[i]} Rank {i+1} {buses_str}</div>
              <div class="value">{row['route_label']}</div>
              <div style="color:{YRT_LIGHT};font-size:.9rem;margin-top:4px">{row['pax_hr_savings']:.1f} pax-hrs saved</div>
              <div style="color:#a0a0a0;font-size:.8rem;margin-top:4px">
                Headway: {row['current_headway']} → {row['scenario_headway']} min<br>
                Wait: {row['current_wait']} → {row['scenario_wait']} min<br>
                Ridership (est.): {row['assumed_ridership']:.0f} pax
              </div>
            </div>""", unsafe_allow_html=True)

    # Full table
    st.markdown("### Full Ranking")
    dcols = {
        "route_label":"Route","direction_id":"Dir",
        "current_headway":"Curr Hdwy","scenario_headway":"Scen Hdwy",
        "current_wait":"Curr Wait","scenario_wait":"Scen Wait",
        "wait_reduction_min":"Wait Saved/pax","buses_added":"Buses Added",
        "assumed_ridership":"Ridership (est.)","pax_hr_savings":"Pax-Hr Savings",
    }
    disp = ranked[list(dcols.keys())].rename(columns=dcols)
    st.dataframe(disp.style.format({
        "Pax-Hr Savings":"{:.2f}","Wait Saved/pax":"{:.1f}",
        "Curr Hdwy":"{:.1f}","Scen Hdwy":"{:.1f}","Ridership (est.)":"{:.0f}",
    }), use_container_width=True, height=420)

    # Bar chart
    fig_bar = px.bar(ranked[ranked["pax_hr_savings"]>0].head(20),
        x="route_label", y="pax_hr_savings", color="pax_hr_savings",
        color_continuous_scale=[[0,YRT_DARK],[1,YRT_LIGHT]],
        labels={"route_label":"Route","pax_hr_savings":"Pax-Hr Savings"},
        template="plotly_dark", text="pax_hr_savings",
        title="Top 20 Routes · Passenger-Hour Savings")
    fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_bar.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_tickangle=-35)
    st.plotly_chart(plotly_yrt(fig_bar), use_container_width=True)

    # Bubble chart
    fig_sc = px.scatter(ranked[ranked["pax_hr_savings"]>0],
        x="current_headway", y="assumed_ridership",
        size=ranked[ranked["pax_hr_savings"]>0]["pax_hr_savings"].clip(lower=0.01),
        color="pax_hr_savings",
        hover_name="route_label",
        hover_data={"pax_hr_savings":":.2f","assumed_ridership":":.0f"},
        color_continuous_scale=[[0,YRT_DARK],[1,YRT_LIGHT]],
        labels={"current_headway":"Current Headway (min)","assumed_ridership":"Est. Ridership"},
        template="plotly_dark", size_max=50,
        title="Route Priority Map · Headway vs. Ridership (bubble = pax-hr savings)")
    st.plotly_chart(plotly_yrt(fig_sc), use_container_width=True)

    st.download_button("⬇ Download ranking CSV",
        data=ranked.to_csv(index=True).encode(),
        file_name="yrt_route_ranking.csv", mime="text/csv",
        key="tab_download_button")

# ───────────────────────────────────────────────────────
# TAB 5 — SENSITIVITY (uses `ranked` computed above tabs)
# ───────────────────────────────────────────────────────
with tab_sensitivity:
    st.markdown("## Sensitivity: All Frequency Scenarios")
    st.caption("How does the top-ranked route respond across all 5 frequency tiers? And how do the top 15 compare?")

    if ranked.empty:
        st.info("No ranking data available."); st.stop()

    top_route = ranked.iloc[0]

    # ── Single-route scenario comparison ─────────────────────────────────
    sens_rows = []
    for lbl, hw in FREQ_OPTIONS.items():
        sim = compute_savings_for_scenario(top_route, hw)
        pax = float(final_ridership.get(str(top_route["route_id"]), 0))
        phs = sim["wait_reduction_min"] * pax / 60
        sens_rows.append({
            "Scenario":        lbl,
            "Target Hdwy":     hw if hw is not None else float(top_route["current_headway"]),
            "New Wait (min)":  sim["scenario_wait_min"],
            "Pax-Hr Savings":  round(phs, 2),
        })
    sens_df = pd.DataFrame(sens_rows)

    fig_s = make_subplots(specs=[[{"secondary_y": True}]])
    fig_s.add_trace(go.Bar(
        x=sens_df["Scenario"], y=sens_df["Pax-Hr Savings"],
        name="Pax-Hr Savings", marker_color=YRT_DARK,
    ), secondary_y=False)
    fig_s.add_trace(go.Scatter(
        x=sens_df["Scenario"], y=sens_df["New Wait (min)"],
        name="New Wait (min)", mode="lines+markers",
        line=dict(color=YRT_LIGHT, width=2),
        marker=dict(size=8, color=YRT_LIGHT),
    ), secondary_y=True)
    fig_s.update_layout(
        title=f"Scenario Comparison — Route {top_route['route_label']}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#d0d8e4"),
        legend=dict(orientation="h", y=1.12),
        xaxis_tickangle=-15,
        margin=dict(t=60, b=80, l=20, r=20),
    )
    fig_s.update_yaxes(title_text="Pax-Hr Savings", secondary_y=False,
                        gridcolor=YRT_BORDER, zerolinecolor=YRT_BORDER)
    fig_s.update_yaxes(title_text="Wait Time (min)", secondary_y=True,
                        gridcolor=YRT_BORDER)
    st.plotly_chart(fig_s, use_container_width=True)

    # ── Heatmap: top 15 routes × 5 scenarios ──────────────────────────────
    st.markdown("### Scenario Heatmap — Top 15 Routes")
    top15 = ranked[ranked["pax_hr_savings"] > 0].head(15)

    if top15.empty:
        st.info("No routes with positive savings to display.")
    else:
        heat_data = []
        for _, r in top15.iterrows():
            pax = float(final_ridership.get(str(r["route_id"]), 0))
            for lbl, hw in FREQ_OPTIONS.items():
                sim = compute_savings_for_scenario(r, hw)
                phs = sim["wait_reduction_min"] * pax / 60
                heat_data.append({
                    "Route":    r["route_label"],
                    "Scenario": lbl,
                    "Pax-Hr Savings": round(phs, 2),
                })

        heat_df = (
            pd.DataFrame(heat_data)
            .pivot(index="Route", columns="Scenario", values="Pax-Hr Savings")
        )
        # Reorder columns to match FREQ_OPTIONS key order
        ordered_cols = [k for k in FREQ_OPTIONS.keys() if k in heat_df.columns]
        heat_df = heat_df[ordered_cols]

        fig_heat = px.imshow(
            heat_df,
            color_continuous_scale=[[0,"#03101f"],[0.3,YRT_DARK],[1,YRT_LIGHT]],
            labels=dict(color="Pax-Hr Savings"),
            aspect="auto", template="plotly_dark",
            title="Pax-Hr Savings Heatmap",
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#d0d8e4"),
            margin=dict(t=50, b=100, l=20, r=20),
            xaxis_tickangle=-20,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ───────────────────────────────────────────────────────
# TAB 6 — ACCESSIBILITY (r5py, offline)
# ───────────────────────────────────────────────────────
with tab_accessibility:
    st.markdown("## Population-Weighted Accessibility")
    st.markdown(r"""
**Method (r5py — offline pre-computation)**

For each census block centroid $i$:
$$A_i = \sum_j \frac{P_j}{t_{ij}}$$
where $P_j$ = population of zone $j$, $t_{ij}$ = transit travel time (minutes).

Network-wide score:
$$\bar{A} = \frac{\sum_i P_i \cdot A_i}{\sum_i P_i}$$

Computed by `precompute_ridership.py --osm york_region.osm.pbf` and stored in `data/accessibility.json`.
""")

    acc_path = Path("data/accessibility.json")
    if acc_path.exists():
        with open(acc_path) as f:
            accessibility = json.load(f)
        acc_df = pd.DataFrame(accessibility)
        acc_df["weighted_accessibility"] = pd.to_numeric(acc_df["weighted_accessibility"], errors="coerce")
        acc_df = acc_df.dropna(subset=["weighted_accessibility","lat","lon"])

        pop_col = acc_df.get("pop") if "pop" in acc_df.columns else None
        if pop_col is not None:
            pop_w_acc = (acc_df["weighted_accessibility"] * acc_df["pop"]).sum() / acc_df["pop"].sum()
        else:
            pop_w_acc = acc_df["weighted_accessibility"].mean()
        metric_card("Population-weighted avg accessibility", f"{pop_w_acc:,.0f}")

        fig_acc = px.scatter_mapbox(acc_df,
            lat="lat", lon="lon", color="weighted_accessibility", size="weighted_accessibility",
            size_max=20,
            color_continuous_scale=[[0,"#03101f"],[0.4,YRT_DARK],[1,YRT_LIGHT]],
            zoom=9, height=500, mapbox_style="carto-darkmatter",
            title="Zone-Level Transit Accessibility (r5py)",
            labels={"weighted_accessibility":"Accessibility Score"},
        )
        st.plotly_chart(plotly_yrt(fig_acc), use_container_width=True)
    else:
        st.markdown(f"""
<div class="warning-box">
No <code>data/accessibility.json</code> found. Generate it with r5py + Java 11:<br><br>
<pre>python precompute_ridership.py \\
    --geojson data/york_census.geojson \\
    --gtfs    google_transit.zip \\
    --osm     york_region.osm.pbf \\
    --out-dir data/</pre>
OSM extract: <a href="https://download.geofabrik.de/north-america/canada/ontario.html" target="_blank" style="color:{YRT_LIGHT}">Geofabrik Ontario</a>, clipped to York Region.
</div>""", unsafe_allow_html=True)

        # Show zone population map as a proxy
        if zones:
            st.markdown("### Census Tract Population (proxy for demand density)")
            zone_df = pd.DataFrame(zones)
            fig_z = px.scatter_mapbox(zone_df,
                lat="lat", lon="lon", color="pop", size="pop",
                size_max=18,
                color_continuous_scale=[[0,YRT_DARK],[1,YRT_LIGHT]],
                zoom=9, height=460, mapbox_style="carto-darkmatter",
                labels={"pop":"Population"},
                title="Census Tract Population · York Region",
            )
            st.plotly_chart(plotly_yrt(fig_z), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
with st.expander("📐 Full Methodology"):
    st.markdown(r"""
**Gravity model:** $G_{ij} = P_i \cdot P_j \;/\; d_{ij}^2$ · random interior sampling from census tract polygons (no geopandas required).

**Route assignment:** nearest GTFS stops within 1.5 km of each O-D endpoint; those stops' routes accumulate the gravity score.

**Peak normalisation:** max-route gravity → 3950 daily pax (YRT peak ref). Hourly: $R_h = R_{peak} \times norm_h$.

**Baseline headway:** $H_r$ = mean consecutive departure gap within window.

**Expected wait:** $W_r = H_r / 2$

**Scenario headway:** target tier (5/10/15/30 min). Routes already below target → zero savings.

**Pax-hr savings:** $PHS_r = (W_r - W_{new}) \times R_{window} \;/\; 60$

**r5py accessibility (offline):** $A_i = \sum_j P_j / t_{ij}$, population-weighted.
""")

col_dl, _ = st.columns([1, 3])
with col_dl:
    st.download_button("⬇ Download ranking CSV",
        data=ranked.to_csv(index=True).encode(),
        file_name="yrt_route_ranking.csv", mime="text/csv",
        key="footer_download_button")

st.caption("YRT Service Allocation Tool · Proof-of-concept · Not a replacement for operational service planning.")
