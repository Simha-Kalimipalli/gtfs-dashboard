"""
precompute_ridership.py
=======================
Offline script — run ONCE before deploying, outputs:
  data/route_ridership.json   — gravity-model ridership per route_id
  data/od_samples.json        — sampled O-D pairs with route assignments (for map display)

Usage:
  python precompute_ridership.py \
      --geojson  york_region_census.geojson \
      --gtfs     google_transit.zip \
      --date     2025-09-16 \
      --samples  2000 \
      --out-dir  data/

Dependencies (install separately, not needed in Streamlit runtime):
  pip install geopandas shapely r5py pyrosm requests tqdm
  # r5py also needs Java 11+ on PATH
"""

import argparse
import json
import math
import os
import random
import zipfile
import io
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape
from tqdm import tqdm

# ── Hourly normalisation table (from brief) ────────────────────────────────
HOURLY_NORM = {
     0: 0.0253,  1: 0.0127,  2: 0.0076,  3: 0.0380,  4: 0.0759,
     5: 0.1392,  6: 0.2785,  7: 0.5063,  8: 0.6709,  9: 0.5823,
    10: 0.5190, 11: 0.5063, 12: 0.7215, 13: 0.6456, 14: 0.6835,
    15: 0.9114, 16: 0.9620, 17: 1.0000, 18: 0.9747, 19: 0.7468,
    20: 0.6582, 21: 0.6076, 22: 0.5696, 23: 0.4937,
}

# York Region bounding box (rough)
YR_BBOX = dict(min_lon=-79.80, max_lon=-79.00, min_lat=43.70, max_lat=44.25)


# ── Helpers ────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def gravity(pop1, pop2, dist_km, beta=2.0, min_dist=0.5):
    """Gravity model: interaction ∝ pop1 * pop2 / dist^beta."""
    d = max(dist_km, min_dist)
    return (pop1 * pop2) / (d ** beta)


def random_point_in_polygon(polygon):
    """Sample a random point inside a Shapely polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(500):
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            return p
    return polygon.centroid  # fallback


def load_gtfs_stops(gtfs_path):
    """Return stops DataFrame with lat/lon from a GTFS zip."""
    with zipfile.ZipFile(gtfs_path) as zf:
        names = zf.namelist()
        prefix = ""
        if "stops.txt" not in names:
            for n in names:
                if n.endswith("stops.txt"):
                    prefix = n.replace("stops.txt", "")
                    break
        with zf.open(prefix + "stops.txt") as f:
            stops = pd.read_csv(f, dtype=str)
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    return stops.dropna(subset=["stop_lat", "stop_lon"])


def load_gtfs_stop_routes(gtfs_path):
    """Return a dict: stop_id → list of route_ids that serve it."""
    tables = {}
    needed = ["trips.txt", "stop_times.txt", "routes.txt"]
    with zipfile.ZipFile(gtfs_path) as zf:
        names = zf.namelist()
        prefix = ""
        if "trips.txt" not in names:
            for n in names:
                if n.endswith("trips.txt"):
                    prefix = n.replace("trips.txt", "")
                    break
        for fname in needed:
            with zf.open(prefix + fname) as f:
                tables[fname.replace(".txt", "")] = pd.read_csv(f, dtype=str)

    trip_route = tables["trips"][["trip_id", "route_id"]].drop_duplicates()
    st = tables["stop_times"][["trip_id", "stop_id"]].drop_duplicates()
    merged = st.merge(trip_route, on="trip_id")
    stop_routes = merged.groupby("stop_id")["route_id"].apply(list).to_dict()
    return stop_routes


def nearest_stop(lat, lon, stops_df, k=3):
    """Return the k nearest stop_ids to (lat, lon)."""
    stops_df = stops_df.copy()
    stops_df["dist"] = stops_df.apply(
        lambda r: haversine_km(lat, lon, r["stop_lat"], r["stop_lon"]), axis=1
    )
    return stops_df.nsmallest(k, "dist")["stop_id"].tolist()


# ── Main ───────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GeoJSON census blocks…")
    gdf = gpd.read_file(args.geojson)

    # Normalise column names (case-insensitive search)
    col_map = {c.lower(): c for c in gdf.columns}
    pop_col = next(
        (col_map[k] for k in col_map
         if any(k.startswith(p) for p in ["pop", "population", "total_pop"])),
        None
    )
    if pop_col is None:
        raise ValueError(
            "Cannot find a population column. "
            f"Available columns: {list(gdf.columns)}"
        )

    uid_col = col_map.get("geouid", col_map.get("uid", col_map.get("dauid", None)))
    if uid_col is None:
        # fall back: use index
        gdf["_uid"] = gdf.index.astype(str)
        uid_col = "_uid"

    gdf = gdf.to_crs(epsg=4326)
    gdf["pop"] = pd.to_numeric(gdf[pop_col], errors="coerce").fillna(0)
    gdf["centroid_lat"] = gdf.geometry.centroid.y
    gdf["centroid_lon"] = gdf.geometry.centroid.x

    # Clip to York Region bounding box
    gdf = gdf[
        (gdf["centroid_lon"] >= YR_BBOX["min_lon"]) &
        (gdf["centroid_lon"] <= YR_BBOX["max_lon"]) &
        (gdf["centroid_lat"] >= YR_BBOX["min_lat"]) &
        (gdf["centroid_lat"] <= YR_BBOX["max_lat"])
    ].copy()
    print(f"  {len(gdf)} census blocks in York Region bounding box.")

    # Filter to blocks with population
    gdf_pop = gdf[gdf["pop"] > 0].reset_index(drop=True)
    pops = gdf_pop["pop"].values
    prob = pops / pops.sum()   # sampling probability ∝ population

    print("Loading GTFS stops and stop→route mapping…")
    stops_df      = load_gtfs_stops(args.gtfs)
    stop_routes   = load_gtfs_stop_routes(args.gtfs)

    print(f"Sampling {args.samples} O-D pairs (gravity-weighted)…")
    od_records = []
    route_interaction_sum = {}  # route_id → total gravity score

    rng = np.random.default_rng(42)

    for _ in tqdm(range(args.samples)):
        # Sample origin block (∝ population)
        o_idx = rng.choice(len(gdf_pop), p=prob)
        # Sample destination block (∝ population, different from origin)
        d_candidates = np.arange(len(gdf_pop))
        d_prob = prob.copy()
        d_prob[o_idx] = 0
        d_prob /= d_prob.sum()
        d_idx = rng.choice(len(gdf_pop), p=d_prob)

        o_row = gdf_pop.iloc[o_idx]
        d_row = gdf_pop.iloc[d_idx]

        # Random point inside each polygon
        o_pt = random_point_in_polygon(o_row.geometry)
        d_pt = random_point_in_polygon(d_row.geometry)

        dist_km = haversine_km(o_pt.y, o_pt.x, d_pt.y, d_pt.x)
        grav    = gravity(o_row["pop"], d_row["pop"], dist_km)

        # Nearest stops to origin and destination
        o_stops = nearest_stop(o_pt.y, o_pt.x, stops_df, k=3)
        d_stops = nearest_stop(d_pt.y, d_pt.x, stops_df, k=3)

        # Routes serving origin stops
        candidate_routes = set()
        for sid in o_stops + d_stops:
            candidate_routes.update(stop_routes.get(sid, []))

        # Accumulate gravity score per route
        for r in candidate_routes:
            route_interaction_sum[r] = route_interaction_sum.get(r, 0.0) + grav

        od_records.append({
            "o_lat": round(o_pt.y, 6), "o_lon": round(o_pt.x, 6),
            "d_lat": round(d_pt.y, 6), "d_lon": round(d_pt.x, 6),
            "o_pop": int(o_row["pop"]), "d_pop": int(d_row["pop"]),
            "dist_km": round(dist_km, 3),
            "gravity": round(grav, 2),
            "o_uid":  str(o_row[uid_col]),
            "d_uid":  str(d_row[uid_col]),
            "routes": list(candidate_routes)[:10],  # cap for JSON size
        })

    # Normalise route gravity sums to a [0, max_daily_ridership] range.
    # We scale so the route with the most interactions maps to 3950 (peak daily)
    # and others scale proportionally.
    max_grav = max(route_interaction_sum.values()) if route_interaction_sum else 1
    SCALE_PEAK_DAILY = 3950   # peak-hour reference from the normalisation table

    route_ridership_raw = {}
    for route_id, g in route_interaction_sum.items():
        route_ridership_raw[route_id] = round((g / max_grav) * SCALE_PEAK_DAILY, 1)

    # Build hourly ridership per route
    route_ridership_hourly = {}
    for route_id, peak_val in route_ridership_raw.items():
        route_ridership_hourly[route_id] = {
            str(h): round(peak_val * norm, 1)
            for h, norm in HOURLY_NORM.items()
        }

    # Also save aggregate daily ridership (sum over all hours)
    route_ridership_daily = {
        r: round(sum(hv.values()), 0)
        for r, hv in route_ridership_hourly.items()
    }

    # ── r5py accessibility (optional — requires r5py + Java) ──────────────────
    r5_available = False
    accessibility_df = None
    try:
        import r5py
        r5_available = True
        print("r5py found — computing accessibility matrix…")
        # Build transport network
        network = r5py.TransportNetwork(
            args.gtfs,
            [args.osm] if hasattr(args, "osm") and args.osm else [],
        )

        # Use census block centroids as both origins and destinations
        origins = gdf_pop[["centroid_lat", "centroid_lon", uid_col, "pop"]].rename(
            columns={"centroid_lat": "lat", "centroid_lon": "lon", uid_col: "id"}
        ).copy()

        # Travel time matrix (transit) for the analysis hour
        analysis_hour = args.hour if hasattr(args, "hour") else 8
        departure = pd.Timestamp(args.date) + pd.Timedelta(hours=analysis_hour)

        tt_computer = r5py.TravelTimeMatrixComputer(
            network,
            origins=origins,
            destinations=origins,
            departure=departure,
            transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
            max_time=pd.Timedelta(minutes=90),
        )
        tt_matrix = tt_computer.compute_travel_times()

        # Population-weighted average accessibility:
        # For each origin i: A_i = sum_j(pop_j / tt_ij) weighted by pop_i
        pop_lookup = dict(zip(origins["id"], origins["pop"]))
        tt_matrix["dest_pop"] = tt_matrix["to_id"].map(pop_lookup).fillna(0)
        tt_matrix["tt_min"]   = pd.to_numeric(tt_matrix["travel_time"], errors="coerce")
        tt_matrix = tt_matrix[(tt_matrix["tt_min"] > 0) & (~tt_matrix["tt_min"].isna())]
        tt_matrix["accessibility"] = tt_matrix["dest_pop"] / tt_matrix["tt_min"]

        acc_by_origin = (
            tt_matrix.groupby("from_id")["accessibility"].sum().reset_index()
            .rename(columns={"from_id": "id", "accessibility": "weighted_accessibility"})
        )
        origins = origins.merge(acc_by_origin, on="id", how="left")
        accessibility_df = origins[["id", "lat", "lon", "pop", "weighted_accessibility"]].to_dict(orient="records")
        print(f"  Accessibility computed for {len(accessibility_df)} zones.")

    except ImportError:
        print("r5py not available — skipping accessibility matrix.")
        print("Install r5py + Java 11 to enable this feature.")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("Writing outputs…")

    with open(out_dir / "route_ridership.json", "w") as f:
        json.dump({
            "metadata": {
                "n_od_samples":   args.samples,
                "geojson_source": args.geojson,
                "gtfs_source":    args.gtfs,
                "date":           args.date,
                "scale_peak":     SCALE_PEAK_DAILY,
                "method":         "gravity model (pop1*pop2/dist^2), nearest-stop route assignment",
            },
            "daily_ridership":  route_ridership_daily,
            "hourly_ridership": route_ridership_hourly,
        }, f, indent=2)

    # Save a sample of O-D pairs (cap at 500 for display)
    display_sample = random.sample(od_records, min(500, len(od_records)))
    with open(out_dir / "od_samples.json", "w") as f:
        json.dump(display_sample, f, indent=2)

    if accessibility_df:
        with open(out_dir / "accessibility.json", "w") as f:
            json.dump(accessibility_df, f, indent=2)

    print(f"\n✅  Done.")
    print(f"   Routes with ridership estimates : {len(route_ridership_daily)}")
    print(f"   O-D sample records saved        : {len(display_sample)}")
    if r5_available:
        print(f"   Accessibility zones             : {len(accessibility_df)}")
    print(f"\n   Output directory: {out_dir.resolve()}")
    print("\nNext step: commit data/ to your repo and re-deploy Streamlit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute gravity-model ridership for YRT routes.")
    parser.add_argument("--geojson",  required=True,  help="Path to York Region census GeoJSON")
    parser.add_argument("--gtfs",     required=True,  help="Path to YRT GTFS .zip")
    parser.add_argument("--date",     default="2025-09-16", help="Analysis date YYYY-MM-DD")
    parser.add_argument("--samples",  type=int, default=2000, help="Number of O-D pairs to sample")
    parser.add_argument("--out-dir",  default="data/", help="Output directory")
    parser.add_argument("--osm",      default=None,   help="Path to .osm.pbf for r5py (optional)")
    parser.add_argument("--hour",     type=int, default=8, help="Departure hour for r5py matrix")
    args = parser.parse_args()
    main(args)
