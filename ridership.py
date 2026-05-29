"""
ridership.py
============
Gravity-model ridership estimates from bundled GeoJSON + GTFS data.
No geopandas/shapely required — pure stdlib + numpy + pandas.

Loaded by app.py at runtime via compute_gravity_ridership() which is
cached by Streamlit so it only runs once per session.
"""

import json, math, random, zipfile, io
from pathlib import Path

import numpy as np
import pandas as pd

# ── Hourly normalisation table ─────────────────────────────────────────────
HOURLY_NORM: dict[int, float] = {
     0: 0.0253,  1: 0.0127,  2: 0.0076,  3: 0.0380,  4: 0.0759,
     5: 0.1392,  6: 0.2785,  7: 0.5063,  8: 0.6709,  9: 0.5823,
    10: 0.5190, 11: 0.5063, 12: 0.7215, 13: 0.6456, 14: 0.6835,
    15: 0.9114, 16: 0.9620, 17: 1.0000, 18: 0.9747, 19: 0.7468,
    20: 0.6582, 21: 0.6076, 22: 0.5696, 23: 0.4937,
}
HOURLY_DF = pd.DataFrame([
    {"Hour": h, "Normalized_Ridership": v, "AvgDaily_Total": round(v * 3950)}
    for h, v in HOURLY_NORM.items()
])

DATA_DIR   = Path(__file__).parent / "data"
GEOJSON_PATH = DATA_DIR / "york_census.geojson"
SCALE_PEAK_DAILY = 3950   # peak-hour reference (hour 17)
N_OD_SAMPLES     = 3000   # pre-computed O-D pairs; more = better accuracy, slower first load
RANDOM_SEED      = 42


# ── Geometry helpers (no shapely) ──────────────────────────────────────────

def _ring_centroid(coords):
    """Mean of ring vertices — good enough for census tract centroids."""
    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    return sum(lats) / len(lats), sum(lons) / len(lons)

def _geom_centroid(geom):
    if geom["type"] == "Polygon":
        return _ring_centroid(geom["coordinates"][0])
    elif geom["type"] == "MultiPolygon":
        # use the ring with the most vertices (largest polygon)
        best = max(geom["coordinates"], key=lambda p: len(p[0]))
        return _ring_centroid(best[0])
    return None, None

def _point_in_ring(ring):
    """
    Sample a random point inside a polygon ring using rejection sampling.
    Falls back to centroid after 200 attempts.
    """
    lons = [c[0] for c in ring]; lats = [c[1] for c in ring]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    # Ray-casting point-in-polygon
    def pip(px, py):
        inside = False
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if ((yi > py) != (yj > py)) and (px < (xj-xi)*(py-yi)/(yj-yi+1e-12)+xi):
                inside = not inside
            j = i
        return inside
    for _ in range(200):
        px = random.uniform(min_lon, max_lon)
        py = random.uniform(min_lat, max_lat)
        if pip(px, py):
            return py, px   # lat, lon
    return (min_lat+max_lat)/2, (min_lon+max_lon)/2   # centroid fallback

def _sample_point_in_geom(geom):
    """Sample a random interior point from a Polygon or MultiPolygon."""
    if geom["type"] == "Polygon":
        return _point_in_ring(geom["coordinates"][0])
    elif geom["type"] == "MultiPolygon":
        # pick largest ring
        best = max(geom["coordinates"], key=lambda p: len(p[0]))
        return _point_in_ring(best[0])
    return _geom_centroid(geom)


# ── Distance ───────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    f1, f2 = math.radians(lat1), math.radians(lat2)
    df = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(df/2)**2 + math.cos(f1)*math.cos(f2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ── GTFS nearest-stop helpers ──────────────────────────────────────────────

def _load_stops_and_routes(gtfs_bytes: bytes):
    """
    From raw GTFS bytes return:
      stops_arr  — list of (stop_id, lat, lon)
      stop_routes — {stop_id: [route_id, ...]}
    """
    tables = {}
    needed = ["stops.txt", "trips.txt", "stop_times.txt"]
    with zipfile.ZipFile(io.BytesIO(gtfs_bytes)) as zf:
        names = zf.namelist()
        prefix = ""
        if "stops.txt" not in names:
            for n in names:
                if n.endswith("stops.txt"):
                    prefix = n.replace("stops.txt", ""); break
        for fname in needed:
            path = prefix + fname
            if path in names:
                with zf.open(path) as f:
                    tables[fname.replace(".txt", "")] = pd.read_csv(f, dtype=str)

    stops = tables["stops"].copy()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    stops_arr = list(zip(stops["stop_id"], stops["stop_lat"], stops["stop_lon"]))

    trip_route = tables["trips"][["trip_id", "route_id"]].drop_duplicates()
    st = tables["stop_times"][["trip_id", "stop_id"]].drop_duplicates()
    merged = st.merge(trip_route, on="trip_id")
    stop_routes = merged.groupby("stop_id")["route_id"].apply(list).to_dict()

    return stops_arr, stop_routes


def _nearest_routes(lat, lon, stops_arr, stop_routes, k=3, max_km=1.5):
    """Return route_ids for the k nearest stops within max_km."""
    dists = [(haversine_km(lat, lon, slat, slon), sid)
             for sid, slat, slon in stops_arr]
    dists.sort()
    routes = set()
    for d, sid in dists[:k]:
        if d <= max_km:
            routes.update(stop_routes.get(sid, []))
    return list(routes)


# ── Main gravity computation ───────────────────────────────────────────────

def compute_gravity_ridership(gtfs_bytes: bytes, n_samples: int = N_OD_SAMPLES):
    """
    Run the gravity-model O-D sampling and return:
      route_ridership : {route_id: window_peak_ridership}   (float, normalised)
      od_samples      : list of dicts for map display
      zones           : list of zone dicts with lat/lon/pop
    """
    rng = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)

    # ── Load zones ────────────────────────────────────────────────────────
    if not GEOJSON_PATH.exists():
        return {}, [], []

    with open(GEOJSON_PATH) as f:
        geojson = json.load(f)

    zones = []
    geoms = []
    for feat in geojson["features"]:
        props = feat["properties"]
        lat, lon = _geom_centroid(feat["geometry"])
        if lat is None:
            continue
        pop = int(props.get("pop", 0) or 0)
        zones.append({
            "id":  props.get("id", str(len(zones))),
            "lat": lat,
            "lon": lon,
            "pop": pop,
        })
        geoms.append(feat["geometry"])

    if not zones:
        return {}, [], zones

    pops   = np.array([z["pop"] for z in zones], dtype=float)
    pops   = np.maximum(pops, 1)   # avoid zero weights
    prob   = pops / pops.sum()

    # ── Load GTFS stops + routes ──────────────────────────────────────────
    stops_arr, stop_routes = _load_stops_and_routes(gtfs_bytes)

    # ── Sample O-D pairs ──────────────────────────────────────────────────
    route_gravity_sum: dict[str, float] = {}
    od_records = []

    for _ in range(n_samples):
        # Sample origin ∝ population
        o_idx = int(np_rng.choice(len(zones), p=prob))
        # Sample destination ∝ population, different zone
        d_prob = prob.copy(); d_prob[o_idx] = 0; d_prob /= d_prob.sum()
        d_idx  = int(np_rng.choice(len(zones), p=d_prob))

        oz, dz = zones[o_idx], zones[d_idx]

        # Random interior point
        o_lat, o_lon = _sample_point_in_geom(geoms[o_idx])
        d_lat, d_lon = _sample_point_in_geom(geoms[d_idx])

        dist_km = haversine_km(o_lat, o_lon, d_lat, d_lon)
        grav    = (oz["pop"] * dz["pop"]) / max(dist_km, 0.5) ** 2

        # Routes near origin and destination
        o_routes = _nearest_routes(o_lat, o_lon, stops_arr, stop_routes)
        d_routes = _nearest_routes(d_lat, d_lon, stops_arr, stop_routes)
        candidate_routes = list(set(o_routes) | set(d_routes))

        for r in candidate_routes:
            route_gravity_sum[r] = route_gravity_sum.get(r, 0.0) + grav

        if len(od_records) < 500:   # keep display sample small
            od_records.append({
                "o_lat": round(o_lat, 5), "o_lon": round(o_lon, 5),
                "d_lat": round(d_lat, 5), "d_lon": round(d_lon, 5),
                "o_pop": oz["pop"], "d_pop": dz["pop"],
                "dist_km": round(dist_km, 2),
                "gravity": round(grav, 1),
                "routes": candidate_routes[:8],
            })

    # ── Normalise to peak-hour ridership ──────────────────────────────────
    max_grav = max(route_gravity_sum.values()) if route_gravity_sum else 1.0
    route_ridership = {
        r: round((g / max_grav) * SCALE_PEAK_DAILY, 1)
        for r, g in route_gravity_sum.items()
    }

    return route_ridership, od_records, zones


# ── Public helpers called by app.py ───────────────────────────────────────

def get_window_ridership(route_id: str, window_start_h: int, window_end_h: int,
                          route_ridership: dict, fallback: float = 300.0) -> float:
    """
    Estimate ridership for a route during [window_start_h, window_end_h).
    Uses hourly normalisation applied to the gravity-model peak estimate.
    """
    peak = route_ridership.get(route_id)
    if peak is None:
        return fallback
    total = sum(peak * HOURLY_NORM.get(h, 0.5) for h in range(window_start_h, window_end_h))
    return round(total, 1)


def ridership_for_all_routes(route_ids: list, window_start_h: int, window_end_h: int,
                              route_ridership: dict, fallback: float = 300.0) -> dict:
    return {
        r: get_window_ridership(r, window_start_h, window_end_h, route_ridership, fallback)
        for r in route_ids
    }


def hourly_profile_chart_data() -> pd.DataFrame:
    return HOURLY_DF.copy()
