# 🚌 GTFS Service Allocation Dashboard

A prototype decision-support tool that estimates **where adding buses saves the most passenger time** in a suburban transit network.

Built around the methodology in the YRT service allocation paper: it combines GTFS schedule data with synthetic ridership assumptions to rank candidate routes by **passenger-hour savings**.

---

## Live Demo

Deploy instantly on **Streamlit Community Cloud** (free):

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your fork
3. Set `app.py` as the main file
4. Click Deploy

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload any GTFS `.zip` file (e.g. [YRT open data](https://www.yrt.ca/en/about-us/developer-resources.aspx)).

---

## What It Does

| Step | What the tool computes |
|------|----------------------|
| **1. Load GTFS** | Parses routes, trips, stop_times, calendar |
| **2. Filter by date & window** | Finds active service IDs; filters trips to your time window (e.g. 7–9 AM) |
| **3. Baseline headways** | Average gap between consecutive departures per route/direction |
| **4. Cycle time** | Estimated from GTFS trip duration × 2 × 1.15 recovery factor |
| **5. Added-service scenario** | Simulates adding N buses; computes new headway via `C_r / (N_r + Δ)` |
| **6. Passenger-hour savings** | `(ΔWait × Ridership) / 60`, ranked descending |

### Key Formula

```
Expected wait     W_r = H_r / 2          (half-headway assumption)
New headway       H_new = C_r / (N_r + Δ)
Pax-hr savings    PHS_r = (W_r - W_new) × R_r / 60
```

---

## Inputs

| Input | Source |
|-------|--------|
| GTFS `.zip` | Any transit agency (YRT, TTC, etc.) |
| Analysis date | Picker — determines active service IDs |
| Time window | e.g. 07:00–09:00 AM peak |
| Buses to add | Slider (1–5) |
| Ridership | **Synthetic / user-defined** — not actual agency data |

> ⚠️ Route-level ridership is not included in public GTFS feeds. The tool accepts user-entered assumed values for demonstration purposes only.

---

## Outputs

- **Route ranking table** — sorted by passenger-hour savings
- **Headway distribution** histogram
- **Bubble chart** — headway vs. ridership, sized by savings
- **Sensitivity chart** — incremental benefit of adding 1–5 buses to the top route
- **CSV export** of full ranking

---

## Limitations

- Uses **scheduled** GTFS data, not real operations (no delays, bunching, cancellations)
- Half-headway assumption may overestimate wait savings on low-frequency routes where passengers use schedules
- Cycle time estimated with a fixed 15% recovery allowance — actual layovers may differ
- Ridership is **synthetic** — results are illustrative, not operational recommendations

---

## Deployment

### Streamlit Community Cloud (recommended)
No server required. Free for public repos. See [docs.streamlit.io/deploy](https://docs.streamlit.io/deploy/streamlit-community-cloud).

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Data Sources

- GTFS specification: [gtfs.org](https://gtfs.org)
- YRT open data: [yrt.ca/developer-resources](https://www.yrt.ca/en/about-us/developer-resources.aspx)

---

*Proof-of-concept tool. Not a replacement for operational transit service planning.*
