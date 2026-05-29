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
    # Added explicit unique key to bypass the Streamlit Duplicate Element ID error
    st.download_button("⬇ Download ranking CSV",
        data=ranked.to_csv(index=True).encode(),
        file_name="yrt_route_ranking.csv", mime="text/csv",
        key="footer_download_button")

st.caption("YRT Service Allocation Tool · Proof-of-concept · Not a replacement for operational service planning.")
