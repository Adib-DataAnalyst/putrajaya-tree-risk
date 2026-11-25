"""
Smart Tree Management — Wind Risk Dashboard (Now + 6-day forecast)
--------------------------------------------------------------------
Static tree data (company platform) + live/forecast wind (Open-Meteo, free).
Dark control-room layout: map on the left, tree detail panel on the right,
lead-time slider along the bottom.

Run with:  streamlit run app.py
"""

import time
from datetime import datetime

import altair as alt
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from fpdf import FPDF
from pyproj import Transformer
from streamlit_folium import st_folium

# ===========================================================
# CONFIG
# ===========================================================
FILE_PATH = "Putrajaya_Tree.csv"
GRID_PRECISION = 2             # ~1.1 km grid cells
YELLOW_THRESHOLD_RATIO = 0.8   # 80-99% of critical speed = AT RISK
FORECAST_DAYS_AHEAD = 6        # Now + 6 days
MAX_MARKERS_ON_MAP = 4000

STATUS_COLORS = {"SAFE": "#1fb85a", "AT RISK": "#f2c200", "DANGER": "#e5393a"}
STATUS_PRIORITY = {"DANGER": 0, "AT RISK": 1, "SAFE": 2}
ACCENT = "#22b8e0"

st.set_page_config(page_title="Putrajaya Tree Wind Risk", layout="wide", page_icon="🌳")

# ===========================================================
# DARK PROFESSIONAL THEME
# ===========================================================
st.markdown(
    f"""
    <style>
    .stApp {{ background: #0b1220; }}
    section[data-testid="stSidebar"] {{ background: #0d1526; border-right: 1px solid #1f2d45; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 1rem; max-width: 100%; }}
    h1, h2, h3, h4 {{ color: #e8f0fb !important; font-family: "Segoe UI", sans-serif; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: #111a2b; border: 1px solid #1f2d45; border-radius: 10px;
    }}
    div[data-testid="stMetric"] {{
        background: #111a2b; border: 1px solid #1f2d45; border-radius: 10px; padding: 8px 12px;
    }}
    div[data-testid="stMetricLabel"] p {{ color: #8fa3bf !important; font-size: 0.78rem; }}
    div[data-testid="stMetricValue"] {{ color: #e8f0fb; }}
    .stButton > button, .stDownloadButton > button {{
        background: #111a2b; color: #dbe5f3; border: 1px solid #2a3d5e; border-radius: 8px;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: {ACCENT}; color: {ACCENT};
    }}
    .hdr-title {{ font-size: 1.25rem; font-weight: 700; color: #e8f0fb; margin: 0; }}
    .hdr-sub {{ font-family: "Consolas", monospace; font-size: 0.8rem; color: #8fa3bf; margin: 2px 0 0 0; }}
    .lbl {{ font-size: 0.7rem; letter-spacing: 0.08em; color: #8fa3bf; text-transform: uppercase; margin-bottom: 4px; }}
    .stat-big {{ font-size: 1.5rem; font-weight: 700; color: #e8f0fb; line-height: 1.1; }}
    .stat-sub {{ font-size: 0.72rem; color: #8fa3bf; }}
    .badge {{ display:inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; color:#0b1220; }}
    .warn-box {{ border: 1px solid #d98a1c; border-radius: 8px; padding: 10px 12px; background: #1a1409; color: #f3dfb8; }}
    .ok-box {{ border: 1px solid #1f6b43; border-radius: 8px; padding: 10px 12px; background: #0c1a14; color: #b8f3d1; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================
# STEP 1 — Static data: load, clean, transform coordinates, grid-assign
# ===========================================================
@st.cache_data
def load_static_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["x", "y", "Critical Wind Speed [kmh]"]).copy()
    df["Critical Wind Speed [kmh]"] = pd.to_numeric(
        df["Critical Wind Speed [kmh]"], errors="coerce"
    )
    df = df.dropna(subset=["Critical Wind Speed [kmh]"]).copy()

    transformer = Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(df["x"].values, df["y"].values)
    df["Latitude"] = lat
    df["Longitude"] = lon
    df["grid_lat"] = df["Latitude"].round(GRID_PRECISION)
    df["grid_lon"] = df["Longitude"].round(GRID_PRECISION)
    return df


try:
    base_df = load_static_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"Couldn't find '{FILE_PATH}'. Put it next to app.py.")
    st.stop()

# ===========================================================
# STEP 2 — Live + 6-day forecast per grid zone (one API call each, cached 1h)
#    daily index 0 = today, 1..6 = next six days
# ===========================================================
@st.cache_data(ttl=3600, show_spinner="Fetching live + 6-day forecast wind data...")
def fetch_weather_for_grids(grid_coords: list[tuple[float, float]]) -> dict:
    results = {}
    total_days = FORECAST_DAYS_AHEAD + 1
    for lat, lon in grid_coords:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current_weather=true"
            "&hourly=windspeed_10m"
            "&daily=windspeed_10m_max,winddirection_10m_dominant"
            f"&forecast_days={total_days}"
            "&windspeed_unit=kmh&timezone=auto"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results[(lat, lon)] = {
                "current": data["current_weather"]["windspeed"],
                "direction": data["current_weather"]["winddirection"],
                "hourly_times": data["hourly"]["time"][:24],
                "hourly_speeds": data["hourly"]["windspeed_10m"][:24],
                "daily_times": data["daily"]["time"],
                "daily_max": data["daily"]["windspeed_10m_max"],
                "daily_dir": data["daily"]["winddirection_10m_dominant"],
            }
        except Exception:
            results[(lat, lon)] = None
        time.sleep(0.5)
    return results


unique_grids = list(
    base_df[["grid_lat", "grid_lon"]].drop_duplicates().itertuples(index=False, name=None)
)
weather_lookup = fetch_weather_for_grids(unique_grids)

sample_weather = next((w for w in weather_lookup.values() if w), None)
if sample_weather is None:
    st.error("Couldn't fetch weather for any grid zone. Check your internet connection.")
    st.stop()

# One row per grid zone: wind_0/dir_0 = live, wind_1..6/dir_1..6 = forecast days
grid_rows = []
for (glat, glon), w in weather_lookup.items():
    if not w:
        continue
    row = {"grid_lat": glat, "grid_lon": glon, "wind_0": w["current"], "dir_0": w["direction"]}
    for i in range(1, FORECAST_DAYS_AHEAD + 1):
        row[f"wind_{i}"] = w["daily_max"][i]
        row[f"dir_{i}"] = w["daily_dir"][i]
    grid_rows.append(row)
grid_df = pd.DataFrame(grid_rows)

# ===========================================================
# LAYOUT SKELETON — header on top, main in the middle, slider at the bottom.
# The slider is filled first so the other sections know the selected day.
# ===========================================================
header_box = st.container()
main_box = st.container()
time_box = st.container()

daily_dates = sample_weather["daily_times"]
day_labels = ["Now"] + [pd.to_datetime(d).strftime("%a %d %b") for d in daily_dates[1:]]

with time_box:
    st.markdown('<div class="lbl">Lead time</div>', unsafe_allow_html=True)
    selected_label = st.select_slider(
        "Lead time", options=day_labels, value=day_labels[0], label_visibility="collapsed"
    )
selected_idx = day_labels.index(selected_label)
valid_text = (
    f"Live · {datetime.now().strftime('%a %d %b %H:%M')}"
    if selected_idx == 0
    else f"Valid {selected_label} · lead +{selected_idx} d · daily max wind"
)

# ===========================================================
# STEP 3 — Attach selected-day wind + 3-tier status to every tree (vectorised)
# ===========================================================
tree_df = base_df.merge(grid_df, on=["grid_lat", "grid_lon"], how="inner")
tree_df["Current Wind Speed (kmh)"] = tree_df["wind_0"]
tree_df["Display Wind Speed (kmh)"] = tree_df[f"wind_{selected_idx}"]
tree_df = tree_df.dropna(subset=["Display Wind Speed (kmh)"]).copy()

crit = tree_df["Critical Wind Speed [kmh]"]
wind = tree_df["Display Wind Speed (kmh)"]
tree_df["Status"] = np.where(
    wind >= crit, "DANGER", np.where(wind >= YELLOW_THRESHOLD_RATIO * crit, "AT RISK", "SAFE")
)

# ===========================================================
# STEP 4 — Sidebar: search, filters, refresh
# ===========================================================
st.sidebar.header("Search & Filter")
search_id = st.sidebar.text_input("Search by Tree ID")
selected_precincts = st.sidebar.multiselect(
    "Filter by Precinct", sorted(tree_df["Precint"].dropna().unique())
)
status_filter = st.sidebar.multiselect(
    "Filter by Status", ["SAFE", "AT RISK", "DANGER"], default=["SAFE", "AT RISK", "DANGER"]
)
show_wind_arrows = st.sidebar.checkbox("Show wind direction arrows", value=True)
if st.sidebar.button("🔄 Refresh live + forecast data"):
    fetch_weather_for_grids.clear()
    st.rerun()
st.sidebar.caption(
    f"{len(base_df):,} trees → {len(unique_grids)} grid zones for weather queries."
)

filtered_df = tree_df.copy()
if search_id:
    filtered_df = filtered_df[
        filtered_df["Tree ID"].astype(str).str.contains(search_id, case=False)
    ]
if selected_precincts:
    filtered_df = filtered_df[filtered_df["Precint"].isin(selected_precincts)]
filtered_df = filtered_df[filtered_df["Status"].isin(status_filter)]

# ===========================================================
# STEP 5 — Header + basemap toggle
# ===========================================================
with header_box:
    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown(
            f'<p class="hdr-title">🌳 Smart Tree Management · Wind Risk · Putrajaya</p>'
            f'<p class="hdr-sub">{valid_text}</p>',
            unsafe_allow_html=True,
        )
    with h2:
        basemap = st.radio("Basemap", ["Dark", "Satellite"], horizontal=True, index=1,
                           label_visibility="collapsed")

# ===========================================================
# STEP 6 — Main area: map (left) + tree detail panel (right)
# ===========================================================
with main_box:
    left, right = st.columns([2.6, 1])

    # ---------------- LEFT: KPIs + map ----------------
    with left:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Trees", f"{len(filtered_df):,}")
        k2.metric("🟢 Safe", f"{int((filtered_df['Status'] == 'SAFE').sum()):,}")
        k3.metric("🟡 At Risk", f"{int((filtered_df['Status'] == 'AT RISK').sum()):,}")
        k4.metric("🔴 Danger", f"{int((filtered_df['Status'] == 'DANGER').sum()):,}")

        map_render_df = filtered_df.copy()
        if len(map_render_df) > MAX_MARKERS_ON_MAP:
            map_render_df["_priority"] = map_render_df["Status"].map(STATUS_PRIORITY)
            map_render_df = (
                map_render_df.sort_values("_priority")
                .groupby("_priority", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), MAX_MARKERS_ON_MAP // 3), random_state=1))
            )
            st.warning(
                f"Showing {len(map_render_df):,} of {len(filtered_df):,} trees "
                "(DANGER / AT RISK prioritised) to keep the map responsive. "
                "Pick a Precinct in the sidebar to see every tree in an area."
            )

        if map_render_df.empty:
            st.warning("No trees match the current filters.")
        else:
            m = folium.Map(
                location=[map_render_df["Latitude"].mean(), map_render_df["Longitude"].mean()],
                zoom_start=13,
                tiles=None,
            )
            if basemap == "Satellite":
                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri World Imagery", overlay=False, control=False,
                ).add_to(m)
            else:
                folium.TileLayer("CartoDB dark_matter", overlay=False, control=False).add_to(m)

            wind_label = "Live wind" if selected_idx == 0 else f"Forecast max wind ({selected_label})"
            for _, row in map_render_df.iterrows():
                popup_html = (
                    f"<b>Tree ID:</b> {row['Tree ID']}<br>"
                    f"<b>Status ({selected_label}):</b> {row['Status']}<br>"
                    f"<b>Precinct:</b> {row.get('Precint', 'N/A')}<br>"
                    f"<b>Street:</b> {row.get('Street Name And Number', 'N/A')}<br>"
                    f"<b>Critical wind speed (TSI threshold):</b> "
                    f"{row['Critical Wind Speed [kmh]']:.1f} km/h<br>"
                    f"<b>{wind_label}:</b> {row['Display Wind Speed (kmh)']:.1f} km/h<br>"
                    f"<b>Trunk lean angle:</b> {row.get('Trunk Lean Angle [°]', 'N/A')}°"
                )
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=4,
                    color=STATUS_COLORS[row["Status"]],
                    fill=True,
                    fill_color=STATUS_COLORS[row["Status"]],
                    fill_opacity=0.9,
                    weight=1,
                    popup=folium.Popup(popup_html, max_width=260),
                ).add_to(m)

            # Wind arrows: now work for live AND forecast days (daily dominant direction).
            # Meteorological direction = where wind comes FROM, so +180° points the way it blows.
            if show_wind_arrows:
                centroids = (
                    map_render_df.groupby(["grid_lat", "grid_lon"])[["Latitude", "Longitude"]]
                    .mean().reset_index()
                    .merge(grid_df[["grid_lat", "grid_lon", f"dir_{selected_idx}", f"wind_{selected_idx}"]],
                           on=["grid_lat", "grid_lon"])
                )
                for _, g in centroids.iterrows():
                    d_from = g[f"dir_{selected_idx}"]
                    if pd.isna(d_from):
                        continue
                    arrow = folium.DivIcon(
                        html=(
                            f'<div style="transform: rotate({d_from + 180}deg); font-size:24px; '
                            f'color:{ACCENT}; text-shadow:0 0 4px #000; text-align:center;">&#8593;</div>'
                        ),
                        icon_size=(24, 24), icon_anchor=(12, 12),
                    )
                    folium.Marker(
                        location=[g["Latitude"], g["Longitude"]],
                        icon=arrow,
                        tooltip=f"Wind from {d_from:.0f}° · {g[f'wind_{selected_idx}']:.1f} km/h",
                    ).add_to(m)

            st_folium(m, use_container_width=True, height=560, returned_objects=[])
            st.caption(
                "🟢 Safe   🟡 At Risk (80–99% of critical speed)   🔴 Danger (≥ critical speed)   "
                "· cyan arrows point the way the wind is blowing."
            )

    # ---------------- RIGHT: tree detail + warnings ----------------
    with right:
        with st.container(border=True):
            st.markdown('<div class="lbl">Selected tree</div>', unsafe_allow_html=True)
            ranked = filtered_df.assign(_p=filtered_df["Status"].map(STATUS_PRIORITY)).sort_values(
                ["_p", "Tree ID"]
            )
            options = ranked["Tree ID"].astype(str).head(300).tolist()
            if not options:
                st.info("No trees match the current filters.")
            else:
                picked_id = st.selectbox(
                    "Tree", options, label_visibility="collapsed",
                    help="Shows the 300 highest-risk matching trees. Use the sidebar search for a specific one.",
                )
                prow = filtered_df[filtered_df["Tree ID"].astype(str) == picked_id].iloc[0]
                w = weather_lookup.get((prow["grid_lat"], prow["grid_lon"]))
                critical = float(prow["Critical Wind Speed [kmh]"])
                color = STATUS_COLORS[prow["Status"]]

                st.markdown(
                    f'<span class="badge" style="background:{color}">{prow["Status"]}</span> '
                    f'<span class="stat-sub">&nbsp;{prow.get("Precint", "N/A")} · '
                    f'{prow.get("Street Name And Number", "N/A")}</span>',
                    unsafe_allow_html=True,
                )

                daily_max = [x for x in w["daily_max"]] if w else []
                peak = max(daily_max) if daily_max else float("nan")
                days_danger = int(sum(1 for x in daily_max if x is not None and x >= critical))

                s1, s2, s3 = st.columns(3)
                s1.markdown(f'<div class="stat-big">{critical:.0f}</div><div class="stat-sub">Critical km/h</div>',
                            unsafe_allow_html=True)
                s2.markdown(f'<div class="stat-big">{peak:.0f}</div><div class="stat-sub">7-day peak km/h</div>',
                            unsafe_allow_html=True)
                s3.markdown(f'<div class="stat-big">{days_danger}</div><div class="stat-sub">Days ≥ critical</div>',
                            unsafe_allow_html=True)
                lean = prow.get("Trunk Lean Angle [°]", "N/A")
                st.markdown(f'<div class="stat-sub">Trunk lean angle: {lean}°</div>', unsafe_allow_html=True)

                if w:
                    # --- Hourly (today) ---
                    st.markdown('<div class="lbl" style="margin-top:10px">Today · hourly wind</div>',
                                unsafe_allow_html=True)
                    hdf = pd.DataFrame({
                        "Time": pd.to_datetime(w["hourly_times"]),
                        "Wind": w["hourly_speeds"],
                    })
                    ymax = max(hdf["Wind"].max(), critical) * 1.15
                    area = alt.Chart(hdf).mark_area(
                        opacity=0.45, color=ACCENT, line={"color": ACCENT}
                    ).encode(
                        x=alt.X("Time:T", title=None),
                        y=alt.Y("Wind:Q", title="km/h", scale=alt.Scale(domain=[0, ymax])),
                    )
                    rule = alt.Chart(pd.DataFrame({"y": [critical]})).mark_rule(
                        color="#e5393a", strokeDash=[6, 4]
                    ).encode(y="y:Q")
                    st.altair_chart((area + rule).properties(height=150), use_container_width=True)

                    # --- Daily (today + 6 days), bars coloured by tree status that day ---
                    st.markdown('<div class="lbl">Today + 6 days · daily max wind</div>',
                                unsafe_allow_html=True)
                    ddf = pd.DataFrame({
                        "Day": ["Today"] + [pd.to_datetime(d).strftime("%a %d") for d in w["daily_times"][1:]],
                        "Wind": w["daily_max"],
                    })
                    ddf["Status"] = np.where(
                        ddf["Wind"] >= critical, "DANGER",
                        np.where(ddf["Wind"] >= YELLOW_THRESHOLD_RATIO * critical, "AT RISK", "SAFE"),
                    )
                    ymax2 = max(ddf["Wind"].max(), critical) * 1.15
                    bars = alt.Chart(ddf).mark_bar().encode(
                        x=alt.X("Day:N", sort=None, title=None),
                        y=alt.Y("Wind:Q", title="km/h", scale=alt.Scale(domain=[0, ymax2])),
                        color=alt.Color(
                            "Status:N",
                            scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values())),
                            legend=None,
                        ),
                    )
                    rule2 = alt.Chart(pd.DataFrame({"y": [critical]})).mark_rule(
                        color="#e5393a", strokeDash=[6, 4]
                    ).encode(y="y:Q")
                    st.altair_chart((bars + rule2).properties(height=150), use_container_width=True)
                else:
                    st.info("No forecast data for this tree's grid zone.")

        with st.container(border=True):
            st.markdown('<div class="lbl">Warnings in view</div>', unsafe_allow_html=True)
            danger_view = filtered_df[filtered_df["Status"] == "DANGER"]
            if danger_view.empty:
                st.markdown(
                    f'<div class="ok-box"><b>No trees in danger</b><br>{selected_label} · current filters</div>',
                    unsafe_allow_html=True,
                )
            else:
                by_precinct = (
                    danger_view.groupby(danger_view["Precint"].fillna("Unknown"))
                    .size().sort_values(ascending=False).head(5)
                )
                lines = "<br>".join(f"{p}: <b>{n:,}</b> trees" for p, n in by_precinct.items())
                st.markdown(
                    f'<div class="warn-box"><b>⚠ Trees above critical wind speed</b><br>'
                    f'{len(danger_view):,} trees · {selected_label}<br><br>{lines}</div>',
                    unsafe_allow_html=True,
                )

# ===========================================================
# STEP 7 — Danger table
# ===========================================================
st.subheader(f"Trees in Danger — {selected_label}")
danger_df = filtered_df[filtered_df["Status"] == "DANGER"]
if danger_df.empty:
    st.success("No trees in the danger zone for this day.")
else:
    cols = [c for c in ["Tree ID", "Precint", "Street Name And Number",
                         "Critical Wind Speed [kmh]", "Display Wind Speed (kmh)",
                         "Trunk Lean Angle [°]", "Status"] if c in danger_df.columns]
    st.dataframe(danger_df[cols], use_container_width=True)

# ===========================================================
# STEP 8 — PDF report + CSV download
# ===========================================================
def generate_pdf_report(df: pd.DataFrame, day_label: str, precincts_filter, status_filter) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 40, 70)
    pdf.cell(0, 12, "Putrajaya Tree Wind Risk Report", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 8, f"Conditions shown for: {day_label}", ln=True)
    if precincts_filter:
        pdf.cell(0, 8, f"Precinct filter: {', '.join(map(str, precincts_filter))}", ln=True)
    pdf.cell(0, 8, f"Status filter: {', '.join(status_filter)}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Total Trees: {len(df)}", ln=True)
    pdf.set_text_color(0, 130, 0)
    pdf.cell(0, 8, f"Safe: {int((df['Status'] == 'SAFE').sum())}", ln=True)
    pdf.set_text_color(180, 140, 0)
    pdf.cell(0, 8, f"At Risk: {int((df['Status'] == 'AT RISK').sum())}", ln=True)
    pdf.set_text_color(200, 30, 30)
    pdf.cell(0, 8, f"Danger: {int((df['Status'] == 'DANGER').sum())}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    danger_trees = df[df["Status"] == "DANGER"]
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Trees in Danger", ln=True)
    if danger_trees.empty:
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, "None - no trees are in the danger zone.", ln=True)
    else:
        show_cols = [c for c in ["Tree ID", "Precint", "Critical Wind Speed [kmh]",
                                  "Display Wind Speed (kmh)"] if c in danger_trees.columns]
        widths = [30, 30, 55, 55][: len(show_cols)]
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(225, 232, 242)
        for c, wd in zip(show_cols, widths):
            pdf.cell(wd, 8, str(c), border=1, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        max_rows = 60
        for _, r in danger_trees.head(max_rows).iterrows():
            for c, wd in zip(show_cols, widths):
                v = r[c]
                pdf.cell(wd, 7, f"{v:.1f}" if isinstance(v, float) else str(v), border=1)
            pdf.ln()
        if len(danger_trees) > max_rows:
            pdf.ln(2)
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 8, f"...and {len(danger_trees) - max_rows} more. See the CSV for the full list.", ln=True)
    return bytes(pdf.output())


st.subheader("Download Report")
pdf_bytes = generate_pdf_report(filtered_df, selected_label, selected_precincts, status_filter)
d1, d2, _ = st.columns([1, 1, 3])
with d1:
    st.download_button(
        "📄 PDF Report", data=pdf_bytes,
        file_name=f"Tree_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
    )
with d2:
    st.download_button(
        "💾 Data as CSV", data=tree_df.to_csv(index=False).encode("utf-8"),
        file_name="Putrajaya_Tree_Status.csv", mime="text/csv",
    )
