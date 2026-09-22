"""
Smart Tree Management — Real-Time & Forecasted Wind Risk Dashboard
--------------------------------------------------------------------
Bridges static tree data from the company platform with live + forecast
wind data from the free Open-Meteo API. The same map can be repainted
to show today's live status or any of the next 3 forecast days.

Run with:  streamlit run app.py
"""

import time

import folium
import pandas as pd
import requests
import streamlit as st
from pyproj import Transformer
from streamlit_folium import st_folium

# ===========================================================
# CONFIG
# ===========================================================
FILE_PATH = "Putrajaya_Tree.csv"
GRID_PRECISION = 2            # rounding decimals -> ~1.1km grid cells
YELLOW_THRESHOLD_RATIO = 0.8  # 80% of critical speed = "at risk"
FORECAST_DAYS_AHEAD = 3       # Mon, Tue, Wed style — days AFTER today
MAX_MARKERS_ON_MAP = 4000     # browser performance safeguard

STATUS_COLORS = {"SAFE": "#00aa00", "AT RISK": "#f0c800", "DANGER": "#dc1e1e"}
STATUS_PRIORITY = {"DANGER": 0, "AT RISK": 1, "SAFE": 2}

st.set_page_config(page_title="Putrajaya Tree Wind Risk", layout="wide")
st.title("🌳 Smart Tree Management — Wind Risk Dashboard")
st.caption(
    "Static tree data (Greehill) + live & forecast wind data "
    "(Open-Meteo, free, no API key)."
)

# ===========================================================
# STEP 1 — Load & clean static data, transform coordinates, grid-assign
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
    tree_df = load_static_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"Couldn't find '{FILE_PATH}'. Put it next to app.py.")
    st.stop()

n_grids = tree_df[["grid_lat", "grid_lon"]].drop_duplicates().shape[0]
st.caption(f"{len(tree_df):,} trees → {n_grids} grid zones for weather queries.")

# ===========================================================
# STEP 2 — Fetch live + forecast wind (+ direction) per grid, cached 1h
#    daily_max[0] = today, daily_max[1..3] = next 3 days
# ===========================================================
@st.cache_data(ttl=3600, show_spinner="Fetching live + forecast wind data...")
def fetch_weather_for_grids(grid_coords: list[tuple[float, float]]) -> dict:
    results = {}
    total_days = FORECAST_DAYS_AHEAD + 1  # today + N days ahead
    for lat, lon in grid_coords:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current_weather=true"
            "&hourly=windspeed_10m"
            "&daily=windspeed_10m_max"
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
                "daily_times": data["daily"]["time"],       # [today, day+1, day+2, day+3]
                "daily_max": data["daily"]["windspeed_10m_max"],
            }
        except Exception:
            results[(lat, lon)] = None
        time.sleep(0.5)
    return results


col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("🔄 Refresh live + forecast data"):
        fetch_weather_for_grids.clear()

unique_grids = list(
    tree_df[["grid_lat", "grid_lon"]]
    .drop_duplicates()
    .itertuples(index=False, name=None)
)
weather_lookup = fetch_weather_for_grids(unique_grids)

tree_df["Current Wind Speed (kmh)"] = tree_df.apply(
    lambda r: (weather_lookup.get((r["grid_lat"], r["grid_lon"])) or {}).get("current"),
    axis=1,
)
tree_df = tree_df.dropna(subset=["Current Wind Speed (kmh)"]).copy()

with col_b:
    st.metric(
        "Average Live Wind Speed — Putrajaya",
        f"{tree_df['Current Wind Speed (kmh)'].mean():.1f} km/h",
    )

# ===========================================================
# STEP 3 — Day selector (Now / Day+1 / Day+2 / Day+3)
#    Uses real calendar dates pulled from the forecast response itself
# ===========================================================
sample_weather = next((w for w in weather_lookup.values() if w), None)
if sample_weather is None:
    st.error("Couldn't fetch weather for any grid zone. Check your internet connection.")
    st.stop()

daily_dates = sample_weather["daily_times"]  # e.g. ['2026-09-06', '2026-09-07', ...]
day_labels = ["🔴 Now (Live)"] + [
    pd.to_datetime(d).strftime("%a %d %b") for d in daily_dates[1:]
]

st.subheader("📅 Select Day to View on Map")
selected_label = st.radio(
    "Map shows tree condition for:", day_labels, horizontal=True, label_visibility="collapsed"
)
selected_day_index = day_labels.index(selected_label)  # 0 = live, 1..3 = forecast day

# ===========================================================
# STEP 4 — Compute wind speed + status for the SELECTED day
# ===========================================================
def get_display_wind(row):
    weather = weather_lookup.get((row["grid_lat"], row["grid_lon"]))
    if not weather:
        return None
    if selected_day_index == 0:
        return weather["current"]
    return weather["daily_max"][selected_day_index]


tree_df["Display Wind Speed (kmh)"] = tree_df.apply(get_display_wind, axis=1)
tree_df = tree_df.dropna(subset=["Display Wind Speed (kmh)"]).copy()


def classify(row):
    wind = row["Display Wind Speed (kmh)"]
    critical = row["Critical Wind Speed [kmh]"]
    if wind >= critical:
        return "DANGER"
    elif wind >= YELLOW_THRESHOLD_RATIO * critical:
        return "AT RISK"
    return "SAFE"


tree_df["Status"] = tree_df.apply(classify, axis=1)

# ===========================================================
# STEP 5 — Sidebar: search & filters
# ===========================================================
st.sidebar.header("🔍 Search & Filter")
search_id = st.sidebar.text_input("Search by Tree ID")
selected_precincts = st.sidebar.multiselect(
    "Filter by Precinct", sorted(tree_df["Precint"].dropna().unique())
)
status_filter = st.sidebar.multiselect(
    "Filter by Status", ["SAFE", "AT RISK", "DANGER"],
    default=["SAFE", "AT RISK", "DANGER"],
)
show_wind_arrows = st.sidebar.checkbox("Show live wind direction arrows", value=True)

filtered_df = tree_df.copy()
if search_id:
    filtered_df = filtered_df[
        filtered_df["Tree ID"].astype(str).str.contains(search_id, case=False)
    ]
if selected_precincts:
    filtered_df = filtered_df[filtered_df["Precint"].isin(selected_precincts)]
filtered_df = filtered_df[filtered_df["Status"].isin(status_filter)]

# ===========================================================
# STEP 6 — Summary metrics (reflect the selected day)
# ===========================================================
st.caption(f"Showing conditions for: **{selected_label}**")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Trees", len(filtered_df))
m2.metric("🟢 Safe", int((filtered_df["Status"] == "SAFE").sum()))
m3.metric("🟡 At Risk", int((filtered_df["Status"] == "AT RISK").sum()))
m4.metric("🔴 Danger", int((filtered_df["Status"] == "DANGER").sum()))

# ===========================================================
# STEP 7 — Satellite map (same map, repainted per selected day)
# ===========================================================
st.subheader(f"Tree Risk Map — {selected_label}")

map_render_df = filtered_df.copy()
truncated = False
if len(map_render_df) > MAX_MARKERS_ON_MAP:
    truncated = True
    map_render_df["_priority"] = map_render_df["Status"].map(STATUS_PRIORITY)
    map_render_df = (
        map_render_df.sort_values("_priority")
        .groupby("_priority", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), MAX_MARKERS_ON_MAP // 3), random_state=1))
    )

if truncated:
    st.warning(
        f"Showing {len(map_render_df):,} of {len(filtered_df):,} trees "
        "(all DANGER/AT RISK trees prioritized) to keep the map responsive. "
        "Narrow the Precinct filter to see every tree in an area."
    )

if map_render_df.empty:
    st.warning("No trees match the current filters.")
else:
    center_lat = map_render_df["Latitude"].mean()
    center_lon = map_render_df["Longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=False,
    ).add_to(m)

    wind_field_label = "Live Wind Speed" if selected_day_index == 0 else f"Forecast Max Wind ({selected_label})"

    for _, row in map_render_df.iterrows():
        popup_html = (
            f"<b>Tree ID:</b> {row['Tree ID']}<br>"
            f"<b>Status ({selected_label}):</b> {row['Status']}<br>"
            f"<b>Precinct:</b> {row.get('Precint', 'N/A')}<br>"
            f"<b>Street:</b> {row.get('Street Name And Number', 'N/A')}<br>"
            f"<b>Critical Wind Speed (TSI threshold):</b> "
            f"{row['Critical Wind Speed [kmh]']:.1f} km/h<br>"
            f"<b>{wind_field_label}:</b> {row['Display Wind Speed (kmh)']:.1f} km/h<br>"
            f"<b>Trunk Lean Angle:</b> {row.get('Trunk Lean Angle [°]', 'N/A')}°"
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

    # Wind direction arrows only make sense for "live" (forecast doesn't give direction here)
    if show_wind_arrows and selected_day_index == 0:
        grid_centroids = (
            map_render_df.groupby(["grid_lat", "grid_lon"])[["Latitude", "Longitude"]]
            .mean()
            .reset_index()
        )
        for _, g in grid_centroids.iterrows():
            weather = weather_lookup.get((g["grid_lat"], g["grid_lon"]))
            if not weather:
                continue
            direction_deg = weather["direction"]
            arrow_icon = folium.DivIcon(
                html=(
                    f'<div style="transform: rotate({direction_deg}deg); '
                    f'font-size: 22px; color: white; text-shadow: 0 0 3px black;">'
                    f'&#8593;</div>'
                )
            )
            folium.Marker(
                location=[g["Latitude"], g["Longitude"]],
                icon=arrow_icon,
                tooltip=f"Wind from {direction_deg}° — {weather['current']} km/h",
            ).add_to(m)

    st_folium(m, use_container_width=True, height=600, returned_objects=[])

    caption = "🟢 Safe   🟡 At Risk (80–99% of critical speed)   🔴 Danger (≥ critical speed)"
    if show_wind_arrows and selected_day_index == 0:
        caption += "   ↑ arrows show live wind direction per grid zone."
    elif show_wind_arrows and selected_day_index != 0:
        caption += "   (Wind direction arrows are only shown for Live — forecast direction isn't fetched.)"
    st.caption(caption)

# ===========================================================
# STEP 8 — Forecast graphs for a selected tree (unchanged)
# ===========================================================
st.subheader("📈 Wind Forecast Detail for a Specific Tree")
tree_options = filtered_df["Tree ID"].astype(str).tolist()
if tree_options:
    picked_id = st.selectbox("Select a Tree ID", tree_options)
    picked_row = filtered_df[filtered_df["Tree ID"].astype(str) == picked_id].iloc[0]
    weather = weather_lookup.get((picked_row["grid_lat"], picked_row["grid_lon"]))
    critical = picked_row["Critical Wind Speed [kmh]"]

    if weather:
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**Next 24 Hours**")
            hourly_df = pd.DataFrame(
                {
                    "Time": pd.to_datetime(weather["hourly_times"]),
                    "Forecast Wind (kmh)": weather["hourly_speeds"],
                }
            ).set_index("Time")
            st.line_chart(hourly_df)

        with fc2:
            st.markdown(f"**Today + Next {FORECAST_DAYS_AHEAD} Days (daily max)**")
            daily_df = pd.DataFrame(
                {
                    "Date": [pd.to_datetime(d).strftime("%a %d %b") for d in weather["daily_times"]],
                    "Max Wind (kmh)": weather["daily_max"],
                }
            ).set_index("Date")
            st.bar_chart(daily_df)
            st.caption(f"Critical threshold for this tree: {critical} km/h")
    else:
        st.info("No forecast data available for this tree's grid zone.")
else:
    st.info("No trees match the current filters to forecast.")

# ===========================================================
# STEP 9 — Danger table + download (reflects selected day)
# ===========================================================
st.subheader(f"🔴 Trees in Danger — {selected_label}")
danger_df = filtered_df[filtered_df["Status"] == "DANGER"]
if danger_df.empty:
    st.success("No trees currently in the danger zone for this day.")
else:
    cols = [c for c in ["Tree ID", "Precint", "Street Name And Number",
                         "Critical Wind Speed [kmh]", "Display Wind Speed (kmh)",
                         "Trunk Lean Angle [°]", "Status"] if c in danger_df.columns]
    st.dataframe(danger_df[cols], use_container_width=True)

st.download_button(
    "💾 Download current view as CSV",
    data=tree_df.to_csv(index=False).encode("utf-8"),
    file_name="Putrajaya_Tree_Live_Status.csv",
    mime="text/csv",
)
