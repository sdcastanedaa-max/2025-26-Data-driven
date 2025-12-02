import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Optional PNG handling (pip install pillow)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# -------------------------------------------------------------------
# CONFIG: where you'll later put your PNG icons
# -------------------------------------------------------------------
ICON_IMAGE_PATHS = {
    "substation": None,  # "assets/substation.png"
    "wind": None,        # "assets/wind.png"
    "pv": None,          # "assets/pv.png"
    "fossil": None,      # "assets/fossil.png"
    "bess": None,        # "assets/bess.png"
    "load_res": None,    # "assets/load_res.png"
    "load_ind": None,    # "assets/load_ind.png"
}


# ==========================================================
# Helpers – dummy forecast data
# ==========================================================

def generate_dummy_forecast_data(n_hours: int = 72) -> pd.DataFrame:
    """Create synthetic PV & wind forecast vs actual for Spain."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    hours = pd.date_range(now, now + timedelta(hours=n_hours - 1), freq="h")

    t = np.arange(n_hours)

    # PV: bell-ish shape over day
    pv_pattern = np.maximum(0, np.sin((t % 24 - 6) / 24 * np.pi * 2))
    pv_actual = 15_000 * pv_pattern + np.random.normal(0, 500, size=n_hours)  # MW

    # Wind: slower, wiggly
    wind_pattern = 0.6 + 0.2 * np.sin(t / 24 * 2 * np.pi) + 0.1 * np.sin(t / 6 * 2 * np.pi)
    wind_actual = (
        20_000 * np.clip(wind_pattern, 0.1, None)
        + np.random.normal(0, 800, size=n_hours)
    )

    # Forecasts = actual + noise
    pv_forecast = pv_actual + np.random.normal(0, 600, size=n_hours)
    wind_forecast = wind_actual + np.random.normal(0, 900, size=n_hours)

    df = pd.DataFrame(
        {
            "time": hours,
            "pv_actual": pv_actual,
            "pv_forecast": pv_forecast,
            "wind_actual": wind_actual,
            "wind_forecast": wind_forecast,
        }
    )
    return df


def compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    bias = np.mean(y_pred - y_true)

    non_zero = y_true != 0
    if np.any(non_zero):
        mape = (
            np.mean(
                np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
            )
            * 100
        )
    else:
        mape = np.nan

    return {
        "MAE (MW)": mae,
        "RMSE (MW)": rmse,
        "Bias (MW)": bias,
        "MAPE (%)": mape,
    }


# ==========================================================
# Grid helpers – topology (more realistic)
# ==========================================================

def get_demo_grid_nodes():
    """
    4 substations in a rough N-W-E-S cross,
    with assets sensibly attached.
    """
    nodes = [
        # Substations (4)
        {"id": "S1", "name": "Substation North",  "type": "substation", "x": 0,  "y": 8,  "capacity_MW": 6000},
        {"id": "S2", "name": "Substation West",   "type": "substation", "x": -8, "y": 0,  "capacity_MW": 5000},
        {"id": "S3", "name": "Substation East",   "type": "substation", "x": 8,  "y": 0,  "capacity_MW": 5000},
        {"id": "S4", "name": "Substation South",  "type": "substation", "x": 0,  "y": -8, "capacity_MW": 6000},

        # Wind farms (clustered nearer north & west / east)
        {"id": "W1", "name": "Wind Farm NW", "type": "wind", "x": -10, "y": 9,  "capacity_MW": 300},
        {"id": "W2", "name": "Wind Farm N1", "type": "wind", "x": -3,  "y": 11, "capacity_MW": 250},
        {"id": "W3", "name": "Wind Farm N2", "type": "wind", "x": 3,   "y": 11, "capacity_MW": 350},
        {"id": "W4", "name": "Wind Farm W",  "type": "wind", "x": -12, "y": 4,  "capacity_MW": 200},
        {"id": "W5", "name": "Wind Farm E",  "type": "wind", "x": 12,  "y": 4,  "capacity_MW": 280},

        # PV farms (more to the south / central)
        {"id": "P1", "name": "PV Farm NW", "type": "pv", "x": -6, "y": 2,  "capacity_MW": 80},
        {"id": "P2", "name": "PV Farm N",  "type": "pv", "x": 1,  "y": 5,  "capacity_MW": 120},
        {"id": "P3", "name": "PV Farm NE", "type": "pv", "x": 6,  "y": 2,  "capacity_MW": 90},
        {"id": "P4", "name": "PV Farm W",  "type": "pv", "x": -10, "y": -2, "capacity_MW": 100},
        {"id": "P5", "name": "PV Farm E",  "type": "pv", "x": 10, "y": -2, "capacity_MW": 110},
        {"id": "P6", "name": "PV Farm SW", "type": "pv", "x": -5, "y": -5, "capacity_MW": 130},
        {"id": "P7", "name": "PV Farm S",  "type": "pv", "x": 0,  "y": -4, "capacity_MW": 140},
        {"id": "P8", "name": "PV Farm SE", "type": "pv", "x": 5,  "y": -6, "capacity_MW": 95},

        # Fossil plants (feeding south & east)
        {"id": "F1", "name": "CCGT West", "type": "fossil", "x": -6, "y": -10, "capacity_MW": 600},
        {"id": "F2", "name": "CCGT East", "type": "fossil", "x": 6,  "y": -10, "capacity_MW": 600},

        # BESS (buffer between renewables & loads)
        {"id": "B1", "name": "BESS North", "type": "bess", "x": 4,  "y": 7,  "capacity_MW": 150},
        {"id": "B2", "name": "BESS South", "type": "bess", "x": -4, "y": -7, "capacity_MW": 200},

        # Loads (city / industrial areas)
        {"id": "L1", "name": "City Load West",  "type": "load_res", "x": -4, "y": 1,  "capacity_MW": 500},
        {"id": "L2", "name": "City Load East",  "type": "load_res", "x": 4,  "y": 1,  "capacity_MW": 550},
        {"id": "L3", "name": "Industrial Hub",  "type": "load_ind", "x": 2,  "y": -3, "capacity_MW": 800},
    ]
    return nodes


def get_demo_grid_lines():
    """
    Lines connect substations in a cross, with some assets fed by two lines
    (more realistic meshing).
    """
    lines = [
        # Transmission backbone between substations
        {"id": "L_S1_S2", "from": "S1", "to": "S2", "rating_MW": 3500, "level": "T"},
        {"id": "L_S1_S3", "from": "S1", "to": "S3", "rating_MW": 3500, "level": "T"},
        {"id": "L_S2_S4", "from": "S2", "to": "S4", "rating_MW": 3000, "level": "T"},
        {"id": "L_S3_S4", "from": "S3", "to": "S4", "rating_MW": 3000, "level": "T"},
        {"id": "L_S2_S3", "from": "S2", "to": "S3", "rating_MW": 2500, "level": "T"},  # cross-link

        # Wind to North substation
        {"id": "L_W1_S1", "from": "W1", "to": "S1", "rating_MW": 400, "level": "T"},
        {"id": "L_W2_S1", "from": "W2", "to": "S1", "rating_MW": 400, "level": "T"},
        {"id": "L_W3_S1", "from": "W3", "to": "S1", "rating_MW": 450, "level": "T"},

        # West wind to West substation
        {"id": "L_W4_S2", "from": "W4", "to": "S2", "rating_MW": 300, "level": "T"},
        # East wind to multiple substations
        {"id": "L_W5_S1", "from": "W5", "to": "S1", "rating_MW": 300, "level": "T"},
        {"id": "L_W5_S3", "from": "W5", "to": "S3", "rating_MW": 300, "level": "T"},

        # PV connected to various substations
        {"id": "L_P1_S2", "from": "P1", "to": "S2", "rating_MW": 200, "level": "D"},
        {"id": "L_P2_S1", "from": "P2", "to": "S1", "rating_MW": 200, "level": "D"},
        {"id": "L_P3_S3", "from": "P3", "to": "S3", "rating_MW": 200, "level": "D"},
        {"id": "L_P4_S2", "from": "P4", "to": "S2", "rating_MW": 200, "level": "D"},
        {"id": "L_P5_S3", "from": "P5", "to": "S3", "rating_MW": 200, "level": "D"},
        {"id": "L_P6_S4", "from": "P6", "to": "S4", "rating_MW": 250, "level": "D"},
        {"id": "L_P7_S4", "from": "P7", "to": "S4", "rating_MW": 250, "level": "D"},
        {"id": "L_P8_S3", "from": "P8", "to": "S3", "rating_MW": 250, "level": "D"},

        # Fossil plants feeding South substation (and East)
        {"id": "L_F1_S4", "from": "F1", "to": "S4", "rating_MW": 700, "level": "T"},
        {"id": "L_F2_S4", "from": "F2", "to": "S4", "rating_MW": 700, "level": "T"},
        {"id": "L_F2_S3", "from": "F2", "to": "S3", "rating_MW": 500, "level": "T"},

        # BESS
        {"id": "L_B1_S1", "from": "B1", "to": "S1", "rating_MW": 200, "level": "D"},
        {"id": "L_B2_S4", "from": "B2", "to": "S4", "rating_MW": 200, "level": "D"},

        # Loads
        {"id": "L_L1_S2", "from": "L1", "to": "S2", "rating_MW": 700, "level": "D"},
        {"id": "L_L2_S3", "from": "L2", "to": "S3", "rating_MW": 700, "level": "D"},
        {"id": "L_L3_S4", "from": "L3", "to": "S4", "rating_MW": 900, "level": "D"},
        # Extra feed to industrial hub (more realistic meshing)
        {"id": "L_L3_S3", "from": "L3", "to": "S3", "rating_MW": 600, "level": "D"},
    ]
    return lines


# ==========================================================
# Grid helpers – flows + plotting
# ==========================================================

def _node_sign(n_type: str) -> float:
    if n_type in ["wind", "pv", "fossil", "bess"]:
        return 1.0  # exporting on average
    if n_type in ["load_res", "load_ind"]:
        return -1.0  # importing
    return 0.0  # neutral (substation)


def simulate_line_flows(nodes, lines, t_hours: float):
    """
    Smooth, time-dependent flows:
    - t_hours is a float (can include seconds fraction)
    - Mostly green, some yellow, rare red
    """
    node_by_id = {n["id"]: n for n in nodes}
    flows = {}

    hour = t_hours % 24.0

    for line in lines:
        n_from = node_by_id[line["from"]]
        n_to = node_by_id[line["to"]]

        sign_from = _node_sign(n_from["type"])
        sign_to = _node_sign(n_to["type"])
        base_scale = (abs(n_from["capacity_MW"]) + abs(n_to["capacity_MW"])) / 2

        # Daily PV profile
        pv_factor = max(0.0, np.sin((hour - 6.0) / 24.0 * 2.0 * np.pi))

        # Technology factor
        if n_from["type"] == "pv" or n_to["type"] == "pv":
            tech_factor = pv_factor
        elif n_from["type"] == "wind" or n_to["type"] == "wind":
            tech_factor = 0.6 + 0.2 * np.sin(2.0 * np.pi * t_hours / 24.0) + 0.1 * np.sin(
                2.0 * np.pi * t_hours / 6.0
            )
        else:
            tech_factor = 0.5 + 0.1 * np.sin(2.0 * np.pi * t_hours / 12.0)

        # Line-specific factor (deterministic, no RNG)
        line_hash = (abs(hash(line["id"])) % 1000) / 1000.0  # 0..1
        line_factor = 0.8 + 0.4 * line_hash  # 0.8..1.2

        # Small smooth wiggle to make it feel alive (seconds-scale)
        wiggle = 1.0 + 0.05 * np.sin(2.0 * np.pi * t_hours)

        # Magnitude tuned so most lines are under ~85–90% loading
        magnitude = 0.28 * base_scale * tech_factor * line_factor * wiggle

        # Direction from "more generating" to "more consuming" side
        direction = np.sign(sign_from - sign_to) or 1.0
        flow = direction * magnitude  # MW

        loading = abs(flow) / line["rating_MW"]

        if loading > 1.0:
            color = "red"
        elif loading > 0.85:
            color = "yellow"
        else:
            color = "green"

        flows[line["id"]] = {
            "flow_MW": flow,
            "loading": loading,
            "color": color,
            "direction": direction,
        }

    return flows


def add_node_icons(fig, nodes):
    """
    Add icons on nodes. Uses emoji by default.
    If PNG paths are provided and Pillow is installed, overlay PNGs.
    """
    emoji_map = {
        "substation": "🔌",
        "wind": "🌀",
        "pv": "☀️",
        "fossil": "⚡",
        "bess": "🔋",
        "load_res": "🏙️",
        "load_ind": "🏭",
    }

    by_type = {}
    for n in nodes:
        by_type.setdefault(n["type"], []).append(n)

    # Emoji scatter
    for n_type, n_list in by_type.items():
        icon = emoji_map.get(n_type, "•")
        fig.add_trace(
            go.Scatter(
                x=[n["x"] for n in n_list],
                y=[n["y"] for n in n_list],
                mode="text",
                text=[icon for _ in n_list],
                textposition="middle center",
                textfont=dict(size=20),
                name=n_type,
                hoverinfo="text",
                hovertext=[
                    f"{n['name']} ({n['type']})<br>Capacity: {n['capacity_MW']} MW"
                    for n in n_list
                ],
            )
        )

    # PNG overlay, if configured
    if PIL_AVAILABLE:
        for n_type, img_path in ICON_IMAGE_PATHS.items():
            if img_path is None:
                continue
            type_nodes = by_type.get(n_type, [])
            if not type_nodes:
                continue
            try:
                img = Image.open(img_path)
            except Exception:
                continue

            for n in type_nodes:
                fig.add_layout_image(
                    source=img,
                    x=n["x"],
                    y=n["y"],
                    xref="x",
                    yref="y",
                    sizex=1.5,
                    sizey=1.5,
                    xanchor="center",
                    yanchor="middle",
                    layer="above",
                )


def build_grid_figure(nodes, lines, flows):
    node_by_id = {n["id"]: n for n in nodes}
    fig = go.Figure()

    # Color mapping with RGBA for glow
    color_hex = {"green": "#27ae60", "yellow": "#f1c40f", "red": "#e74c3c"}
    glow_rgba = {
        "green": "rgba(39, 174, 96, 0.25)",
        "yellow": "rgba(241, 196, 15, 0.25)",
        "red": "rgba(231, 76, 60, 0.25)",
    }

    # Lines with glow + core + arrow
    for line in lines:
        n1 = node_by_id[line["from"]]
        n2 = node_by_id[line["to"]]
        f = flows[line["id"]]

        base_color = color_hex[f["color"]]
        glow_color = glow_rgba[f["color"]]

        core_width = 2 + 4 * min(f["loading"], 1.5)
        glow_width = core_width + 4

        x1, y1 = n1["x"], n1["y"]
        x2, y2 = n2["x"], n2["y"]

        # Glow trace
        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(color=glow_color, width=glow_width),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Core line trace
        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(color=base_color, width=core_width),
                hoverinfo="text",
                showlegend=False,
                text=(
                    f"{line['id']}<br>"
                    f"{n1['name']} → {n2['name']}<br>"
                    f"Flow: {f['flow_MW']:.0f} MW<br>"
                    f"Loading: {100 * f['loading']:.1f}%"
                ),
            )
        )

        # Arrow in direction of power flow
        if f["direction"] >= 0:
            xa, ya = (0.75 * x1 + 0.25 * x2), (0.75 * y1 + 0.25 * y2)
        else:
            xa, ya = (0.25 * x1 + 0.75 * x2), (0.25 * y1 + 0.75 * y2)

        fig.add_trace(
            go.Scatter(
                x=[xa],
                y=[ya],
                mode="text",
                text=["➤"],
                textfont=dict(size=14, color=base_color),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add node icons (emoji + optional PNGs)
    add_node_icons(fig, nodes)

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title="Demo Grid – Power Flows & Congestion",
        legend_title="Asset Type",
        margin=dict(l=20, r=20, t=40, b=20),
        height=600,
    )
    return fig


# ==========================================================
# PAGE 1 – Forecast view
# ==========================================================

def show_forecast_view():
    st.title("Spain PV & Wind Forecast")

    df = generate_dummy_forecast_data(n_hours=72)

    st.subheader("Forecast horizon")
    horizon = st.slider(
        "Select horizon (hours ahead)",
        min_value=24,
        max_value=72,
        value=48,
        step=6,
    )
    df_view = df.iloc[:horizon].copy()

    col_chart, col_metrics = st.columns([2.2, 1.0])

    with col_chart:
        fig_forecast = go.Figure()

        fig_forecast.add_trace(
            go.Scatter(
                x=df_view["time"],
                y=df_view["pv_actual"],
                mode="lines",
                name="PV Actual (MW)",
                line=dict(dash="solid"),
            )
        )
        fig_forecast.add_trace(
            go.Scatter(
                x=df_view["time"],
                y=df_view["pv_forecast"],
                mode="lines",
                name="PV Forecast (MW)",
                line=dict(dash="dash"),
            )
        )
        fig_forecast.add_trace(
            go.Scatter(
                x=df_view["time"],
                y=df_view["wind_actual"],
                mode="lines",
                name="Wind Actual (MW)",
                line=dict(dash="solid"),
            )
        )
        fig_forecast.add_trace(
            go.Scatter(
                x=df_view["time"],
                y=df_view["wind_forecast"],
                mode="lines",
                name="Wind Forecast (MW)",
                line=dict(dash="dash"),
            )
        )

        fig_forecast.update_layout(
            xaxis_title="Time",
            yaxis_title="Power (MW)",
            legend_title="Series",
            margin=dict(l=20, r=20, t=40, b=20),
            height=450,
        )
        st.plotly_chart(fig_forecast, width="stretch")

    with col_metrics:
        st.subheader("Model Performance")

        pv_metrics = compute_error_metrics(
            df_view["pv_actual"], df_view["pv_forecast"]
        )
        wind_metrics = compute_error_metrics(
            df_view["wind_actual"], df_view["wind_forecast"]
        )

        st.markdown("**PV Forecast**")
        c1, c2 = st.columns(2)
        c1.metric("MAE (MW)", f"{pv_metrics['MAE (MW)']:.0f}")
        c2.metric("RMSE (MW)", f"{pv_metrics['RMSE (MW)']:.0f}")
        c1.metric("Bias (MW)", f"{pv_metrics['Bias (MW)']:.0f}")
        if not np.isnan(pv_metrics["MAPE (%)"]):
            c2.metric("MAPE (%)", f"{pv_metrics['MAPE (%)']:.1f}")
        else:
            c2.metric("MAPE (%)", "N/A")

        st.markdown("---")
        st.markdown("**Wind Forecast**")
        c3, c4 = st.columns(2)
        c3.metric("MAE (MW)", f"{wind_metrics['MAE (MW)']:.0f}")
        c4.metric("RMSE (MW)", f"{wind_metrics['RMSE (MW)']:.0f}")
        c3.metric("Bias (MW)", f"{wind_metrics['Bias (MW)']:.0f}")
        if not np.isnan(wind_metrics["MAPE (%)"]):
            c4.metric("MAPE (%)", f"{wind_metrics['MAPE (%)']:.1f}")
        else:
            c4.metric("MAPE (%)", "N/A")

    st.markdown("---")
    st.info("Use the button below to open the grid view.")
    if st.button("Go to Grid View"):
        st.session_state.page = "grid"


# ==========================================================
# PAGE 2 – Grid view (48h window, second resolution, live + manual)
# ==========================================================

def show_grid_view():
    st.title("Spain Demo Grid View")

    st.caption(
        "Synthetic grid: 5 wind farms, 8 PV farms, 2 fossil plants, 2 BESS, "
        "4 substations, 3 main loads. In a real product, these nodes/lines come from your grid model."
    )

    nodes = get_demo_grid_nodes()
    lines = get_demo_grid_lines()

    # Two-day horizon in SECONDS
    if "grid_start" not in st.session_state:
        st.session_state.grid_start = datetime.now(timezone.utc).replace(
            microsecond=0
        )

    start = st.session_state.grid_start
    horizon_seconds = 48 * 3600  # 48 hours

    mode = st.radio(
        "Time mode",
        ["Live (real-time)", "Manual"],
        horizontal=True,
    )

    if mode == "Live (real-time)":
        now_utc = datetime.now(timezone.utc)
        delta_seconds = (now_utc - start).total_seconds()
        # Wrap into 0..horizon_seconds-1
        t_seconds = delta_seconds % horizon_seconds
        idx_seconds = int(t_seconds)
    else:
        idx_seconds = st.slider(
            "Select second in the 2-day window",
            min_value=0,
            max_value=horizon_seconds - 1,
            value=0,
            step=1,
        )
        t_seconds = float(idx_seconds)

    t_hours = t_seconds / 3600.0
    selected_time = start + timedelta(seconds=idx_seconds)

    st.caption(
        f"Showing grid state at: **{selected_time.strftime('%Y-%m-%d %H:%M:%S UTC')}**"
    )

    flows = simulate_line_flows(nodes, lines, t_hours=t_hours)
    fig_grid = build_grid_figure(nodes, lines, flows)

    col_grid, col_table = st.columns([2.0, 1.2])

    with col_grid:
        st.plotly_chart(fig_grid, width="stretch")

    with col_table:
        st.subheader("Line Loading Snapshot")

        data = []
        for line in lines:
            f = flows[line["id"]]
            data.append(
                {
                    "Line": line["id"],
                    "From": line["from"],
                    "To": line["to"],
                    "Rating (MW)": line["rating_MW"],
                    "Flow (MW)": round(f["flow_MW"], 1),
                    "Loading (%)": round(100 * f["loading"], 1),
                    "Status": "Emergency"
                    if f["color"] == "red"
                    else ("Warning" if f["color"] == "yellow" else "Normal"),
                }
            )
        df_lines = pd.DataFrame(data).sort_values("Loading (%)", ascending=False)
        st.dataframe(df_lines, width="stretch")

        st.markdown(
            """
            - **Green**: normal loading  
            - **Yellow**: high loading (warning)  
            - **Red**: near/over rating (emergency)
            """
        )

    st.markdown("---")
    back_clicked = st.button("Back to Forecast View")
    if back_clicked:
        st.session_state.page = "forecast"
        # Important: rerun immediately so we don't also trigger the live loop
        st.rerun()

    # Live mode: auto-advance every second
    if mode == "Live (real-time)":
        time.sleep(1)
        st.rerun()


# ==========================================================
# MAIN
# ==========================================================

def main():
    st.set_page_config(
        page_title="Spain PV & Wind – Demo",
        layout="wide",
    )

    if "page" not in st.session_state:
        st.session_state.page = "forecast"  # default view

    if st.session_state.page == "forecast":
        show_forecast_view()
    else:
        show_grid_view()


if __name__ == "__main__":
    main()

