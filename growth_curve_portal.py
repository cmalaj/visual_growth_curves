
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from io import StringIO
import matplotlib.colors as mcolors

st.set_page_config(layout="wide")
st.title("Growth Curve Visualisation Portal")

# --- Helper functions ---

@st.cache_data
def parse_metadata_file(uploaded_txt):
    layout = {}
    for line in uploaded_txt.getvalue().decode().splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2:
            well = parts[0].strip()
            label = parts[1].strip()
            layout[well] = label
    return layout

@st.cache_data
def load_csv(file):
    try:
        df = pd.read_csv(file, skiprows=lambda x: x < 2 or x == 3)
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)
        df["Plate"] = file.name
        df["Time_min"] = df["Time"].str.extract(r"(\d+)h").fillna(0).astype(int) * 60
        df["Time_min"] += df["Time"].str.extract(r"(\d+)\s?min").fillna(0).astype(int)
        df.set_index("Time_min", inplace=True)
        df.drop(columns=["Time"], inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to load {file.name}: {e}")
        return None

def get_linestyle_for_plate(plate_name, idx):
    styles = ["solid", "dash", "dot", "dashdot"]
    return styles[idx % len(styles)]

# --- File upload section ---

uploaded_files = st.file_uploader("Upload growth curve CSV files", type=["csv"], accept_multiple_files=True)
uploaded_metadata = st.file_uploader("Optional: Upload metadata layout file", type=["txt"])

# --- Load data and metadata ---

all_data = []
all_layouts = {}

if uploaded_files:
    for i, file in enumerate(uploaded_files):
        df = load_csv(file)
        if df is not None:
            plate = df["Plate"].iloc[0]
            all_data.append(df)
            all_layouts[plate] = {"well_map": {}}

    if uploaded_metadata:
        metadata_layout = parse_metadata_file(uploaded_metadata)
        for df in all_data:
            plate = df["Plate"].iloc[0]
            layout = {}
            for col in df.columns:
                if re.match(r"^[A-H]\d{1,2}$", col):
                    layout[col] = metadata_layout.get(col, col)
            all_layouts[plate]["well_map"] = layout

# --- Per-file layout and plots ---

if all_data:
    st.markdown("---")
    st.header("Individual Plate Layouts and Plots")

    for df in all_data:
        plate = df["Plate"].iloc[0]
        st.subheader(f"Plate: {plate}")

        layout_map = all_layouts.get(plate, {}).get("well_map", {})

        cols_in_plate = [col for col in df.columns if re.match(r"^[A-H]\d{1,2}$", col)]

        with st.expander("Customise well labels"):
            for well in cols_in_plate:
                default_label = layout_map.get(well, well)
                label = st.text_input(f"Label for {well}", value=default_label, key=f"{plate}_{well}_label")
                layout_map[well] = label

        st.markdown("**Growth Curves:**")
        time_unit = st.radio(
            f"X-axis time unit for {plate}",
            options=["Minutes", "Hours"],
            horizontal=True,
            key=f"time_unit_{plate}"
        )

        fig = go.Figure()
        for well in cols_in_plate:
            label = layout_map.get(well, well)
            x_vals = df.index if time_unit == "Minutes" else df.index / 60
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=df[well],
                mode='lines',
                name=label
            ))

        fig.update_layout(
            title=f"Growth Curves for {plate}",
            xaxis_title=f"Time ({time_unit})",
            yaxis_title="OD600",
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Comparison Plot Section ---
if all_data:
    st.markdown("---")
    st.header("Across-File Comparison")

    show_comparison = st.checkbox("Enable Comparison Plot", value=False)

    if show_comparison:
        compare_by = st.radio(
            "Compare by:",
            options=["Well location", "Custom label"],
            horizontal=True
        )

        time_unit_compare = st.radio(
            "X-axis time unit for comparison plot",
            options=["Minutes", "Hours"],
            horizontal=True,
            key="time_unit_compare"
        )

        label_pool = set()
        well_pool = set()

        for df in all_data:
            plate = df["Plate"].iloc[0]
            layout_map = all_layouts.get(plate, {}).get("well_map", {})
            for col in df.columns:
                if re.match(r"^[A-H]\d{1,2}$", col):
                    label = layout_map.get(col, col)
                    label_pool.add(label)
                    well_pool.add(col)

        selection = st.multiselect(
            "Select wells or labels to compare",
            options=sorted(well_pool if compare_by == "Well location" else label_pool),
            key="compare_selector"
        )

        fig = go.Figure()

        for idx, df in enumerate(all_data):
            plate = df["Plate"].iloc[0]
            layout_map = all_layouts.get(plate, {}).get("well_map", {})

            linestyle = get_linestyle_for_plate(plate, idx)
            matched = {}

            for col in df.columns:
                if not re.match(r"^[A-H]\d{1,2}$", col):
                    continue

                label = layout_map.get(col, col)
                match_key = col if compare_by == "Well location" else label

                if match_key in selection:
                    matched.setdefault(f"{plate}::{label}", []).append(col)

            for name, cols in matched.items():
                if not cols:
                    continue
                colour = "#1f77b4"
                x_vals = df.index if time_unit_compare == "Minutes" else df.index / 60
                values = df[cols].values
                mean_vals = np.nanmean(values, axis=1)
                std_vals = np.nanstd(values, axis=1)

                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=mean_vals,
                    mode='lines',
                    name=name,
                    line=dict(dash=linestyle)
                ))

        fig.update_layout(
            title="Comparison Plot",
            xaxis_title=f"Time ({time_unit_compare})",
            yaxis_title="OD600",
            legend_title="Sample",
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)
