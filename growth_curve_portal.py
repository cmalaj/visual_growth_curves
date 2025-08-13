import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from io import StringIO

# ==================== METADATA PARSER ====================

def parse_metadata_txt(metadata_file):
    content = metadata_file.read().decode("utf-8").splitlines()

    layouts = []
    current_plate = None
    capture_layout = False
    plate_layout = {}

    for line in content:
        line = line.strip()

        if line.startswith("--- Plate"):
            if plate_layout:
                layouts.append(plate_layout)
                plate_layout = {}
            capture_layout = False

        elif line.startswith("Plate Layout:"):
            capture_layout = True

        elif capture_layout and re.match(r"^[A-H]\t", line):
            parts = line.split('\t')
            row = parts[0]
            for col_idx, label in enumerate(parts[1:], start=1):
                well = f"{row}{col_idx}"
                plate_layout[well] = label

    if plate_layout:
        layouts.append(plate_layout)

    return layouts


# ==================== STREAMLIT APP START ====================

st.set_page_config(layout="wide")
st.title("Growth Curve Visualisation Portal")

uploaded_metadata = st.file_uploader("Upload metadata file (.txt)", type=["txt"])
metadata_layouts = []
if uploaded_metadata:
    try:
        metadata_layouts = parse_metadata_txt(uploaded_metadata)
        st.success(f"Metadata file parsed: {len(metadata_layouts)} plate layouts loaded.")
    except Exception as e:
        st.error(f"Error parsing metadata file: {e}")
        metadata_layouts = []

uploaded_files = st.file_uploader("Upload growth curve CSV files", type=["csv"], accept_multiple_files=True)

if not uploaded_files:
    st.stop()

well_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
well_numbers = [str(i) for i in range(1, 13)]
default_layout = {f"{row}{col}": f"{row}{col}" for row in well_letters for col in well_numbers}

all_data = {}

for file_index, file in enumerate(uploaded_files):
    with st.expander(f"📈 File {file_index + 1}: {file.name}", expanded=True):
        df = pd.read_csv(file, skiprows=2)  # Adjust skiprows if needed
        time_col = df.columns[0]
        time_vals = df[time_col].str.extract(r'(\d+)[hH]?\s*(\d*)[mM]?', expand=True).fillna(0).astype(int)
        time_minutes = time_vals[0] * 60 + time_vals[1]
        df['Time (min)'] = time_minutes
        df = df.drop(columns=[time_col])

        well_names = list(df.columns)
        df = df.melt(id_vars=["Time (min)"], var_name="Well", value_name="OD")

        # Use metadata layout if available
        layout = metadata_layouts[file_index] if file_index < len(metadata_layouts) else default_layout

        st.markdown("#### Custom Well Labels")
        col1, col2 = st.columns([3, 1])
        with col1:
            for row in well_letters:
                cols = st.columns(12)
                for col_idx, col in enumerate(well_numbers):
                    well = f"{row}{col}"
                    default_label = layout.get(well, well)
                    label = st.text_input(f"Label for {well}", value=default_label, key=f"{file_index}_{well}")
                    layout[well] = label
        with col2:
            st.write(" ")

        df["Label"] = df["Well"].map(layout)
        df_grouped = df.groupby(["Time (min)", "Label"]).OD.agg(["mean", "std"]).reset_index()

        unit_toggle = st.radio(f"Time unit for {file.name}", ["Minutes", "Hours"], horizontal=True, key=f"time_unit_{file_index}")
        x_vals = df_grouped["Time (min)"] / 60 if unit_toggle == "Hours" else df_grouped["Time (min)"]
        x_label = "Time (h)" if unit_toggle == "Hours" else "Time (min)"

        ymin, ymax = st.slider(f"Y-axis range for {file.name}", 0.0, 2.0, (0.0, 1.0), 0.05, key=f"yrange_{file_index}")

        fig, ax = plt.subplots(figsize=(12, 6))
        for label in df_grouped["Label"].unique():
            data = df_grouped[df_grouped["Label"] == label]
            ax.plot(x_vals[data.index], data["mean"], label=label)
            ax.fill_between(x_vals[data.index], data["mean"] - data["std"], data["mean"] + data["std"], alpha=0.3)
        ax.set_xlabel(x_label)
        ax.set_ylabel("OD")
        ax.set_ylim(ymin, ymax)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig)

        # Store for global comparison
        all_data[file_index] = {
            "df": df,
            "label_map": layout,
            "time_unit": unit_toggle,
        }

# ============== Comparison Section ==============

st.header("🔍 Comparison Plot Across Files")

comparison_options = {}
for idx, d in all_data.items():
    for well in d["df"]["Well"].unique():
        label = d["label_map"].get(well, well)
        comparison_options[f"{label} ({well}) from File {idx+1}"] = (idx, well)

selected_keys = st.multiselect("Select up to 5 profiles to compare", list(comparison_options.keys()), max_selections=5)

linestyles = ["-", "--", "-.", ":"]

if selected_keys:
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, key in enumerate(selected_keys):
        file_idx, well = comparison_options[key]
        d = all_data[file_idx]
        df_sub = d["df"][d["df"]["Well"] == well]
        x_vals = df_sub["Time (min)"] / 60 if d["time_unit"] == "Hours" else df_sub["Time (min)"]
        ax.plot(x_vals, df_sub["OD"], label=key, linestyle=linestyles[file_idx % len(linestyles)])
    ax.set_xlabel("Time (min/h)")
    ax.set_ylabel("OD")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig)