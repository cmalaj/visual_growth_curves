
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import re

st.set_page_config(layout="wide")
st.title("Growth Curve Visualisation Portal")

# Helper to convert time column from '0h 30 min' to float (minutes or hours)
def parse_time_column(col):
    time_pattern = re.compile(r'(?:(\d+)h)?\s*(?:(\d+)\s*min)?')
    result = []
    for val in col:
        match = time_pattern.match(str(val))
        if match:
            hours = int(match.group(1)) if match.group(1) else 0
            minutes = int(match.group(2)) if match.group(2) else 0
            result.append(hours * 60 + minutes)
        else:
            result.append(np.nan)
    return result

# --- Metadata parsing section ---
def parse_metadata_file(file):
    try:
        raw_text = file.read().decode("utf-8")
        plate_layouts = {}
        current_plate = None
        in_layout_section = False

        for line in raw_text.splitlines():
            line = line.strip()
            if line.startswith("--- Plate"):
                current_plate = f"Plate {len(plate_layouts) + 1}"
                plate_layouts[current_plate] = {}
                in_layout_section = False
            elif line.startswith("Plate Layout"):
                in_layout_section = True
                continue
            elif in_layout_section:
                if not line or line.startswith("Plate ID") or line.startswith("Strain") or line.startswith("Phage"):
                    continue
                row_data = re.split(r"\t|\s{2,}", line)
                if len(row_data) < 13:
                    continue
                row_letter = row_data[0].strip()
                for col_index, cell_value in enumerate(row_data[1:], start=1):
                    well_id = f"{row_letter}{col_index}"
                    plate_layouts[current_plate][well_id] = cell_value.strip()
        return plate_layouts
    except Exception as e:
        st.error(f"Error parsing metadata file: {e}")
        return {}

# --- File upload and metadata section ---
uploaded_metadata = st.file_uploader("Upload optional plate layout metadata file (.txt)", type=["txt"], key="metadata")
metadata_layouts = {}
if uploaded_metadata:
    metadata_layouts = parse_metadata_file(uploaded_metadata)

uploaded_files = st.file_uploader("Upload growth curve CSV files", type=["csv"], accept_multiple_files=True)

plate_data = {}
file_labels = []
if uploaded_files:
    for idx, file in enumerate(uploaded_files):
        df = pd.read_csv(file, skiprows=0)
        df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
        time_col = df.columns[0]
        df[time_col] = parse_time_column(df[time_col])
        df = df.rename(columns={time_col: "Time"})
        plate_name = f"Plate {idx+1}"
        plate_data[plate_name] = df
        file_labels.append(file.name)

# --- Display with metadata integration ---
if plate_data:
    for plate_name, df in plate_data.items():
        st.subheader(f"{plate_name} — {file_labels[int(plate_name.split()[-1]) - 1]}")
        metadata = metadata_layouts.get(plate_name, {})
        wells = df.columns[1:]
        labels = {}
        for well in wells:
            label_key = f"{plate_name}_{well}"
            default_label = metadata.get(well, well)
            labels[well] = st.text_input(f"Label for {well}", value=default_label, key=label_key)
        df_labeled = df.rename(columns=labels)
        # Plot
        fig = go.Figure()
        for label in df_labeled.columns[1:]:
            fig.add_trace(go.Scatter(x=df_labeled["Time"], y=df_labeled[label], mode="lines", name=label))
        fig.update_layout(title="Growth Curves", xaxis_title="Time (min)", yaxis_title="OD", height=400)
        st.plotly_chart(fig, use_container_width=True)
