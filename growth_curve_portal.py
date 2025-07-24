import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from io import StringIO
import re
import copy

st.set_page_config(layout="wide")
st.title("Growth Curve Visualisation Portal v. 1.0")

# Generate 96 distinct colours from the rainbow colormap
rainbow_cmap = cm.get_cmap("gist_rainbow", 96)
well_order = [f"{row}{col}" for row in "ABCDEFGH" for col in range(1, 13)]
well_colours = {well: mcolors.to_hex(rainbow_cmap(i)) for i, well in enumerate(well_order)}

uploaded_files = st.file_uploader("Upload up to 4 LogPhase600 .txt files", type="txt", accept_multiple_files=True)

def time_to_minutes(t):
    h, m, s = map(int, t.split(":"))
    return h * 60 + m + s / 60

def parse_growth_file(file, plate_num):
    content = file.getvalue().decode("ISO-8859-1")
    lines = content.splitlines()

    header_line = next(i for i, line in enumerate(lines) if line.strip().startswith("Time"))
    headers = lines[header_line].split("\t")

    data_rows = []
    for row in lines[header_line + 1:]:
        if not re.match(r'^\d+:\d+:\d+', row):
            continue
        cols = row.split("\t")
        if len(cols) != len(headers):
            continue
        if sum([1 for v in cols[1:] if v.strip()]) < 5:
            continue
        data_rows.append(cols)

    df = pd.DataFrame(data_rows, columns=headers)
    df["Time"] = df["Time"].apply(time_to_minutes)
    for col in df.columns:
        if col != "Time":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Plate"] = f"Plate {plate_num}"
    return df.set_index("Time")

def generate_preset_layout(strain, phages):
    rows = list("ABCDEFGH")
    cols = [str(c) for c in range(1, 13)]
    layout_df = pd.DataFrame("", index=rows, columns=cols)

    tech_reps = ["T1", "T2"]
    batches = ["B1", "B2", "B3"]

    for row_idx, row_letter in enumerate(rows):
        phage_id = phages[row_idx // 2]
        tech_rep = tech_reps[row_idx % 2]

        # Columns 1–9 (MOI combinations)
        well_values = []
        for moi in ["MOI1", "MOI0.5", "MOI0.1"]:
            for batch in batches:
                label = f"{phage_id}_{moi}-{strain}_{batch}-{tech_rep}"
                well_values.append(label)

        # Columns 10–12 (specials)
        if row_idx == 0:
            well_values += [phage_id, "BROTH", f"{strain}_B1"]
        elif row_idx == 1:
            well_values += [phage_id, "VEHICLE", f"{strain}_B1"]
        elif row_idx == 2:
            well_values += [phage_id, "PAO1", "EMPTY"]
        elif row_idx == 3:
            well_values += [phage_id, "EMPTY", f"{strain}_B2"]
        elif row_idx == 4:
            well_values += [phage_id, "BROTH", f"{strain}_B2"]
        elif row_idx == 5:
            well_values += [phage_id, "VEHICLE", "EMPTY"]
        elif row_idx == 6:
            well_values += [phage_id, "PAO1", f"{strain}_B3"]
        elif row_idx == 7:
            well_values += [phage_id, "EMPTY", f"{strain}_B3"]

        layout_df.loc[row_letter, :] = well_values

    return layout_df

if uploaded_files:
    all_data = []
    all_layouts = {}  # Dict to store well label maps per plate
    all_summary = []

    for i, file in enumerate(uploaded_files):
        df = parse_growth_file(file, i + 1)
        plate_name = f"Plate {i + 1}"
        filename = file.name
        default_title = filename or plate_name
        custom_title = st.text_input(
            f"Custom Title for {plate_name}",
            value=default_title,
            key=f"title_{plate_name}"
        )


        st.markdown(f"---\n### {plate_name} Layout Settings")

        layout_mode = st.radio(
            f"Layout Mode for {plate_name}",
            ["Use preset layout", "Start with empty layout"],
            horizontal=True,
            key=f"layout_mode_{plate_name}"
        )

        host_strain = st.text_input(
            f"{plate_name} - Bacterial Host Strain", value="PAO1", key=f"strain_{plate_name}"
        )

        phage_input = st.text_input(
            f"{plate_name} - Phage(s) (comma-separated)", value="P1,P2,P3,P4", key=f"phages_{plate_name}"
        )

        phages = [p.strip() for p in phage_input.split(",") if p.strip()]
        well_label_map = {}

        if layout_mode == "Use preset layout" and len(phages) == 4:
            layout_df = generate_preset_layout(host_strain, phages)
            for row in layout_df.index:
                for col in layout_df.columns:
                    well = f"{row}{col}"
                    label = layout_df.loc[row, col]
                    well_label_map[well] = label

            # Optional: show layout table
            st.markdown(f"**{plate_name} - Auto-generated Layout Preview**")
            st.dataframe(layout_df, use_container_width=True)
        elif layout_mode == "Use preset layout":
            st.warning(f"{plate_name}: You must enter exactly 4 phages for the preset layout.")

        # Store layout
        all_layouts[plate_name] = well_label_map

        if df.empty:
            st.warning(f"The file **{file.name}** could not be processed (empty or invalid data). Skipping.")
            continue

        all_data.append(df)

        numeric_cols = df.columns.drop(["Plate"], errors="ignore")
        numeric_cols = [col for col in numeric_cols if not col.startswith("T°")]

        summary = pd.DataFrame({
            "Well": numeric_cols,
            "Mean": df[numeric_cols].mean(),
            "SD": df[numeric_cols].std()
        }).reset_index(drop=True)
        summary["Plate"] = f"Plate {i + 1}"
        all_summary.append(summary)

    # Sidebar: Time-series well selection controls
    st.sidebar.header("Time-Series Controls")

    all_rows = list("ABCDEFGH")
    all_cols = list(range(1, 13))

    # Row selector
    st.sidebar.subheader("Rows")
    select_all_rows = st.sidebar.checkbox("Select all rows", value=True)
    if select_all_rows:
        selected_rows = all_rows
    else:
        selected_rows = st.sidebar.multiselect("Choose rows (A–H):", all_rows, default=all_rows, key="row_select")

    # Column selector
    st.sidebar.subheader("Columns")
    select_all_cols = st.sidebar.checkbox("Select all columns", value=True)
    if select_all_cols:
        selected_cols = all_cols
    else:
        selected_cols = st.sidebar.multiselect("Choose columns (1–12):", all_cols, default=all_cols, key="col_select")

    # Per-plate visualisation
    for idx, df in enumerate(all_data):
        plate = df["Plate"].iloc[0]
        st.subheader(f"{plate} - Time Series")

        # Custom well labels for this plate only
        custom_labels = {}
        layout_map = all_layouts.get(plate, {})
        with st.sidebar.expander(f"Custom Labels for {plate}"):
            for row in selected_rows:
                for col_num in selected_cols:
                    well_id = f"{row}{col_num}"
                    default_label = layout_map.get(well_id, well_id)
                    label_key = f"{plate}_{well_id}_label"
                    label = st.text_input(f"{plate} - Label for {well_id}", value=default_label, key=label_key)
                    custom_labels[well_id] = label

        # Time unit toggle
        time_unit = st.radio(
            f"{plate} – X-axis time unit",
            options=["Minutes", "Hours"],
            horizontal=True,
            key=f"time_unit_{plate}"
        )

        # Axis range override UI
        with st.expander(f"Adjust axis ranges for {plate}"):
            col1, col2 = st.columns(2)
            with col1:
                x_min_raw = df.index.min()
                x_max_raw = df.index.max()
                x_min_default = x_min_raw if time_unit == "Minutes" else x_min_raw / 60
                x_max_default = x_max_raw if time_unit == "Minutes" else x_max_raw / 60

                x_min = st.number_input(f"{plate} X min ({time_unit})", value=x_min_default, step=0.1, key=f"{plate}_xmin")
                x_max = st.number_input(f"{plate} X max ({time_unit})", value=x_max_default, step=0.1, key=f"{plate}_xmax")
            with col2:
                y_min = st.number_input(f"{plate} Y min (OD600)", value=float(df.drop(columns='Plate', errors='ignore').min().min()), step=0.1, key=f"{plate}_ymin")
                y_max = st.number_input(f"{plate} Y max (OD600)", value=float(df.drop(columns='Plate', errors='ignore').max().max()), step=0.1, key=f"{plate}_ymax")


        # Blank correction UI
        with st.expander(f"Blank Correction for {plate}"):
            apply_blank = st.checkbox(f"Apply blank correction for {plate}", key=f"{plate}_blank_toggle")
            blank_well = st.selectbox(
                f"Select blank well for {plate}",
                options=[f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)],
                index=95,  # Default to H12
                key=f"{plate}_blank_select"
            )
        if apply_blank and blank_well in df.columns:
            df_corrected = df.copy()
            blank_values = df[blank_well]
            for col in df.columns:
                for col in df.filter(regex=r"^[A-H]\d{1,2}$").columns:
                    df_corrected[col] = df[col] - blank_values
            df = df_corrected



        # Build plot
        fig = go.Figure()

        for col in df.columns:
            if col not in ["Plate"] and not col.startswith("T°"):
                match = re.match(r"([A-H])(\d{1,2})", col)
                if not match:
                    continue
                row, col_num = match.groups()
                col_num = int(col_num)
                well_id = f"{row}{col_num}"
                if row not in selected_rows or col_num not in selected_cols:
                    continue
                colour = well_colours.get(well_id, "#CCCCCC")  # fallback grey
                label = custom_labels.get(well_id, well_id)
                x_vals = df.index if time_unit == "Minutes" else df.index / 60
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=df[col],
                    name=label,
                    mode='lines',
                    line=dict(color=colour)
                ))
        

        fig.update_layout(
            title=custom_title,
            xaxis_title=f"Time ({time_unit})",
            yaxis_title="OD600",
            legend_title="Well Label",
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max])
        )

        st.plotly_chart(fig, use_container_width=True)

    