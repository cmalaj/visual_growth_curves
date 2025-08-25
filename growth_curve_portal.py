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
from io import BytesIO
import re
import copy
import scipy
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Growth Curve Visualisation Portal v. 1.0")

DEFAULT_TIME_UNIT = "Hours"

# Generate 96 distinct colours from the rainbow colormap
rainbow_cmap = cm.get_cmap("gist_rainbow", 96)
well_order = [f"{row}{col}" for row in "ABCDEFGH" for col in range(1, 13)]
well_colours = {well: mcolors.to_hex(rainbow_cmap(i)) for i, well in enumerate(well_order)}

uploaded_files = st.file_uploader("Upload up to 4 LogPhase600 .txt files", type="txt", accept_multiple_files=True)
all_data = []  # Ensure this exists for conditional rendering later

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

        st.markdown(f"---\n### {plate_name} Layout Settings")

        custom_title = st.text_input(
            f"Custom Title for {plate_name}",
            value=default_title,
            key=f"title_{plate_name}"
        )

        plate_titles = st.session_state.setdefault("plate_titles", {})
        plate_titles[plate_name] = custom_title


        

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

    # -------------------------
    # Align all growth curves for comparison plot
    # -------------------------
    all_data_scaled = []
    for df in all_data:
        df_scaled = df.copy()
        well_cols = df_scaled.filter(regex=r"^[A-H]\d{1,2}$").columns
        baseline_vals = df_scaled[well_cols].iloc[0]
        for col in well_cols:
            baseline = baseline_vals[col]
            if pd.notna(baseline):
                df_scaled[col] = df_scaled[col] - baseline + 0.0001
        all_data_scaled.append(df_scaled)

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
            key=f"time_unit_{plate}",
            index=1 if DEFAULT_TIME_UNIT == "Hours" else 0
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


        group_replicates = st.checkbox(
            f"Group technical replicates for {plate}?",
            value=True,
            key=f"group_reps_{plate}"
        )
        # Align all wells to start at OD600 = 0.0001
        baseline_vals = df.filter(regex=r"^[A-H]\d{1,2}$").iloc[0]
        df_scaled = df.copy()
        for col in baseline_vals.index:
            baseline = baseline_vals[col]
            if pd.notna(baseline):
                df_scaled[col] = df[col] - baseline + 0.0001
        
        # Build plot
        fig = go.Figure()

        if group_replicates:
            import re

            # Group by normalized label (e.g., strip '-T1', '-T2')
            label_to_wells = {}

            for col in df.columns:
                if col in ["Plate"] or col.startswith("T°"):
                    continue
                match = re.match(r"([A-H])(\d{1,2})", col)
                if not match:
                    continue
                row, col_num = match.groups()
                col_num = int(col_num)
                well_id = f"{row}{col_num}"
                if row not in selected_rows or col_num not in selected_cols:
                    continue

                # Get the label from layout or fallback
                label = custom_labels.get(well_id, well_id)

                # ✅ Strip trailing '-T1', '-T2', etc. for grouping
                group_label = re.sub(r"-T\d$", "", label)

                label_to_wells.setdefault(group_label, []).append(col)

            for group_label, replicate_cols in label_to_wells.items():
                if not replicate_cols:
                    continue
                # Use first well to determine color
                colour = well_colours.get(replicate_cols[0], "#CCCCCC")
                x_vals = df_scaled.index if time_unit == "Minutes" else df_scaled.index / 60
                values = df_scaled[replicate_cols].values
                mean_vals = np.nanmean(values, axis=1)
                std_vals = np.nanstd(values, axis=1)

                # Convert matplotlib RGBA to valid Plotly rgba string
                rgba = mcolors.to_rgba(colour, alpha=0.2)
                fillcolor = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"

                # Mean line
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=mean_vals,
                    mode='lines',
                    name=group_label,
                    line=dict(color=colour, width=2),
                    legendgroup=group_label,
                    showlegend=True
                ))

                # SD ribbon
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_vals, x_vals[::-1]]),
                    y=np.concatenate([mean_vals + std_vals, (mean_vals - std_vals)[::-1]]),
                    fill='toself',
                    fillcolor=fillcolor,
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=group_label
                ))

        else:
            # Fall back to individual wells
            for col in df.columns:
                if col in ["Plate"] or col.startswith("T°"):
                    continue
                match = re.match(r"([A-H])(\d{1,2})", col)
                if not match:
                    continue
                row, col_num = match.groups()
                col_num = int(col_num)
                well_id = f"{row}{col_num}"
                if row not in selected_rows or col_num not in selected_cols:
                    continue
                label = custom_labels.get(well_id, well_id)
                colour = well_colours.get(well_id, "#CCCCCC")
                x_vals = df.index if time_unit == "Minutes" else df.index / 60

                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=df_scaled[col],
                    name=label,
                    mode='lines',
                    line=dict(color=colour)
                ))

        # Final plot layout + render
        title = st.session_state["plate_titles"].get(plate, plate)
        fig.update_layout(
            title=title,
            xaxis_title=f"Time ({time_unit})",
            yaxis_title="OD600",
            legend_title="Well Label",
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[0, y_max])
        )

        st.plotly_chart(fig, use_container_width=True)



###############
# Bacterial growth threshold analysis
###############

# ----------------------
# Helper for session state
# ----------------------
def toggle_well_selection(plate, well):
    key = f"selected_wells_{plate}"
    if key not in st.session_state:
        st.session_state[key] = set()
    if well in st.session_state[key]:
        st.session_state[key].remove(well)
    else:
        st.session_state[key].add(well)


# ----------------------
# Main Threshold Analysis Section
# ----------------------
if all_data:
    st.markdown("---")
    st.header("Growth Threshold Analysis")

    thresholds = [1000, 2000, 5000, 7500]

    for idx, df in enumerate(all_data_scaled):
        plate = df["Plate"].iloc[0]
        candidate_wells = [col for col in df.columns if re.fullmatch(r"[A-H]1[0-2]?|[A-H][1-9]", col)]

        st.subheader(f"{plate} – Select Bacterial Control Wells by Position")
        preferred_default_wells = ["A12", "B12", "D12", "E12", "G12", "H12"]
        default = [w for w in preferred_default_wells if w in candidate_wells]

        selected_positions = st.multiselect(
            f"Choose control well positions (e.g. A12, B11) for {plate}",
            options=candidate_wells,
            default=default
        )

        if not selected_positions:
            st.warning(f"No control well positions selected for {plate}. Skipping.")
            continue

        mean_vals = df[selected_positions].mean(axis=1)
        baseline = mean_vals.iloc[0]
        time_vals = df.index

        st.subheader(f"{plate} – Select Growth Threshold for AUC Integration")
        threshold_to_use = st.selectbox(
            f"Select threshold (× growth) for {plate}",
            thresholds,
            index=0,
            key=f"threshold_selector_{plate}"
        )

        # Threshold Crossing
        thresh_val = baseline * threshold_to_use
        cross_idx = np.argmax(mean_vals.values >= thresh_val)
        cross_time = time_vals[cross_idx] if cross_idx < len(time_vals) else None

        # Interactive Curve Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_vals,
            y=mean_vals,
            name="Mean Control Growth",
            line=dict(color="blue", width=3)
        ))

        if cross_time is not None:
            fig.add_shape(
                type="line",
                x0=cross_time, x1=cross_time,
                y0=0, y1=mean_vals.max() * 1.1,
                line=dict(dash="dash", color="red")
            )
            fig.add_trace(go.Scatter(
                x=[cross_time], y=[thresh_val],
                mode="markers+text",
                marker=dict(color="red", size=6),
                text=[f"{threshold_to_use}×"],
                textposition="top center",
                showlegend=False
            ))

        # Plot additional selected well curves
        well_key = f"selected_wells_{plate}"
        selected_wells = st.session_state.get(well_key, set())
        color_cycle = plt.cm.tab10.colors
        for i, well in enumerate(selected_wells):
            if well in df.columns:
                fig.add_trace(go.Scatter(
                    x=time_vals,
                    y=df[well],
                    name=f"Well {well}",
                    line=dict(color=f"rgba{color_cycle[i % 10]}", width=2)
                ))

        fig.update_layout(
            title=f"{st.session_state['plate_titles'].get(plate, plate)} – Growth Curves",
            xaxis_title="Time (minutes)",
            yaxis_title="OD600",
            yaxis=dict(range=[0, mean_vals.max() * 1.1]),
            margin=dict(l=50, r=50, t=50, b=50),
        )

        #st.plotly_chart(fig, use_container_width=True, key=f"growth_plot_base_{plate}_{idx}")

        # -------------------------
        # ΔAUC Well Grid Section (Streamlit-Compatible)
        # -------------------------
        if cross_time is not None:
            valid_mask = time_vals <= cross_time
            control_auc = np.trapz(mean_vals[valid_mask], time_vals[valid_mask])

            delta_auc_grid = pd.DataFrame(index=list("ABCDEFGH"), columns=[str(i) for i in range(1, 13)])
            well_to_auc_diff = {}

            for well in candidate_wells:
                row, col = well[0], well[1:]
                if well not in df.columns or row not in delta_auc_grid.index or col not in delta_auc_grid.columns:
                    continue
                curve = df[well]
                well_auc = np.trapz(curve[valid_mask], time_vals[valid_mask])
                delta_auc = well_auc - control_auc
                delta_auc_grid.loc[row, col] = delta_auc
                well_to_auc_diff[well] = delta_auc

            delta_auc_grid = delta_auc_grid.apply(pd.to_numeric)

            # Set up color normalization and colormap
            norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=delta_auc_grid.min().min(), vmax=delta_auc_grid.max().max())
            cmap = cm.get_cmap("coolwarm_r")

            st.subheader(f"{plate} – ΔAUC Well Grid (up to {threshold_to_use}×)")

            # Session key for storing selected wells
            session_key = f"selected_wells_{plate}_{idx}"
            if session_key not in st.session_state:
                st.session_state[session_key] = []

            selected_wells = st.session_state[session_key]

            # Build grid
            for row in list("ABCDEFGH"):
                cols = st.columns(12)
                for i, col_num in enumerate(range(1, 13)):
                    well_id = f"{row}{col_num}"
                    delta = delta_auc_grid.loc[row, str(col_num)]

                    if pd.isna(delta):
                        cols[i].markdown(" ", unsafe_allow_html=True)
                        continue

                    colour = mcolors.to_hex(cmap(norm(delta)))
                    button_label = f"{well_id}"

                    button_html = f"""
                        <div style="
                            background-color: {colour};
                            border: 1px solid #333;
                            border-radius: 3px;
                            padding: 0.2rem;
                            text-align: center;
                            font-size: 11px;
                            font-weight: bold;
                            height: 24px;
                            line-height: 1.2;
                        ">
                            {button_label}
                        </div>
                    """

                    # Make it clickable with invisible Streamlit button
                    if cols[i].button(button_label, key=f"{plate}_{well_id}_{idx}"):
                        if well_id in selected_wells:
                            selected_wells.remove(well_id)
                        else:
                            selected_wells.append(well_id)

                    # Overlay the coloured div to simulate background colour
                    cols[i].markdown(button_html, unsafe_allow_html=True)

            # Plot selected wells
            for well in selected_wells:
                if well in df.columns:
                    fig.add_trace(go.Scatter(
                        x=time_vals,
                        y=df[well],
                        name=f"Well {well}",
                        mode="lines",
                        line=dict(dash="dot", width=2)
                    ))

            st.plotly_chart(fig, use_container_width=True, key=f"growth_plot_updated_{plate}_{idx}")


# ========================
# Comparison Plot Section
# ========================

if all_data:  # Only run if data has been loaded
    st.markdown("---")
    st.header("Comparison Plot")

    # Time unit selection for comparison plot
    comparison_time_unit = st.radio(
        "Time Axis Unit for Comparison Plot",
        options=["Minutes", "Hours"],
        horizontal=True,
        key="comparison_time_unit",
        index=1 if DEFAULT_TIME_UNIT == "Hours" else 0
    )


    # NEW: Option to select same wells across all plates
    use_shared_selection = st.checkbox("Use same wells across all plates?", value=False)
    
    # Apply alignment
    apply_alignment = st.checkbox("Align curves to start at OD600 = 0.0001", value=True)
    data_source = all_data_scaled if apply_alignment else all_data

    # Store selected wells per plate
    selected_wells_per_plate = {}

    st.subheader("Select wells to compare")

    if use_shared_selection:
        # Shared well selector
        shared_wells = st.multiselect(
            "Select wells (applies to all plates)",
            options=[f"{row}{col}" for row in "ABCDEFGH" for col in range(1, 13)],
            help="These wells will be plotted from all uploaded plates.",
            key="shared_well_selector"
        )
        show_mean_with_ribbon = st.checkbox(
            "Show average ± SD for selected wells",
            value=True,
            help="Plots the average profile across all plates for each selected well with a shaded SD band"
        )
        # Assign selected wells to each plate
        for df in data_source:
            plate = df["Plate"].iloc[0]
            wells_in_plate = [col for col in df.columns if re.match(r"^[A-H]\d{1,2}$", col)]
            valid_shared = [w for w in shared_wells if w in wells_in_plate]
            if valid_shared:
                selected_wells_per_plate[plate] = valid_shared

    else:
        # Per-plate multiselects
        for df in data_source:
            plate = df["Plate"].iloc[0]
            wells = [col for col in df.columns if re.match(r"^[A-H]\d{1,2}$", col)]

            selected = st.multiselect(
                f"{plate} – Select wells to compare",
                options=wells,
                key=f"compare_select_{plate}"
            )

            if selected:
                selected_wells_per_plate[plate] = selected

    # Axis range control
    with st.expander("Adjust axes for comparison plot"):
        col1, col2 = st.columns(2)

        with col1:
            all_times = pd.concat([pd.Series(df.index) for df in all_data])
            x_min_default = all_times.min() if comparison_time_unit == "Minutes" else all_times.min() / 60
            x_max_default = all_times.max() if comparison_time_unit == "Minutes" else all_times.max() / 60
            comp_x_min = st.number_input("X min", value=x_min_default, step=0.1, key="comp_xmin")
            comp_x_max = st.number_input("X max", value=x_max_default, step=0.1, key="comp_xmax")

        with col2:
            all_values = pd.concat([df.drop(columns=["Plate"], errors="ignore") for df in data_source], axis=1)
            y_min_default = all_values.min().min()
            y_max_default = all_values.max().max()
            comp_y_min = st.number_input("Y min (OD600)", value=y_min_default, step=0.1, key="comp_ymin")
            comp_y_max = st.number_input("Y max (OD600)", value=y_max_default, step=0.1, key="comp_ymax")

    # Plot if any wells are selected
    if any(selected_wells_per_plate.values()):
        fig = go.Figure()

        if use_shared_selection and show_mean_with_ribbon:
            # For each shared well, collect matching data across plates
            for well_id in shared_wells:
                time_grid = None
                all_profiles = []

                for df in all_data:
                    if well_id in df.columns:
                        x_vals = df.index if comparison_time_unit == "Minutes" else df.index / 60
                        y_vals = df[well_id]

                        if time_grid is None:
                            time_grid = x_vals
                        all_profiles.append(y_vals.values)

                if all_profiles:
                    y_array = np.array(all_profiles)
                    mean_vals = np.nanmean(y_array, axis=0)
                    std_vals = np.nanstd(y_array, axis=0)

                    colour = well_colours.get(well_id, "#CCCCCC")

                    # Convert matplotlib RGBA to valid Plotly 'rgba(...)' string
                    rgba = mcolors.to_rgba(colour, alpha=0.2)
                    fillcolor = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"

                    # Mean line
                    fig.add_trace(go.Scatter(
                        x=time_grid,
                        y=mean_vals,
                        mode='lines',
                        name=f"{well_id} – Mean",
                        line=dict(color=colour, width=2),
                        legendgroup=well_id,         # 🔗 Link to the same group
                        showlegend=True
                    ))

                    # Shaded SD ribbon
                    fig.add_trace(go.Scatter(
                    x=np.concatenate([time_grid, time_grid[::-1]]),
                    y=np.concatenate([mean_vals + std_vals, (mean_vals - std_vals)[::-1]]),
                    fill='toself',
                    fillcolor=fillcolor,
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=well_id  # Ribbon toggles with mean line
                ))

        else:
            # Default: plot each trace individually
            for plate_name, well_list in selected_wells_per_plate.items():
                df = next((d for d in data_source if d["Plate"].iloc[0] == plate_name), None)
                if df is None:
                    continue

                for well_id in well_list:
                    if well_id not in df.columns:
                        continue

                    custom_key = f"{plate_name}_{well_id}_label"
                    label = st.session_state.get(custom_key, f"{plate_name} - {well_id}")
                    colour = well_colours.get(well_id, "#CCCCCC")
                    x_vals = df.index if comparison_time_unit == "Minutes" else df.index / 60

                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=df[well_id],
                        name=label,
                        mode='lines',
                        line=dict(color=colour)
                    ))

        fig.update_layout(
            title="Overlay Comparison Plot",
            xaxis_title=f"Time ({comparison_time_unit})",
            yaxis_title="OD600",
            legend_title="Well Label",
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(range=[comp_x_min, comp_x_max]),
            yaxis=dict(range=[comp_y_min, comp_y_max])
        )

        st.plotly_chart(fig, use_container_width=True)