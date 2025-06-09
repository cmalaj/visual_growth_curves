
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
st.title("Growth Curve Visualisation Portal")


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

if uploaded_files:
    all_data = []
    all_summary = []

    for i, file in enumerate(uploaded_files):
        df = parse_growth_file(file, i + 1)

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

    # Add well selection filters
    # Sidebar: Time-series well selection controls
    st.sidebar.header("Time-Series Controls")

    # Define full row/col lists
    all_rows = list("ABCDEFGH")
    all_cols = list(range(1, 13))

    # Add row toggle buttons
    st.sidebar.subheader("Rows")
    row_col1, row_col2 = st.sidebar.columns([1, 2])
    with row_col1:
        if st.button("Select all rows"):
            selected_rows = all_rows
        else:
            selected_rows = st.sidebar.multiselect("Choose rows (A–H):", all_rows, default=all_rows, key="row_select")

    # Add column toggle buttons
    st.sidebar.subheader("Columns")
    col_col1, col_col2 = st.sidebar.columns([1, 2])
    with col_col1:
        if st.button("Select all columns"):
            selected_cols = all_cols
        else:
            selected_cols = st.sidebar.multiselect("Choose columns (1–12):", all_cols, default=all_cols, key="col_select")  
    
    
    # Interactive line plots using Plotly
    for idx, df in enumerate(all_data):
        plate = df["Plate"].iloc[0]
        st.subheader(f"{plate} - Time Series")

        # Axis range override UI
        with st.expander(f"Adjust axis ranges for {plate}"):
            col1, col2 = st.columns(2)
            with col1:
                x_min = st.number_input(f"{plate} X min (minutes)", value=float(df.index.min()), step=1.0, key=f"{plate}_xmin")
                x_max = st.number_input(f"{plate} X max (minutes)", value=float(df.index.max()), step=1.0, key=f"{plate}_xmax")
            with col2:
                y_min = st.number_input(f"{plate} Y min (OD600)", value=float(df.drop(columns='Plate', errors='ignore').min().min()), step=0.1, key=f"{plate}_ymin")
                y_max = st.number_input(f"{plate} Y max (OD600)", value=float(df.drop(columns='Plate', errors='ignore').max().max()), step=0.1, key=f"{plate}_ymax")

        # Build plot
        fig = go.Figure()

        for col in df.columns:
            if col not in ["Plate"] and not col.startswith("T°"):
                match = re.match(r"([A-H])(\d{1,2})", col)
                if not match:
                    continue
                row, col_num = match.groups()
                col_num = int(col_num)
                if row not in selected_rows or col_num not in selected_cols:
                    continue
                colour = well_colours.get(col, "#CCCCCC")  # fallback grey
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col,
                    mode='lines',
                    line=dict(color=colour)
                ))

        fig.update_layout(
            xaxis_title="Time (minutes)",
            yaxis_title="OD600",
            legend_title="Well ID",
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max])
        )

        st.plotly_chart(fig, use_container_width=True)

    

    # Generalised Heatmap Visualisation for "Mean" and "SD"
    metrics = ["Mean", "SD"]
    fig, axes = plt.subplots(len(metrics), len(all_summary), figsize=(5 * len(all_summary), 5 * len(metrics)))

    if len(all_summary) == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # Ensure 2D shape

    for j, metric in enumerate(metrics):
        for i, summary in enumerate(all_summary):
            plate = summary["Plate"].iloc[0]
            sub = summary[["Well", metric]]

            # Dynamically extract row and column layout from wells
            well_ids = sub["Well"].dropna().unique()
            rows = sorted(set([w[0] for w in well_ids if re.match(r"^[A-Z]\d+$", w)]))
            cols = sorted(set([int(re.search(r"\d+$", w).group()) for w in well_ids if re.match(r"^[A-Z]\d+$", w)]))

            heatmap = pd.DataFrame(index=rows, columns=cols, dtype=float)

            for _, row in sub.iterrows():
                match = re.match(r"([A-Z])(\d{1,2})", row["Well"])
                if match:
                    r, c = match.groups()
                    if r in heatmap.index and int(c) in heatmap.columns:
                        heatmap.loc[r, int(c)] = row[metric]

            heatmap.columns = heatmap.columns.astype(int)

            sns.heatmap(
                heatmap,
                ax=axes[j][i],
                cmap="rainbow_r",
                annot=False,
                cbar=True
            )
            axes[j][i].set_title(f"{plate} - {metric}")
            axes[j][i].set_xlabel("Column")
            axes[j][i].set_ylabel("Row")

    plt.tight_layout()
    st.subheader("Plate Summary Heatmaps")
    st.pyplot(fig)

    