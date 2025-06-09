
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
st.title("Growth Curve Visualisation Portal (Interactive + Heatmaps)")


# Custom colormaps per row group
custom_colormaps = {
    "A": LinearSegmentedColormap.from_list("reds_custom", ["#8B0000", "#FFA07A"]),        # Dark red to light red
    "B": LinearSegmentedColormap.from_list("oranges_custom", ["#FF8C00", "#FFE4B5"]),     # Dark orange to light
    "C": LinearSegmentedColormap.from_list("yellows_custom", ["#CCCC00", "#FFFFE0"]),     # Mustard to pale yellow
    "D": LinearSegmentedColormap.from_list("greens_custom", ["#006400", "#98FB98"]),      # Forest green to mint
    "E": LinearSegmentedColormap.from_list("teals_custom", ["#008080", "#AFEEEE"]),       # Teal to pale turquoise
    "F": LinearSegmentedColormap.from_list("blues_custom", ["#00008B", "#ADD8E6"]),       # Navy to light blue
    "G": LinearSegmentedColormap.from_list("indigos_custom", ["#4B0082", "#D8BFD8"]),     # Indigo to thistle
    "H": LinearSegmentedColormap.from_list("violets_custom", ["#800080", "#E6E6FA"]),     # Purple to lavender
}

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

    # Interactive line plots using Plotly
    for df in all_data:
        plate = df["Plate"].iloc[0]
        st.subheader(f"{plate} - Time Series (Coloured by Row ID)")

        fig = go.Figure()

        for col in df.columns:
            if col not in ["Plate"] and not col.startswith("T°"):
                match = re.match(r"([A-H])(\d{1,2})", col)
                if match:
                    row_letter, col_num = match.groups()
                    col_num = int(col_num)

                    # Get appropriate custom colormap
                    base_cmap = custom_colormaps.get(row_letter)
                    if base_cmap:
                        norm = mcolors.Normalize(vmin=1, vmax=12)
                        rgba = base_cmap(norm(col_num))
                        hex_colour = mcolors.to_hex(rgba)
                    else:
                        hex_colour = "#CCCCCC"  # fallback grey
                else:
                    hex_colour = "#CCCCCC"

                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col,
                    mode='lines',
                    line=dict(color=hex_colour)
                ))

        fig.update_layout(
            xaxis_title="Time (minutes)",
            yaxis_title="OD600",
            legend_title="Well ID",
            margin=dict(l=50, r=50, t=50, b=50)
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
