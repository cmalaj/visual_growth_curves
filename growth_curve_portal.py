
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from io import StringIO
import re
import copy

st.set_page_config(layout="wide")
st.title("Growth Curve Visualisation Portal (Interactive + Heatmaps)")

uploaded_files = st.file_uploader("Upload up to 4 LogPhase600 .txt files", type="txt", accept_multiple_files=True)

def time_to_minutes(t):
    h, m, s = map(int, t.split(":"))
    return h * 60 + m + s / 60

def parse_growth_file(file, plate_num):
    content = file.getvalue().decode('ISO-8859-1')
    lines = content.splitlines()

    header_line = next(i for i, line in enumerate(lines) if line.strip().startswith('Time'))
    headers = lines[header_line].split("\t")

    data_rows = []
    for row in lines[header_line + 1:]:
        if not re.match(r'^\d+:\d+:\d+', row):
            continue
        cols = row.split("\t")
        if len(cols) != len(headers):
            continue
        # Skip rows that are mostly empty
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
            st.warning(f"⚠️ The file **{file.name}** could not be processed (empty or invalid data). Skipping.")
            continue

        all_data.append(df)

        numeric_cols = df.columns.drop(["Plate"], errors="ignore")
        numeric_cols = [col for col in numeric_cols if not col.startswith("T°")]

        summary = pd.DataFrame({
            "Well": numeric_cols,
            "Mean": df[numeric_cols].mean(),
            "SD": df[numeric_cols].std()
        }).reset_index(drop=True)
        summary["Plate"] = df["Plate"].iloc[0]
        all_summary.append(summary)

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
        st.subheader(f"{plate} - Time Series (Interactive)")
        fig = go.Figure()
        for col in df.columns:
            if col not in ["Plate"] and not col.startswith("T°"):
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
        fig.update_layout(xaxis_title="Time (min)", yaxis_title="OD600", height=500, showlegend=False)
        fig_copy = copy.deepcopy(fig)
        st.plotly_chart(fig_copy, use_container_width=True, key=f"plot_{plate}")

    # Heatmaps of Mean and SD using matplotlib
    st.subheader("Well Summary Heatmaps (Mean and SD)")
    fig, axes = plt.subplots(2, len(all_summary), figsize=(5 * len(all_summary), 10))
    if len(all_summary) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i, summary in enumerate(all_summary):
        for j, metric in enumerate(["Mean", "SD"]):
            sub = summary[["Well", metric]]
            heatmap = pd.DataFrame(index=list("ABCDEFGH"), columns=range(1, 13), dtype=float)
            for _, row in sub.iterrows():
                match = re.match(r"([A-H])([1-9]|1[0-2])", row["Well"])
                if match:
                    r, c = match.groups()
                    heatmap.loc[r, int(c)] = row[metric]
            sns.heatmap(heatmap, ax=axes[j][i], cmap="rainbow", annot=False, fmt=".2f", cbar=True)
            axes[j][i].set_title(f"{summary['Plate'].iloc[0]} - {metric}")
    st.pyplot(fig)
