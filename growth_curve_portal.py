# growth_curve_portal.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from io import StringIO

st.title("Growth Curve Visualisation Portal")

uploaded_files = st.file_uploader("Upload LogPhase600 .txt files", type="txt", accept_multiple_files=True)

def parse_growth_file(file):
    content = file.getvalue().decode('ISO-8859-1')
    lines = content.splitlines()

    # Find header
    header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("Time"))
    headers = lines[header_idx].split("\t")

    # Parse data rows
    data_rows = [
        row.split("\t") for row in lines[header_idx + 1:]
        if re.match(r'^\d+:\d+:\d+', row)
        and len(row.split("\t")) == len(headers)
    ]

    df = pd.DataFrame(data_rows, columns=headers)
    df["Time"] = df["Time"].apply(lambda t: sum(int(x) * 60**i for i, x in enumerate(reversed(t.split(":")))))
    for col in df.columns:
        if col != "Time":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("Time")

if uploaded_files:
    for file in uploaded_files:
        df = parse_growth_file(file)
        st.subheader(f"Plot for: {file.name}")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df.drop(columns=[col for col in df.columns if "T°" in col or col.startswith("T°")]), ax=ax)
        ax.set_title("OD600 over Time")
        ax.set_ylabel("OD600")
        ax.set_xlabel("Time (minutes)")
        st.pyplot(fig)
