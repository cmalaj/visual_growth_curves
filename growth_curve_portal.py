
# Streamlit app to visualize growth curves, now with plate layout autofill from metadata file
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from matplotlib import colors as mcolors

st.set_page_config(layout="wide")
st.title("Growth Curve Portal with Metadata Layout Import")

# === Helper functions ===
def parse_metadata_layout(uploaded_file):
    try:
        df_layout = pd.read_csv(uploaded_file, sep="\t", header=None)
        df_layout.columns = ["Row"] + [str(i+1) for i in range(12)]
        df_layout.set_index("Row", inplace=True)
        layout_dict = {}
        for row in df_layout.index:
            for col in df_layout.columns:
                well = f"{row}{col}"
                layout_dict[well] = str(df_layout.loc[row, col])
        return layout_dict
    except Exception as e:
        st.error(f"Error parsing metadata file: {e}")
        return {}

def apply_layout_to_data(df, layout_dict, plate_id):
    for col in df.columns:
        if re.match(r"^[A-H]\d{1,2}$", col):
            label_key = f"{plate_id}_{col}_label"
            default_label = layout_dict.get(col, col)
            st.session_state[label_key] = default_label
    return layout_dict

# === Load and parse uploaded metadata file ===
uploaded_metadata = st.file_uploader("Upload plate metadata file", type=["txt", "tsv"])
preloaded_layouts = {}
if uploaded_metadata:
    preloaded_layouts = parse_metadata_layout(uploaded_metadata)
    st.success("Metadata file loaded. Layout will be auto-filled.")

# === Simulate CSV file loading for demo (replace this with actual file upload and parsing) ===
@st.cache_data
def load_example_csv():
    time = np.arange(0, 180, 15)
    data = {
        "Time": time,
        "A1": np.random.normal(loc=1, scale=0.05, size=len(time)),
        "A2": np.random.normal(loc=0.8, scale=0.03, size=len(time)),
        "B1": np.random.normal(loc=0.5, scale=0.07, size=len(time)),
        "B2": np.random.normal(loc=0.3, scale=0.02, size=len(time)),
    }
    df = pd.DataFrame(data)
    df.set_index("Time", inplace=True)
    df["Plate"] = "Plate1"
    return [df]

all_data = load_example_csv()

# === Display and allow layout editing ===
st.header("Custom Plate Layout")
all_layouts = {}

for df in all_data:
    plate = df["Plate"].iloc[0]
    st.subheader(f"Plate: {plate}")
    layout_map = {}

    cols = st.columns(12)
    for row in "ABCDEFGH":
        with st.container():
            st.markdown(f"**Row {row}**")
            for i, col_num in enumerate(range(1, 13)):
                well = f"{row}{col_num}"
                key = f"{plate}_{well}_label"
                default_label = preloaded_layouts.get(well, well)
                layout_map[well] = st.text_input(f"Label for {well}", value=default_label, key=key)

    all_layouts[plate] = {"well_map": layout_map}

# === Placeholder for plotting logic ===
st.markdown("---")
st.header("Growth Curve Plot (placeholder)")
st.info("Plotting logic would be implemented below using all_data and all_layouts")
