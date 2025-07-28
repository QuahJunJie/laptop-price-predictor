
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Setup ---
st.set_page_config(
    page_title="Laptop Price Predictor 💻",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS & Animation ---
def set_custom_css():
    st.markdown("""
        <style>
        #MainMenu, header, footer {visibility: hidden;}
        .stButton>button {
            background-color: #00aaaa;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1.5em;
            font-weight: bold;
            transition: 0.3s;
            animation: pulse 2s infinite;
        }
        .stButton>button:hover {
            background-color: #008888;
        }
        label {font-size: 0.9rem; color: #cccccc;}
        .stSlider {margin-top: -10px;}
        h1, h2, h3 {color: #00dddd; text-align: center;}
        .stAlert {
            background-color: #e6fff2;
            border-left: 5px solid #00cc99;
        }
        @keyframes pulse {
            0% {box-shadow: 0 0 0 0 rgba(0,255,255, 0.4);}
            70% {box-shadow: 0 0 0 10px rgba(0,255,255, 0);}
            100% {box-shadow: 0 0 0 0 rgba(0,255,255, 0);}
        }
        @media only screen and (max-width: 768px) {
            .stButton>button, .stSelectbox, .stSlider {
                width: 100% !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)

set_custom_css()

# --- Load Model & Scaler ---
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Title & Subtitle ---
st.title("💻 Laptop Price Predictor")
st.caption("Estimate your laptop’s SGD value instantly, based on specifications.")

# --- Input Encodings ---
core_map = {
    "2-Core Entry": 0, "4-Core Entry": 1, "6-Core Mainstream": 2,
    "8-Core Performance": 3, "8-Core Hybrid (4P+4E)": 4,
    "10-Core High-Perf": 5, "12-14 Core Power": 6, "16+ Core Extreme": 7
}
ram_map = {
    "4 GB DDR4": 0, "8 GB DDR4": 1, "16 GB DDR4": 2,
    "8 GB LPDDR5": 3, "16 GB LPDDR5": 4, "Others": 5
}
ssd_map = {
    "128 GB": 0, "256 GB": 1, "512 GB": 2, "1 TB": 3, "2 TB": 4, "Others": 5
}
display_map = {"FHD": 0, "Touch": 1, "OLED": 2, "Others": 3}
graphics_map = {"Intel": 0, "NVIDIA": 1, "AMD": 2, "Integrated": 3, "Others": 4}
os_map = {"Windows 10": 0, "Windows 11": 1, "Linux": 2, "Mac": 3, "Others": 4}

# --- Initialize History ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- Input Section ---
with st.container():
    st.markdown("<h3 style='text-align:center;'>🔧 Choose Your Laptop Specs</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        core = st.selectbox("Core", list(core_map.keys()), index=3)
        ram = st.selectbox("RAM", list(ram_map.keys()), index=2)
        ssd = st.selectbox("SSD", list(ssd_map.keys()), index=2)
        os = st.selectbox("Operating System", list(os_map.keys()), index=1)
    with col2:
        display = st.selectbox("Display", list(display_map.keys()), index=0)
        graphics = st.selectbox("Graphics", list(graphics_map.keys()), index=1)
        generation = st.selectbox("CPU Generation", list(range(0, 14)), index=11)
        warranty_int = st.selectbox("Warranty (Years)", [0, 1, 2, 3], index=1)
    rating = st.slider("⭐ Rating", 1.0, 5.0, value=4.2, step=0.1)
    rating_scaled = (rating - 1.0) / 4.0

# --- Predict ---
if st.button("🔮 Estimate My Laptop Value"):
    input_df = pd.DataFrame([[
        rating_scaled,
        generation,
        core_map[core],
        ram_map[ram],
        ssd_map[ssd],
        display_map[display],
        graphics_map[graphics],
        os_map[os],
        warranty_int
    ]], columns=['Rating_scaled', 'Generation', 'Core', 'Ram', 'SSD',
                 'Display', 'Graphics', 'OS', 'warranty_int'])

    try:
        norm_price = model.predict(input_df)[0]
        sgd_price = scaler.inverse_transform([[norm_price]])[0][0]
        st.markdown(f"""
<div style='
    background: rgba(0, 255, 204, 0.1);
    backdrop-filter: blur(10px);
    padding: 1.5em;
    margin-top: 1em;
    border-radius: 16px;
    border: 1px solid #00ffee;
    color: #ffffff;
    font-size: 18px;
    text-align: center;'>
💰 <b>Estimated Price:</b><br>
<span style='font-size:30px;font-weight:bold;color:#00ffee;'>${sgd_price:,.2f}</span>
</div>
""", unsafe_allow_html=True)

        st.session_state['history'].append((input_df.iloc[0].to_dict(), round(sgd_price, 2)))

        if sgd_price < 800:
            remark = "Entry-level laptop 💼"
        elif sgd_price < 2000:
            remark = "Mid-range performance 💪"
        else:
            remark = "High-end machine 🚀"
        st.markdown(f"<div style='color:#cccccc;text-align:center'>{remark}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- History Viewer ---
if st.checkbox("📜 Show past predictions"):
    st.write(pd.DataFrame([
        {**features, "Predicted SGD": price}
        for features, price in st.session_state['history']
    ]))

# --- Extra Notes ---
with st.expander("📌 What affects the price most?"):
    st.markdown("""
- CPU Core & Generation
- SSD Size
- RAM Type
- Display Type
- Rating (after scaling)
""")
