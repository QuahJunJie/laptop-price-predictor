import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Laptop Price Predictor 💻", page_icon="💰")
st.title("💻 Laptop Price Predictor")
st.write("Enter laptop specifications to predict the price (INR & SGD).")

# --- Load Model & Scaler ---
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')  # You must upload this too
    st.success("✅ Model & Scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model or scaler. Error: {e}")
    st.stop()

# --- Input Fields ---
generation = st.selectbox("CPU Generation", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

core = st.selectbox("Core Category", [
    "2-Core Entry", "4-Core Entry", "6-Core Mainstream",
    "8-Core Performance", "8-Core Hybrid (4P+4E)", 
    "10-Core High-Perf", "12-14 Core Power", "16+ Core Extreme"
])
core_map = {
    "2-Core Entry": 0, "4-Core Entry": 1, "6-Core Mainstream": 2,
    "8-Core Performance": 3, "8-Core Hybrid (4P+4E)": 4,
    "10-Core High-Perf": 5, "12-14 Core Power": 6, "16+ Core Extreme": 7
}

ram = st.selectbox("RAM", ["4 GB DDR4", "8 GB DDR4", "16 GB DDR4", "8 GB LPDDR5", "16 GB LPDDR5", "Others"])
ram_map = {
    "4 GB DDR4": 0, "8 GB DDR4": 1, "16 GB DDR4": 2, 
    "8 GB LPDDR5": 3, "16 GB LPDDR5": 4, "Others": 5
}

ssd = st.selectbox("SSD", ["128 GB", "256 GB", "512 GB", "1 TB", "2 TB", "Others"])
ssd_map = {
    "128 GB": 0, "256 GB": 1, "512 GB": 2, "1 TB": 3, "2 TB": 4, "Others": 5
}

display = st.selectbox("Display Type", ["FHD", "Touch", "OLED", "Others"])
display_map = {"FHD": 0, "Touch": 1, "OLED": 2, "Others": 3}

graphics = st.selectbox("Graphics", ["Intel", "NVIDIA", "AMD", "Integrated", "Others"])
graphics_map = {"Intel": 0, "NVIDIA": 1, "AMD": 2, "Integrated": 3, "Others": 4}

os = st.selectbox("Operating System", ["Windows 10", "Windows 11", "Linux", "Mac", "Others"])
os_map = {"Windows 10": 0, "Windows 11": 1, "Linux": 2, "Mac": 3, "Others": 4}

warranty = st.selectbox("Warranty Length", ["0", "1", "2", "3"])
warranty_int = int(warranty)

rating = st.slider("Product Rating", min_value=1.0, max_value=5.0, step=0.1)

# --- Predict ---
if st.button("Predict Price"):
    input_data = pd.DataFrame([[
        generation,
        core_map[core],
        ram_map[ram],
        ssd_map[ssd],
        display_map[display],
        graphics_map[graphics],
        os_map[os],
        warranty_int,
        rating
    ]], columns=[
        'Generation', 'Core', 'Ram', 'SSD',
        'Display', 'Graphics', 'OS', 'warranty_int', 'Rating'
    ])
    
    # Predict (normalized)
    predicted = model.predict(input_data)[0]
    
    # Denormalize (get actual price in INR)
    actual_price_inr = scaler.inverse_transform([[predicted]])[0][0]

    # Convert INR to SGD (₹1 ≈ 0.016 SGD, approx)
    price_sgd = actual_price_inr * 0.016

    # Display both
    st.success(f"💸 Predicted Price: ₹{actual_price_inr:,.2f}  (~SGD ${price_sgd:,.2f})")
    st.caption("Note: This is an approximate conversion. Exchange rates may vary.")
