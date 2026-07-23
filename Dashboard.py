import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
# --- Load the trained SGDRegressor model ---
try:
    # Assuming 'RumahBandung.pkl' is in the same directory as the Streamlit app
    with open('RumahBandung.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'RumahBandung.pkl' not found. Please ensure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- Define the RMSE for heuristic range (SGDRegressor) ---
# This value is hardcoded from the notebook's last execution of rmse_sgd
rmse_sgd_value = 890748835

# --- Feature Ranges (derived from HouseDF after preprocessing in the notebook) ---
# These values ensure the sliders reflect the data the model was trained on
bedroom_count_min, bedroom_count_max, bedroom_count_mean = 2.0, 5.0, 4.0
bathroom_count_min, bathroom_count_max, bathroom_count_mean = 1.0, 4.0, 2.6 # Adjusted mean for better slider default
carport_count_min, carport_count_max, carport_count_mean = 1.0, 2.0, 1.0
land_area_min, land_area_max, land_area_mean = 55.0, 463.5, 224.2 # Adjusted mean for better slider default
building_area_min, building_area_max, building_area_mean = 40.0, 300.0, 188.7 # Adjusted mean for better slider default

# --- Streamlit App Layout ---
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.png')
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Prediction Widget")
st.write("Adjust the sliders to predict the house price using the trained SGDRegressor model.")
st.info("This model is specifically trained for the Bojongloa Kidul, Bandung area.")

st.header("Input House Features")

# Create Streamlit sliders for each feature
# The order below matches the input order expected by the model:
# bedroom_count (kt), bathroom_count (km), carport_count (grs), land_area (lt), building_area (m2) (lb)
kt = st.slider("Jumlah Kamar Tidur (KT)",
               min_value=bedroom_count_min, max_value=bedroom_count_max,
               value=bedroom_count_min, step=1.0)
km = st.slider("Jumlah Kamar Mandi (KM)",
               min_value=bathroom_count_min, max_value=bathroom_count_max,
               value=bathroom_count_min, step=1.0)
gr = st.slider("Jumlah Carport (GRS)",
                min_value=carport_count_min, max_value=carport_count_max,
                value=carport_count_min, step=1.0)
lt = st.slider("Luas Tanah (LT) (m²)",
               min_value=land_area_min, max_value=land_area_max,
               value=land_area_min, step=10.0)
lb = st.slider("Luas Bangunan (LB) (m²)",
               min_value=building_area_min, max_value=building_area_max,
               value=building_area_min, step=10.0)
path = "img/%d%d%d.png"%(kt, km, gr)
if st.button("Predict Price"):
    # Prepare input features in the correct order for the model
    input_features = np.array([[kt, km, gr, lt, lb]])

    # Make prediction using the loaded model
    log_predicted_price = model.predict(input_features)
    predicted_price_point = np.expm1(log_predicted_price)[0]

    # Ensure the final displayed price is not negative
    final_predicted_price = max(0, predicted_price_point)

    # Calculate heuristic price range based on RMSE
    range_factor = 0.5 # Same factor used in the notebook's interactive widget
    lower_bound = max(0, final_predicted_price - (range_factor * rmse_sgd_value))
    upper_bound = final_predicted_price + (range_factor * rmse_sgd_value)

    st.subheader("Prediction Result:")
    st.success(f"*Predicted Price: IDR {final_predicted_price:,.0f}*")
    st.write(f"Heuristic Price Range (± {range_factor}*RMSE): [IDR {lower_bound:,.0f} - IDR {upper_bound:,.0f}]")
    st.info(f"Note: This range is an approximate measure of uncertainty based on the overall model's RMSE (IDR {rmse_sgd_value:,.0f}).")
    st.image(path, caption="Rumah dengan %d kamar tidur, %d kamar mandi, dan %d garasi. (Gambar hanya mockup yang dibuat oleh GEMINI AI)"%(kt, km, gr), width=700)




