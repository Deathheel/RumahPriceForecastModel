import pandas as pd
import streamlit as st
import pickle
import sklearn

st.set_page_config(page_title="Dashboard", page_icon=":bar_chart:", layout="wide")


with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

    # Buat slider untuk setiap fitur
slider_lb = st.slider('Luas Bangunan:', value=100, min_value=40, max_value=500, step=10)
slider_lt = st.slider('Luas Tanah:',value=300, min_value=65, max_value=485, step=10)
slider_kt = st.slider('Kamar Tidur:',value=2, min_value=2, max_value=7, step=1)
slider_km = st.slider('Kamar Mandi:',value=1, min_value=1, max_value=5, step=1)
slider_grs = st.slider('Garasi:',value=1, min_value=1, max_value=4, step=1)
prediksi = clf.predict([[slider_lb, slider_lt, slider_kt, slider_km, slider_grs]])

path = "img/%d%d%d.png"%(slider_kt, slider_km, slider_grs)

col1, col2, col3 = st.columns([2,2,2])

with col1:
    st.write("")

with col2:
    # Buat tampilan interaktif
    st.write("Harga rumah impian anda diperkirakan sekitar IDR {:,.0f}".format(prediksi[0]))

    st.image(path, caption="Rumah dengan %d kamar tidur, %d kamar mandi, dan %d garasi. (Gambar hanya mockup yang dibuat oleh GEMINI AI)"%(slider_kt, slider_km, slider_grs), width=600)

    
with col3:
    st.write("")

