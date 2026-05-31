import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IoT Vulnerability Detektor", layout="wide")
st.title("🛡️ IoT Vulnerability Detection System")
st.write("Aplikasi deteksi jaringan IoT. Masukkan file CSV yang mau di-scan.")

@st.cache_resource
def load_models():
    pipeline = joblib.load('pipeline_terbaik.pkl')
    le = joblib.load('label_encoder.pkl')
    return pipeline, le

pipeline, le = load_models()

uploaded_file = st.sidebar.file_uploader("Upload CSV Data (Fitur Saja)", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    
    kolom_contekan = ['Label', 'Attack_Category', 'Attack_sub_category']
    new_data_clean = new_data.drop(columns=[col for col in kolom_contekan if col in new_data.columns])
    
    st.subheader("Data yang mau diprediksi:")
    st.dataframe(new_data_clean.head())
    
    if st.button("Lakukan Prediksi"):
        try:
          
            predictions_encoded = pipeline.predict(new_data_clean)
            
    
            predictions_label = le.inverse_transform(predictions_encoded)
            
           
            new_data_clean['HASIL_DETEKSI'] = predictions_label
            
            st.success("Prediksi Berhasil! Ini hasilnya:")
            st.dataframe(new_data_clean)
            
        except Exception as e:
            st.error(f"Waduh ada error nih: {e}")
else:
    st.info("Upload file CSV-nya dulu di sebelah kiri ya!")