import streamlit as st
import subprocess
import sys

st.title("Pilih Jenis Analisis")

option = st.selectbox("Pilih jenis analisis:", ["Data Mining", "Text Mining"])

if option == "Data Mining":
    st.info("Menjalankan Analisis Asosiasi (Data Mining)")
    subprocess.run([sys.executable, "index.py"])

elif option == "Text Mining":
    st.info("Menjalankan Analisis Teks (Text Mining)")
    subprocess.run([sys.executable, "text_mining.py"])
