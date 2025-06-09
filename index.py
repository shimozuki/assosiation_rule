import streamlit as st

st.set_page_config(page_title="Beranda", page_icon="🏠")

st.title("📊 Aplikasi Analisis Data Kominfo")
st.markdown("""
Selamat datang di aplikasi analisis data:

- 📋 **Data Mining**: Temukan asosiasi barang menggunakan algoritma Apriori atau FP-Growth.
- 🧠 **Text Mining**: Lakukan klasifikasi sentimen terhadap teks menggunakan SVM atau Naive Bayes.

Silakan pilih menu di sidebar untuk memulai analisis.
""")
