from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from io import BytesIO
import pandas as pd
import streamlit as st

# Judul aplikasi
st.title("Analisis Asosiasi Menggunakan Algoritma Apriori")

# Sidebar untuk memilih algoritma asosiasi
algoritma = st.sidebar.selectbox("Pilih Algoritma Asosiasi:", ["Apriori", "FP-Growth"])

# Upload file Excel
uploaded_file = st.file_uploader("Unggah file Excel dengan data Anda:", type=["xlsx"])

if uploaded_file:
    # Baca data dari file Excel yang diunggah
    df = pd.read_excel(uploaded_file)

    # Tampilkan data awal
    st.subheader("Data Awal:")
    st.write(df)

    # Memilih kolom untuk digunakan dalam analisis
    nomor_transaksi_col = st.selectbox('Pilih kolom untuk nomor transaksi:', df.columns)
    barang_col = st.selectbox('Pilih kolom untuk barang:', df.columns)
    qty_col = st.selectbox('Pilih kolom untuk jumlah barang (qty):', df.columns)

    # 1. Cleaning Data
    df_cleaned = df.dropna(subset=[nomor_transaksi_col, barang_col, qty_col])
    df_cleaned = df_cleaned.astype({nomor_transaksi_col: str, barang_col: str, qty_col: int})
    st.subheader("Data Setelah Cleaning:")
    st.write(df_cleaned)

    # 2. Memecah qty sesuai dengan jumlah barang
    expanded_data = []
    for _, row in df_cleaned.iterrows():
        for _ in range(row[qty_col]):
            expanded_data.append([row[nomor_transaksi_col], row[barang_col]])

    df_expanded = pd.DataFrame(expanded_data, columns=['nomor_transaksi', 'barang'])
    st.subheader("Data Setelah Pemecahan Barang Berdasarkan Qty:")
    st.write(df_expanded)

    # 3. Membuat transaksi dalam bentuk matrix
    basket = pd.pivot_table(df_expanded, index='nomor_transaksi', columns='barang', aggfunc='size', fill_value=0)
    basket[basket > 0] = 1
    st.subheader("Data Transaksi dalam Bentuk Matrix (1 dan 0):")
    st.write(basket)

    # 4. Menjalankan Algoritma Asosiasi
    if algoritma == "Apriori":
        frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
    elif algoritma == "FP-Growth":
        frequent_itemsets = fpgrowth(basket, min_support=0.0613, use_colnames=True)

    frequent_itemsets['count'] = frequent_itemsets['support'] * len(basket)
    frequent_itemsets['support_percentage'] = frequent_itemsets['support'] * 100

    st.subheader("Frequent Itemsets:")
    st.write(frequent_itemsets[['itemsets', 'count', 'support_percentage']])

    # Menampilkan aturan asosiasi
    min_confidence = st.slider("Pilih nilai minimum confidence (%):", 0, 100, 0) / 100
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))
    rules['confidence_percentage'] = rules['confidence'] * 100
    st.subheader("Association Rules:")
    st.write(rules[['antecedents', 'consequents', 'support', 'confidence_percentage', 'lift']])

    # Komparasi antara 2-itemset dan 3-itemset
    two_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 2]
    three_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 3]

    st.subheader("Komparasi 2-Itemset dan 3-Itemset:")
    col1, col2 = st.columns(2)

    with col1:
        st.write("2-Itemset:")
        st.write(two_itemsets[['itemsets', 'count', 'support_percentage']])

    with col2:
        st.write("3-Itemset:")
        st.write(three_itemsets[['itemsets', 'count', 'support_percentage']])

    # Unduh hasil
    def convert_df_to_excel(dataframe):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            dataframe.to_excel(writer, index=False, sheet_name="Frequent_Itemsets")
        return output.getvalue()

    # Unduh hasil 2-itemset
    if not two_itemsets.empty:
        excel_data_2_itemsets = convert_df_to_excel(two_itemsets[['itemsets', 'count', 'support_percentage']])
    st.download_button(
        label="Unduh Hasil 2-Itemset",
        data=excel_data_2_itemsets,
        file_name="hasil_2_itemset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
