import streamlit as st
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download stopwords jika belum ada
try:
    stopwords_ind = set(stopwords.words('indonesian'))
except LookupError:
    nltk.download('stopwords')
    stopwords_ind = set(stopwords.words('indonesian'))

st.title("Text Mining: Klasifikasi Sentimen dan WordCloud")

# Pilih algoritma
algoritma = st.sidebar.selectbox("Pilih Algoritma:", ["SVM", "Naive Bayes"])

# Upload file
uploaded_file = st.file_uploader("Unggah file Excel dengan kolom teks:", type=["xlsx"])

# Fungsi preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_ind]
    return " ".join(tokens)

# Fungsi label otomatis
def label_sentimen(teks):
    kata_positif = [
        'baik', 'bagus', 'senang', 'puas', 'mantap', 'luar biasa', 'cepat', 'menyenangkan',
        'iqbal', 'dinda', 'bang zull', 'rohim', 'pirin', 'pilih', 'pilihan', 'terbaik',
    ]
    kata_negatif = [
        'buruk', 'jelek', 'lambat', 'mengecewakan', 'tidak puas', 'parah', 'benci',
        'tidak bagus', 'tidak baik', 'sangat buruk', 'kurang bagus', 'payah'
    ]
    teks = teks.lower()
    skor = 0
    for kata in kata_positif:
        if kata in teks:
            skor += 1
    for kata in kata_negatif:
        if kata in teks:
            skor -= 1
    if skor > 0:
        return "positif"
    elif skor < 0:
        return "negatif"
    else:
        return "netral"

# Fungsi download
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Data Awal:")
    st.write(df)

    text_col = st.selectbox("Pilih kolom Teks:", df.columns)
    df = df[[text_col]].dropna()
    df.columns = ["teks"]

    # Preprocessing dan label otomatis
    df["teks_bersih"] = df["teks"].apply(clean_text)
    df["label"] = df["teks_bersih"].apply(label_sentimen)

    st.subheader("Data Setelah Preprocessing dan Label Otomatis:")
    st.write(df)

    # Unduh data
    st.download_button(
        label="Unduh Data dengan Label",
        data=convert_df_to_excel(df),
        file_name="data_label_otomatis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Distribusi label
    st.subheader("Distribusi Label Sebelum Balancing:")
    label_counts = df["label"].value_counts()
    st.write(label_counts)
    st.bar_chart(label_counts)

    if len(label_counts) < 2:
        st.warning("Label terlalu sedikit atau tidak ada variasi. Pastikan data mencakup kata negatif dan positif.")
    else:
        # Balancing (undersampling)
        min_count = label_counts.min()
        df_balanced = df.groupby("label").apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

        st.subheader("Distribusi Label Setelah Balancing:")
        balanced_counts = df_balanced["label"].value_counts()
        st.write(balanced_counts)
        st.bar_chart(balanced_counts)

        # TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df_balanced["teks_bersih"])
        y = df_balanced["label"]

        st.subheader("Contoh Hasil TF-IDF:")
        tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        st.write(tfidf_df.head())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = MultinomialNB() if algoritma == "Naive Bayes" else LinearSVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi
        st.subheader("Evaluasi Model:")
        st.write(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # WordCloud global
        st.subheader("WordCloud Keseluruhan:")
        all_text = " ".join(df["teks_bersih"])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # # WordCloud per label
        # st.subheader("WordCloud per Label:")
        # for label in df["label"].unique():
        #     st.markdown(f"**Label: {label}**")
        #     text_per_label = " ".join(df[df["label"] == label]["teks_bersih"])
        #     wc = WordCloud(width=800, height=300, background_color='white').generate(text_per_label)
        #     fig, ax = plt.subplots()
        #     ax.imshow(wc, interpolation="bilinear")
        #     ax.axis("off")
        #     st.pyplot(fig)
