import streamlit as st
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from collections import Counter
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit_shadcn_ui as ui
from local_components import card_container

# Download stopwords jika belum
try:
    stopwords_ind = set(stopwords.words('indonesian'))
except LookupError:
    nltk.download('stopwords')
    stopwords_ind = set(stopwords.words('indonesian'))

st.set_page_config(page_title="Text Mining", layout="wide")
st.title("Text Mining: Klasifikasi Sentimen & Visualisasi Tokoh Politik")

algoritma = st.sidebar.selectbox("Pilih Algoritma:", ["SVM", "Naive Bayes"])
uploaded_file = st.file_uploader("Unggah file Excel dengan kolom teks:", type=["xlsx"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_ind]
    return " ".join(tokens)

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

    df["teks_bersih"] = df["teks"].apply(clean_text)
    df["label"] = df["teks_bersih"].apply(label_sentimen)

    st.subheader("Data Setelah Preprocessing dan Labeling:")
    st.write(df)

    st.download_button(
        label="Unduh Data dengan Label",
        data=convert_df_to_excel(df),
        file_name="data_label_otomatis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.subheader("Grafik Sentimen Masyarakat")
    with card_container(key="Pie"):
        st.subheader("Grafik Sentimen Masyarakat")
        sentimen_counts = df["label"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(13, 5))
        ax1.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

    # 📊 Sentimen per Tokoh
    st.subheader("Grafik Sentimen Terhadap Tokoh Politik")
    tokoh = ['rohim', 'firin', 'zul', 'uhel', 'iqbal', 'dinda']
    tokoh_sentimen = {t: {"positif": 0, "negatif": 0, "netral": 0} for t in tokoh}

    for i, row in df.iterrows():
        teks = row["teks_bersih"]
        label = row["label"]
        for t in tokoh:
            if t in teks:
                tokoh_sentimen[t][label] += 1

    df_tokoh = pd.DataFrame(tokoh_sentimen).T
    st.bar_chart(df_tokoh)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Label Sebelum Balancing:")
        with card_container(key="Distribusi Label"):
    # 📊 Distribusi label
            st.subheader("Distribusi Label Sebelum Balancing:")
    # st.write(sentimen_counts)
            st.bar_chart(sentimen_counts)

    if len(sentimen_counts) < 2:
        st.warning("Label terlalu sedikit atau tidak ada variasi.")
    else:
        min_count = sentimen_counts.min()
        df_balanced = df.groupby("label").apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
        with col2:
            st.subheader("Distribusi Label Setelah Balancing:")
            with card_container(key="Balancing"):
                st.subheader("Distribusi Label Setelah Balancing:")
                # st.write(df_balanced["label"].value_counts())
                st.bar_chart(df_balanced["label"].value_counts())

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df_balanced["teks_bersih"])
        y = df_balanced["label"]

        st.subheader("Proses TF-IDF:")
        tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        st.write(tfidf_df.head())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = MultinomialNB() if algoritma == "Naive Bayes" else LinearSVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Evaluasi Model:")

        # Hitung metrik utama
        akurasi = accuracy_score(y_test, y_pred)
        presisi = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Tampilkan dengan 3 kolom
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Akurasi", f"{akurasi:.2f}")
        with col2:
            st.metric("Presisi", f"{presisi:.2f}")
        with col3:
            st.metric("Recall", f"{recall:.2f}")
        with col4:
            st.metric("F1-Score", f"{f1:.2f}")

        # st.text("Classification Report:")
        # st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix:")
            with card_container(key="Confusion Matrix"):
                st.subheader("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)
        with col2:
            st.subheader("Grafik Penyebaran Data (Label Aktual vs Prediksi)")
            with card_container(key="Data Distribution"):
                st.subheader("Grafik Penyebaran Data (Label Aktual vs Prediksi)")
                df_dist = pd.DataFrame({
                    "Label Aktual": y_test,
                    "Label Prediksi": y_pred
                })

            # Plot side-by-side bar (countplot aktual & prediksi)
                fig_dist, ax_dist = plt.subplots(figsize=(4,3))
            # Bar untuk label aktual
                sns.countplot(x="Label Aktual", data=df_dist, ax=ax_dist, palette="crest", alpha=0.7, label="Aktual")
            # Bar untuk label prediksi di atasnya (warna lain)
                sns.countplot(x="Label Prediksi", data=df_dist, ax=ax_dist, palette="pastel", alpha=0.5, label="Prediksi")

                ax_dist.legend(["Aktual", "Prediksi"])
                ax_dist.set_title("Sebaran Label Aktual vs Prediksi")
                st.pyplot(fig_dist)

        # WordCloud Global
        st.subheader("WordCloud Keseluruhan:")
        with card_container(key="WordCloud"):
            st.subheader("WordCloud Keseluruhan:")
            all_text = " ".join(df["teks_bersih"])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

        # # WordCloud per Label
        # st.subheader("WordCloud per Label:")
        # for label in df["label"].unique():
        #     st.markdown(f"**Label: {label}**")
        #     text_per_label = " ".join(df[df["label"] == label]["teks_bersih"])
        #     wc = WordCloud(width=800, height=300, background_color='white').generate(text_per_label)
        #     fig, ax = plt.subplots()
        #     ax.imshow(wc, interpolation="bilinear")
        #     ax.axis("off")
        #     st.pyplot(fig)
