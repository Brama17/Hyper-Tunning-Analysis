# streamlit_churn_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------- CONFIGURASI HALAMAN -------------------- #
st.set_page_config(
    page_title="Portofolio Harwian",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📌"
)

# -------------------- FUNGSI UTILITAS -------------------- #
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Brama17/Hyper-Tunning-Analysis/main/churn.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengunduh data: {e}")
        return None

    df.dropna(inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

@st.cache_data
def split_data(df, features):
    X = df[features]
    y = df['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- TAMPILKAN SIDEBAR -------------------- #
def sidebar_navigation():
    st.sidebar.title("📂 Navigasi")
    return st.sidebar.radio("Pilih Halaman", ["Tentang Saya", "Proyek", "Machine Learning", "Insight & Rekomendasi", "Kontak"])

# -------------------- TENTANG SAYA -------------------- #
def tentang_saya():
    st.markdown("## 📌 Tentang Saya")
    st.markdown("### Full Stack Data Science")
    st.markdown("""
    Hai, saya **Harwian Brama Enggar Pramukti**, seorang *Full Stack Data Science* dengan pengalaman:
    - 📊 Eksplorasi Data & Visualisasi menggunakan Python dan Power BI  
    - 🧹 Data preprocessing: missing value, outlier, feature engineering  
    - 🧠 Membangun model ML untuk prediksi churn pelanggan (Telco Churn)  
    - 🚀 Deploy model dan dashboard interaktif menggunakan Streamlit  
    - 🔧 Tools: pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn
    """)

# -------------------- PROYEK -------------------- #
def proyek(df):
    st.title("📈 Proyek")
    st.markdown("#### Telco Customer Churn Analysis")

    # Filter interaktif
    st.sidebar.markdown("## 🔍 Filter Data")
    min_tenure, max_tenure = st.sidebar.slider("Rentang Tenure", 0, int(df['Tenure'].max()), (0, 72))
    df_filtered = df[(df['Tenure'] >= min_tenure) & (df['Tenure'] <= max_tenure)]

    # Distribusi Tenure
    st.subheader("Distribusi Tenure")
    fig1, ax1 = plt.subplots()
    sns.histplot(df_filtered['Tenure'], bins=30, kde=True, color='skyblue', ax=ax1)
    ax1.set_xlabel("Tenure (bulan)")
    st.pyplot(fig1)

    # Churn Count
    st.subheader("Proporsi Churn")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_filtered, x='Churn', palette='Set2', ax=ax2)
    ax2.set_xticklabels(['Tidak Churn', 'Churn'])
    st.pyplot(fig2)

    # Monthly Charges vs Churn
    st.subheader("Monthly Charges vs Churn")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df_filtered, ax=ax3, palette='Set1')
    ax3.set_xticklabels(['Tidak Churn', 'Churn'])
    st.pyplot(fig3)

# -------------------- MACHINE LEARNING -------------------- #
def machine_learning(df):
    st.title("🤖 Machine Learning: Prediksi Churn")
    st.markdown("Gunakan model untuk memprediksi pelanggan yang akan churn.")

    model_choice = st.selectbox("Pilih Model", ["Random Forest", "Logistic Regression"])
    features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    X_train, X_test, y_train, y_test = split_data(df, features)

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi Model
    st.subheader("Evaluasi Model")
    st.write(f"**Akurasi:** {accuracy_score(y_test, y_pred):.2f}")
    st.code(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig4, ax4 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax4)
    st.pyplot(fig4)

    # Feature Importance (hanya untuk Random Forest)
    if model_choice == "Random Forest":
        st.subheader("Feature Importance")
        importance = pd.Series(model.feature_importances_, index=features)
        fig5, ax5 = plt.subplots()
        importance.sort_values().plot(kind='barh', color='orange', ax=ax5)
        st.pyplot(fig5)

# -------------------- INSIGHT & REKOMENDASI -------------------- #
def insight_rekomendasi():
    st.title("📊 Insight & Rekomendasi")
    st.subheader("🔍 Insight Penting")
    st.markdown("""
    - 📉 Pelanggan dengan **Tenure pendek (< 10 bulan)** memiliki tingkat churn yang tinggi.
    - 💰 Pelanggan dengan **biaya bulanan tinggi** lebih berisiko churn.
    - 📦 Pelanggan dengan **Total Charges rendah** sering kali adalah pelanggan baru.
    """)

    st.subheader("✅ Rekomendasi Bisnis")
    st.markdown("""
    - 🎁 Beri loyalty reward untuk pelanggan baru (< 6 bulan).
    - 📢 Tawarkan promo khusus untuk pelanggan dengan tagihan tinggi.
    - 📞 Fokuskan kampanye retensi pada segmen rawan churn.
    """)

# -------------------- KONTAK -------------------- #
def kontak():
    st.title("📞 Kontak")
    st.markdown("Silakan hubungi saya melalui informasi berikut:")
    st.markdown("""
    - 📧 **Email**: [harwianbrama02@gmail.com](mailto:harwianbrama02@gmail.com)
    - 💼 **LinkedIn**: [Harwian Brama Enggar P](https://www.linkedin.com)
    - 📱 **WhatsApp**: [082141922446](https://wa.me/6282141922446)
    - 💻 **GitHub**: [GitHub Profil](https://github.com/Brama17)
    """)

# -------------------- MAIN APP -------------------- #
df = load_data()

if df is not None:
    halaman = sidebar_navigation()
    if halaman == "Tentang Saya":
        tentang_saya()
    elif halaman == "Proyek":
        proyek(df)
    elif halaman == "Machine Learning":
        machine_learning(df)
    elif halaman == "Insight & Rekomendasi":
        insight_rekomendasi()
    elif halaman == "Kontak":
        kontak()
else:
    st.error("Gagal memuat data.")
