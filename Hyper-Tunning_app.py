# streamlit_churn_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

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
    url = "https://raw.githubusercontent.com/Harwian-Brama-DS/Hyper-Tunning-Analysis/main/churn.csv"
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

def evaluasi_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, class_report, cm, y_pred

# -------------------- TAMPILKAN SIDEBAR -------------------- #
def sidebar_navigation():
    with st.sidebar:
        st.title("📂 Navigasi / Navigation")
        language = st.selectbox("🌐 Pilih Bahasa / Choose Language", ["id", "en"])
        st.session_state.language = language
        if language == "id":
            return st.radio("Pilih Halaman", ["Tentang Saya", "Proyek", "Machine Learning", "Insight & Rekomendasi", "Kontak"])
        else:
            return st.radio("Select Page", ["About Me", "Project", "Machine Learning", "Insight & Recommendation", "Contact"])

# -------------------- TENTANG SAYA -------------------- #
def tentang_saya():
    st.markdown("## 📌 Tentang Saya / About Me")
    st.markdown("### Full Stack Data Science")
    st.markdown("""
    Hai, saya **Harwian Brama Enggar Pramukti**, seorang *Full Stack Data Science* dengan pengalaman:
    - 📊 Eksplorasi Data & Visualisasi menggunakan Python dan Power BI  
    - 🧹 Data preprocessing: missing value, outlier, feature engineering  
    - 🧠 Membangun model ML untuk prediksi churn pelanggan (Telco Churn)  
    - 🚀 Deploy model dan dashboard interaktif menggunakan Streamlit  
    - 🔧 Tools: pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn

    ---

    Hello, I’m **Harwian Brama Enggar Pramukti**, a *Full Stack Data Scientist* with experience:
    - 📊 Data Exploration & Visualization using Python and Power BI  
    - 🧹 Data preprocessing: missing value, outlier, feature engineering  
    - 🧠 Building ML models for customer churn prediction (Telco Churn)  
    - 🚀 Deploying interactive dashboards with Streamlit  
    - 🔧 Tools: pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn
    """)

# -------------------- PROYEK -------------------- #
def proyek(df):
    st.title("📈 Proyek / Project")
    st.markdown("#### Telco Customer Churn Analysis")

    st.sidebar.markdown("## 🔍 Filter Data")
    min_tenure, max_tenure = st.sidebar.slider("Rentang Tenure", 0, int(df['Tenure'].max()), (0, 72))
    df_filtered = df[(df['Tenure'] >= min_tenure) & (df['Tenure'] <= max_tenure)]

    st.subheader("Distribusi Tenure")
    fig1, ax1 = plt.subplots()
    sns.histplot(df_filtered['Tenure'], bins=30, kde=True, color='skyblue', ax=ax1)
    ax1.set_xlabel("Tenure (bulan)")
    st.pyplot(fig1)

    st.subheader("Proporsi Churn")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_filtered, x='Churn', palette='Set2', ax=ax2)
    ax2.set_xticklabels(['Tidak Churn', 'Churn'])
    st.pyplot(fig2)

    st.subheader("Monthly Charges vs Churn")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df_filtered, ax=ax3, palette='Set1')
    ax3.set_xticklabels(['Tidak Churn', 'Churn'])
    st.pyplot(fig3)

# -------------------- MACHINE LEARNING -------------------- #
def machine_learning(df):
    st.title("🤖 Model Machine Learning")
    st.markdown("Gunakan model untuk memprediksi pelanggan yang akan churn.")

    model_choice = st.selectbox("Pilih Model", ["Random Forest", "Logistic Regression"])
    features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    X_train, X_test, y_train, y_test = split_data(df, features)

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        alasan = "Random Forest dipilih karena mampu menangani data tidak linear dan memiliki performa baik dengan default parameter."
        deskripsi = "- Cocok untuk data non-linear\n- Tahan terhadap overfitting\n- Bisa menangani missing value & outlier"
    else:
        model = LogisticRegression(max_iter=1000)
        alasan = "Logistic Regression dipilih karena interpretabilitasnya tinggi dan cocok untuk binary classification."
        deskripsi = "- Sederhana dan interpretatif\n- Cocok jika hubungan linier\n- Cepat dan ringan"

    model.fit(X_train, y_train)
    acc, report, cm, y_pred = evaluasi_model(model, X_test, y_test)

    st.subheader("Evaluasi Model")
    st.markdown(f"### 🔎 Akurasi: `{acc:.2f}`")
    st.code(report)

    st.subheader("Confusion Matrix")
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
    st.pyplot(fig4)

    if model_choice == "Random Forest":
        st.subheader("Feature Importance")
        importance = pd.Series(model.feature_importances_, index=features)
        fig5, ax5 = plt.subplots()
        importance.sort_values().plot(kind='barh', color='orange', ax=ax5)
        st.pyplot(fig5)

    st.subheader("ROC Curve dan AUC")
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)

        fig6, ax6 = plt.subplots()
        ax6.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color='darkorange')
        ax6.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax6.set_xlabel("False Positive Rate")
        ax6.set_ylabel("True Positive Rate")
        ax6.set_title("ROC Curve")
        ax6.legend()
        st.pyplot(fig6)

    st.subheader("📌 Alasan & Penjelasan Model")
    st.markdown(alasan)
    st.info(deskripsi)

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
    - 💻 **GitHub**: [GitHub Profil](https://github.com/Harwian-Brama-DS)
    """)

# -------------------- MAIN APP -------------------- #
df = load_data()

if df is not None and not df.empty:
    page = sidebar_navigation()
    if "Tentang" in page or "About" in page:
        tentang_saya()
    elif "Proyek" in page or "Project" in page:
        proyek(df)
    elif "Machine Learning" in page:
        machine_learning(df)
    elif "Insight" in page:
        insight_rekomendasi()
    elif "Kontak" in page or "Contact" in page:
        kontak()
else:
    st.error("Gagal memuat data.")
