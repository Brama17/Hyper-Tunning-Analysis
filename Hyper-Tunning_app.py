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
    st.sidebar.title("📂 Navigasi / Navigation")
    language = st.sidebar.selectbox("Select Language / Pilih Bahasa", ["id", "en"])
    st.session_state.language = language
    if language == "id":
        return st.sidebar.radio("Pilih Halaman / Select Page", ["Tentang Saya", "Proyek", "Machine Learning", "Insight & Rekomendasi", "Kontak"])
    else:
        return st.sidebar.radio("Pilih Halaman / Select Page", ["About Me", "Project", "Machine Learning", "Insight & Recommendation", "Contact"])

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
    st.title("🤖 Model Machine Learning")
    st.markdown("Gunakan model untuk memprediksi pelanggan yang akan churn.")

    model_choice = st.selectbox("Pilih Model", ["Random Forest", "Logistic Regression"])
    features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    X_train, X_test, y_train, y_test = split_data(df, features)

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        alasan = "Random Forest dipilih karena mampu menangani data tidak linear dan memiliki performa baik dengan default parameter."
    else:
        model = LogisticRegression(max_iter=1000)
        alasan = "Logistic Regression dipilih karena interpretabilitasnya tinggi dan cocok untuk binary classification."

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

    # Feature Importance
    if model_choice == "Random Forest":
        st.subheader("Feature Importance")
        importance = pd.Series(model.feature_importances_, index=features)
        fig5, ax5 = plt.subplots()
        importance.sort_values().plot(kind='barh', color='orange', ax=ax5)
        st.pyplot(fig5)

    # ROC Curve dan AUC
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
    else:
        st.warning("Model ini tidak mendukung ROC Curve (predict_proba tidak tersedia).")

    # Penjelasan Model
    st.subheader("📌 Alasan Memilih Model")
    st.markdown(alasan)
    st.markdown("Model terbaik dipilih berdasarkan kombinasi akurasi dan nilai AUC. Silakan bandingkan hasil evaluasi untuk masing-masing model.")

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
    """)

# -------------------- MAIN APP -------------------- #
df = load_data()

if df is not None:
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
