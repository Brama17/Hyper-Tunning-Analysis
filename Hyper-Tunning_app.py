import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# -------------------- KONFIGURASI HALAMAN -------------------- #
st.set_page_config(
    page_title="Portofolio Harwian",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📌"
)

# -------------------- FUNGSI BILINGUAL -------------------- #
def _(id_text, en_text):
    return id_text if st.session_state.language == 'id' else en_text

# -------------------- PILIH BAHASA -------------------- #
with st.sidebar:
    selected_language = st.selectbox("Pilih Bahasa / Select Language", ["id", "en"])
    st.session_state.language = selected_language

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

# -------------------- SIDEBAR -------------------- #
def sidebar_navigation():
    st.sidebar.title(_("\ud83d\udcc1 Navigasi", "\ud83d\udcc1 Navigation"))
    return st.sidebar.radio(_("Pilih Halaman / Select Page", "Choose Page"),
                            ["Tentang Saya", "Proyek", "Machine Learning", "Insight & Rekomendasi", "Kontak"])

# -------------------- TENTANG SAYA -------------------- #
def tentang_saya():
    st.markdown("## \ud83d\udccc " + _("Tentang Saya", "About Me"))
    st.markdown("### Full Stack Data Science")
    st.markdown(_(
        """
        Hai, saya **Harwian Brama Enggar Pramukti**, seorang *Full Stack Data Science* dengan pengalaman:
        - \ud83d\udcca Eksplorasi Data & Visualisasi menggunakan Python dan Power BI
        - \ud83e\uddf9 Data preprocessing: missing value, outlier, feature engineering
        - \ud83e\udde0 Membangun model ML untuk prediksi churn pelanggan (Telco Churn)
        - \ud83d\ude80 Deploy model dan dashboard interaktif menggunakan Streamlit
        - \ud83d\udd27 Tools: pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn
        """,
        """
        Hello, I'm **Harwian Brama Enggar Pramukti**, a *Full Stack Data Scientist* with experience in:
        - \ud83d\udcca Data exploration & visualization using Python and Power BI
        - \ud83e\uddf9 Data preprocessing: missing value, outlier, feature engineering
        - \ud83e\udde0 Building ML models for churn prediction (Telco Churn)
        - \ud83d\ude80 Deploying models and dashboards using Streamlit
        - \ud83d\udd27 Tools: pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn
        """
    ))

# -------------------- PROYEK -------------------- #
def proyek(df):
    st.title("\ud83d\udcc8 " + _("Proyek", "Projects"))
    st.markdown("#### Telco Customer Churn Analysis")

    st.sidebar.markdown("## \ud83d\udd0d " + _("Filter Data", "Filter Data"))
    min_tenure, max_tenure = st.sidebar.slider("Tenure", 0, int(df['Tenure'].max()), (0, 72))
    df_filtered = df[(df['Tenure'] >= min_tenure) & (df['Tenure'] <= max_tenure)]

    st.subheader(_("Distribusi Tenure", "Tenure Distribution"))
    fig1, ax1 = plt.subplots()
    sns.histplot(df_filtered['Tenure'], bins=30, kde=True, color='skyblue', ax=ax1)
    ax1.set_xlabel("Tenure (bulan)")
    st.pyplot(fig1)

    st.subheader(_("Proporsi Churn", "Churn Proportion"))
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_filtered, x='Churn', palette='Set2', ax=ax2)
    ax2.set_xticklabels(["Tidak Churn", "Churn"])
    st.pyplot(fig2)

    st.subheader("Monthly Charges vs Churn")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df_filtered, ax=ax3, palette='Set1')
    ax3.set_xticklabels(["Tidak Churn", "Churn"])
    st.pyplot(fig3)

# -------------------- MACHINE LEARNING -------------------- #
def ml(df):
    st.title("\ud83e\udd16 " + _("Model Machine Learning", "Machine Learning Model"))

    model_choice = st.selectbox("Pilih Model", ["Random Forest", "Logistic Regression"])
    features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    X_train, X_test, y_train, y_test = split_data(df, features)

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("\ud83d\udcca " + _("Evaluasi Model", "Model Evaluation"))
    st.write(f"**Akurasi:** {accuracy_score(y_test, y_pred):.2f}")
    st.code(classification_report(y_test, y_pred))

    st.subheader("\ud83d\udd22 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    if model_choice == "Random Forest":
        st.subheader("\ud83d\udd0d Feature Importance")
        importance = pd.Series(model.feature_importances_, index=features)
        fig, ax = plt.subplots()
        importance.sort_values().plot(kind='barh', color='orange', ax=ax)
        st.pyplot(fig)

    # ROC Curve
    st.subheader("\ud83d\udd3a ROC Curve")
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

# -------------------- INSIGHT & REKOMENDASI -------------------- #
def insight_rekomendasi():
    st.title("\ud83d\udcca " + _("Insight & Rekomendasi", "Insights & Recommendations"))
    st.subheader("\ud83d\udd0d " + _("Insight Penting", "Key Insights"))
    st.markdown(_("""
    - \ud83d\udcc9 Pelanggan dengan **Tenure pendek (< 10 bulan)** memiliki tingkat churn yang tinggi.
    - \ud83d\udcb0 Pelanggan dengan **biaya bulanan tinggi** lebih berisiko churn.
    - \ud83d\udce6 Pelanggan dengan **Total Charges rendah** sering kali adalah pelanggan baru.
    """,
    """
    - \ud83d\udcc9 Customers with **short Tenure (< 10 months)** have high churn rate.
    - \ud83d\udcb0 Customers with **high monthly charges** are more likely to churn.
    - \ud83d\udce6 Customers with **low Total Charges** are often new customers.
    """))

    st.subheader("\u2705 " + _("Rekomendasi Bisnis", "Business Recommendations"))
    st.markdown(_("""
    - \ud83c\udff1 Beri loyalty reward untuk pelanggan baru (< 6 bulan).
    - \ud83d\udce2 Tawarkan promo khusus untuk pelanggan dengan tagihan tinggi.
    - \ud83d\udcde Fokuskan kampanye retensi pada segmen rawan churn.
    """,
    """
    - \ud83c\udff1 Provide loyalty rewards to new customers (< 6 months).
    - \ud83d\udce2 Offer special promos to high-bill customers.
    - \ud83d\udcde Focus retention campaigns on high-risk churn segments.
    """))

# -------------------- KONTAK -------------------- #
def kontak():
    st.title("\ud83d\udcde " + _("Kontak", "Contact"))
    st.markdown("""
    - \ud83d\udce7 **Email**: [harwianbrama02@gmail.com](mailto:harwianbrama02@gmail.com)
    - \ud83d\udcbc **LinkedIn**: [Harwian Brama Enggar P](https://www.linkedin.com)
    - \ud83d\udcf1 **WhatsApp**: [082141922446](https://wa.me/6282141922446)
    - \ud83d\udcbb **GitHub**: [GitHub Profil](https://github.com/Brama17)
    """)

# -------------------- MAIN APP -------------------- #
df = load_data()
if df is not None:
    page = sidebar_navigation()
    if page == "Tentang Saya": tentang_saya()
    elif page == "Proyek": proyek(df)
    elif page == "Machine Learning": ml(df)
    elif page == "Insight & Rekomendasi": insight_rekomendasi()
    elif page == "Kontak": kontak()
else:
    st.error("Gagal memuat data.")
