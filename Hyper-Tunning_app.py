# bilingual_churn_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# -------------------- CONFIGURASI HALAMAN -------------------- #
st.set_page_config(
    page_title="Portofolio Harwian",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📌"
)

# -------------------- PILIHAN BAHASA -------------------- #
language = st.sidebar.selectbox("Pilih Bahasa / Select Language", ["id", "en"])

def _(id_text, en_text):
    return id_text if language == "id" else en_text

# -------------------- FUNGSI UTILITAS -------------------- #
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Brama17/Hyper-Tunning-Analysis/main/churn.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"{_('Terjadi kesalahan saat mengunduh data:', 'Error loading dataset:')} {e}")
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
    st.sidebar.title("📂 Navigasi / Navigation")
    return st.sidebar.radio("Pilih Halaman / Select Page", [
        _("Tentang Saya", "About Me"),
        _("Proyek", "Project"),
        _("Machine Learning", "Machine Learning"),
        _("Insight & Rekomendasi", "Insight & Recommendation"),
        _("Kontak", "Contact")
    ])

# -------------------- TENTANG SAYA -------------------- #
def tentang_saya():
    st.markdown(f"## ✪ {_('Tentang Saya', 'About Me')}")
    st.markdown(f"### {_('Full Stack Data Science', 'Full Stack Data Science')}")
    st.markdown(_(
        """
        Hai, saya **Harwian Brama Enggar Pramukti**, seorang *Full Stack Data Science* dengan pengalaman:
        - 📊 Eksplorasi Data & Visualisasi menggunakan Python dan Power BI  
        - 🧹 Data preprocessing: missing value, outlier, feature engineering  
        - 🧠 Membangun model ML untuk prediksi churn pelanggan (Telco Churn)  
        - 🚀 Deploy model dan dashboard interaktif menggunakan Streamlit  
        - 🔧 Tools: pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn
        """,
        """
        Hi, I'm **Harwian Brama Enggar Pramukti**, a *Full Stack Data Scientist* with experience in:
        - 📊 Data Exploration & Visualization using Python and Power BI  
        - 🧹 Data Preprocessing: missing value, outlier, feature engineering  
        - 🧠 Building ML models to predict customer churn (Telco Churn)  
        - 🚀 Deploying interactive models and dashboards using Streamlit  
        - 🔧 Tools: pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn
        """
    ))

# -------------------- PROYEK -------------------- #
def proyek(df):
    st.title(f"📈 {_('Proyek Analisis Churn', 'Churn Analysis Project')}")

    st.sidebar.markdown(f"### ⌖ {_('Filter Data', 'Filter Data')}")
    min_tenure, max_tenure = st.sidebar.slider("Tenure", 0, int(df['Tenure'].max()), (0, 72))
    df_filtered = df[(df['Tenure'] >= min_tenure) & (df['Tenure'] <= max_tenure)]

    st.subheader(_("Distribusi Tenure", "Tenure Distribution"))
    fig1, ax1 = plt.subplots()
    sns.histplot(df_filtered['Tenure'], bins=30, kde=True, color='skyblue', ax=ax1)
    st.pyplot(fig1)

    st.subheader(_("Proporsi Churn", "Churn Proportion"))
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_filtered, x='Churn', palette='Set2', ax=ax2)
    ax2.set_xticklabels([_("Tidak Churn", "Not Churn"), "Churn"])
    st.pyplot(fig2)

    st.subheader(_("Monthly Charges vs Churn", "Monthly Charges vs Churn"))
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df_filtered, ax=ax3, palette='Set1')
    st.pyplot(fig3)

    st.subheader(_("Korelasi Fitur", "Feature Correlation"))
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(df_filtered[['Tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

# -------------------- MACHINE LEARNING -------------------- #
def ml(df):
    st.title("🤖 " + _("Model Machine Learning", "Machine Learning Model"))

    model_choice = st.selectbox(_("Pilih Model", "Select Model"), ["Random Forest", "Logistic Regression"])
    features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    X_train, X_test, y_train, y_test = split_data(df, features)

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.subheader(_("Evaluasi Model", "Model Evaluation"))
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.code(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    fig4, ax4 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax4)
    st.pyplot(fig4)

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    if model_choice == "Random Forest":
        st.subheader("Feature Importance")
        importance = pd.Series(model.feature_importances_, index=features)
        fig5, ax5 = plt.subplots()
        importance.sort_values().plot(kind='barh', color='orange', ax=ax5)
        st.pyplot(fig5)

    st.info(_(
        """
        Random Forest lebih baik jika ingin menangkap pola kompleks, Logistic Regression cocok jika kamu butuh interpretasi dan kecepatan.
        """,
        """
        Random Forest is better for capturing complex patterns, while Logistic Regression is good for interpretability and speed.
        """
    ))

# -------------------- INSIGHT & REKOMENDASI -------------------- #
def insight():
    st.title("📊 " + _("Insight & Rekomendasi", "Insight & Recommendation"))
    st.subheader("🔍 " + _("Insight Penting", "Key Insight"))
    st.markdown(_(
        """
        - Pelanggan dengan **Tenure pendek (< 10 bulan)** cenderung lebih sering churn.
        - Pelanggan dengan **Monthly Charges tinggi** lebih berisiko churn.
        - **TotalCharges rendah** sering kali berasal dari pelanggan baru.
        """,
        """
        - Customers with **short tenure (< 10 months)** are more likely to churn.
        - Customers with **high monthly charges** are at higher risk.
        - **Low total charges** often indicate new customers.
        """
    ))

    st.subheader("✅ " + _("Rekomendasi Bisnis", "Business Recommendations"))
    st.markdown(_(
        """
        - Tawarkan insentif loyalitas untuk pelanggan baru.
        - Buat promo bundling untuk pengguna dengan biaya tinggi.
        - Fokuskan kampanye retensi pada segmen pelanggan dengan risiko tinggi.
        """,
        """
        - Offer loyalty incentives to new customers.
        - Provide bundling offers to users with high spending.
        - Focus retention campaigns on high-risk customer segments.
        """
    ))

# -------------------- KONTAK -------------------- #
def kontak():
    st.title("📞 " + _("Kontak", "Contact"))
    st.markdown(_(
        """
        - 📧 **Email**: harwianbrama02@gmail.com  
        - 💼 **LinkedIn**: [Harwian Brama Enggar P](https://www.linkedin.com)  
        - 📱 **WhatsApp**: [082141922446](https://wa.me/6282141922446)  
        - 💻 **GitHub**: [Brama17](https://github.com/Brama17)
        """,
        """
        - 📧 **Email**: harwianbrama02@gmail.com  
        - 💼 **LinkedIn**: [Harwian Brama Enggar P](https://www.linkedin.com)  
        - 📱 **WhatsApp**: [082141922446](https://wa.me/6282141922446)  
        - 💻 **GitHub**: [Brama17](https://github.com/Brama17)
        """
    ))

# -------------------- MAIN -------------------- #
df = load_data()

if df is not None:
    page = sidebar_navigation()
    if page == _("Tentang Saya", "About Me"): tentang_saya()
    elif page == _("Proyek", "Project"): proyek(df)
    elif page == _("Machine Learning", "Machine Learning"): ml(df)
    elif page == _("Insight & Rekomendasi", "Insight & Recommendation"): insight()
    elif page == _("Kontak", "Contact"): kontak()
else:
    st.error(_("Gagal memuat data.", "Failed to load data."))
