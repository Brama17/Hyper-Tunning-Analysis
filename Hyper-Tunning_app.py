import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# -------------------- TRANSLATION FUNCTION -------------------- #
def _(id_text, en_text):
    lang = st.session_state.get("language", "id")
    return id_text if lang == "id" else en_text

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(
    page_title="Portofolio Harwian",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📌"
)

# -------------------- LOAD DATA -------------------- #
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Brama17/Hyper-Tunning-Analysis/main/churn.csv"
    try:
        df = pd.read_csv(url)
        df.dropna(inplace=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def split_data(df, features):
    X = df[features]
    y = df['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- SIDEBAR -------------------- #
def sidebar_navigation():
    st.sidebar.selectbox("Select Language / Pilih Bahasa", ["id", "en"], key="language")
    st.sidebar.title(_("📂 Navigasi", "📂 Navigation"))
    return st.sidebar.radio(_("Pilih Halaman", "Select Page"),
                            ["Tentang Saya", "Proyek", "Machine Learning", "Insight & Rekomendasi", "Kontak"])

# -------------------- TENTANG SAYA -------------------- #
def tentang_saya():
    st.markdown("## 📌 " + _("Tentang Saya", "About Me"))
    st.markdown("### Full Stack Data Science")
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
        Hi, I'm **Harwian Brama Enggar Pramukti**, a *Full Stack Data Scientist* with experience:
        - 📊 Data exploration & visualization using Python and Power BI
        - 🧹 Data preprocessing: missing value, outlier, feature engineering
        - 🧠 Building ML models for customer churn prediction (Telco Churn)
        - 🚀 Deploying models and interactive dashboards using Streamlit
        - 🔧 Tools: pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn
        """
    ))

# -------------------- PROYEK -------------------- #
def proyek(df):
    st.title("📈 " + _("Proyek Analisis", "Analysis Project"))
    st.markdown("#### Telco Customer Churn Analysis")

    st.sidebar.markdown("## 🔍 " + _("Filter Data", "Data Filter"))
    min_tenure, max_tenure = st.sidebar.slider("Tenure", 0, int(df['Tenure'].max()), (0, 72))
    df_filtered = df[(df['Tenure'] >= min_tenure) & (df['Tenure'] <= max_tenure)]

    st.subheader(_("Distribusi Tenure", "Tenure Distribution"))
    fig1, ax1 = plt.subplots()
    sns.histplot(df_filtered['Tenure'], bins=30, kde=True, ax=ax1, color='skyblue')
    st.pyplot(fig1)

    st.subheader(_("Proporsi Churn", "Churn Proportion"))
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Churn', data=df_filtered, palette='Set2', ax=ax2)
    ax2.set_xticklabels([_("Tidak Churn", "No Churn"), _("Churn", "Churn")])
    st.pyplot(fig2)

    st.subheader(_("Monthly Charges vs Churn", "Monthly Charges vs Churn"))
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df_filtered, palette='Set1', ax=ax3)
    ax3.set_xticklabels([_("Tidak Churn", "No Churn"), _("Churn", "Churn")])
    st.pyplot(fig3)

# -------------------- MACHINE LEARNING -------------------- #
def ml(df):
    st.title("🤖 " + _("Model Machine Learning", "Machine Learning Model"))
    st.markdown(_("Gunakan model ML untuk memprediksi churn pelanggan.", "Use ML models to predict customer churn."))

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

    # Evaluasi
    st.subheader(_("Evaluasi Model", "Model Evaluation"))
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.code(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig4, ax4 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax4)
    st.pyplot(fig4)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # Feature Importance
    if model_choice == "Random Forest":
        st.subheader("Feature Importance")
        importance = pd.Series(model.feature_importances_, index=features)
        fig5, ax5 = plt.subplots()
        importance.sort_values().plot(kind='barh', ax=ax5, color='orange')
        st.pyplot(fig5)

# -------------------- INSIGHT -------------------- #
def insight():
    st.title("📊 " + _("Insight & Rekomendasi", "Insights & Recommendation"))

    st.subheader("🔍 Insight")
    st.markdown(_(
        """
        - Pelanggan dengan **Tenure < 10 bulan** lebih rawan churn.
        - Biaya bulanan tinggi berkorelasi dengan risiko churn.
        - Total Charges rendah sering merupakan pelanggan baru.
        """,
        """
        - Customers with **Tenure < 10 months** are more likely to churn.
        - High monthly charges correlate with higher churn risk.
        - Low Total Charges usually indicate new customers.
        """
    ))

    st.subheader("✅ Rekomendasi")
    st.markdown(_(
        """
        - Berikan insentif untuk pelanggan baru.
        - Tawarkan diskon pada pelanggan berisiko tinggi.
        - Fokus retensi pada segmen Tenure pendek.
        """,
        """
        - Give incentives to new customers.
        - Offer discounts to high-risk customers.
        - Focus retention on short-tenure segments.
        """
    ))

# -------------------- KONTAK -------------------- #
def kontak():
    st.title("📞 " + _("Kontak", "Contact"))
    st.markdown("""
    - 📧 Email: [harwianbrama02@gmail.com](mailto:harwianbrama02@gmail.com)
    - 💼 LinkedIn: [Harwian Brama Enggar P](https://www.linkedin.com)
    - 📱 WhatsApp: [082141922446](https://wa.me/6282141922446)
    - 💻 GitHub: [GitHub Profil](https://github.com/Brama17)
    """)

# -------------------- MAIN -------------------- #
df = load_data()
if df is not None:
    page = sidebar_navigation()
    if page == "Tentang Saya": tentang_saya()
    elif page == "Proyek": proyek(df)
    elif page == "Machine Learning": ml(df)
    elif page == "Insight & Rekomendasi": insight()
    elif page == "Kontak": kontak()
else:
    st.error("Gagal memuat data.")
