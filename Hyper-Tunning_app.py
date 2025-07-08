# bilingual_churn_app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# -------------------- CONFIG HALAMAN -------------------- #
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# -------------------- BAHASA -------------------- #
language = st.sidebar.radio("Language / Bahasa", ["🇮🇩 Indonesia", "🇺🇸 English"])
IN_ID = language == "🇮🇩 Indonesia"

def _(id, en):
    return id if IN_ID else en

# -------------------- LOAD DATA -------------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Brama17/Hyper-Tunning-Analysis/main/churn.csv")
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

# -------------------- NAVIGASI -------------------- #
def sidebar():
    st.sidebar.title(_("Navigasi", "Navigation"))
    return st.sidebar.radio(_("Pilih Halaman", "Choose Page"),
        [_("Tentang Saya", "About Me"), "📈 Proyek", "🤖 Machine Learning", "📊 Insight", "📞 Kontak"])

# -------------------- TENTANG SAYA -------------------- #
def about():
    st.title("👤 " + _("Tentang Saya", "About Me"))
    st.markdown(_(
        """
        Hai, saya **Harwian Brama Enggar Pramukti**. Saya seorang *Full Stack Data Scientist* dengan keahlian dalam:
        - 📊 Visualisasi & Analisis Data
        - 🧠 Machine Learning untuk prediksi Churn
        - 🧹 Data preprocessing
        - 🚀 Deploy model via Streamlit
        """,
        """
        Hi, I'm **Harwian Brama Enggar Pramukti**. I'm a *Full Stack Data Scientist* skilled in:
        - 📊 Data Visualization & Analysis
        - 🧠 Churn Prediction ML models
        - 🧹 Data preprocessing
        - 🚀 Model deployment with Streamlit
        """
    ))

# -------------------- PROYEK -------------------- #
def project(df):
    st.title("📈 " + _("Analisis Proyek", "Project Analysis"))

    # Filter
    st.sidebar.subheader(_("Filter Data", "Filter Data"))
    min_tenure, max_tenure = st.sidebar.slider("Tenure", 0, int(df['Tenure'].max()), (0, 72))
    df = df[(df['Tenure'] >= min_tenure) & (df['Tenure'] <= max_tenure)]

    # Visualisasi
    st.subheader(_("Distribusi Tenure", "Tenure Distribution"))
    fig, ax = plt.subplots()
    sns.histplot(df['Tenure'], bins=30, kde=True, color='skyblue', ax=ax)
    st.pyplot(fig)

    st.subheader(_("Churn Berdasarkan Kontrak", "Churn by Contract"))
    fig, ax = plt.subplots()
    sns.countplot(x='Contract', hue='Churn', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader(_("Korelasi Fitur", "Feature Correlation"))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# -------------------- MACHINE LEARNING -------------------- #
def ml(df):
    st.title("🤖 " + _("Model Machine Learning", "Machine Learning Model"))
    st.markdown(_("Pilih model untuk prediksi Churn:", "Choose a model to predict churn:"))

    model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
    features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    X_train, X_test, y_train, y_test = split_data(df, features)

    model = RandomForestClassifier(n_estimators=100, random_state=42) if model_choice == "Random Forest" else LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi
    st.subheader(_("Evaluasi Model", "Model Evaluation"))
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.code(classification_report(y_test, y_pred))

    # ROC Curve
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Feature Importance
    if model_choice == "Random Forest":
        st.subheader(_("Feature Importance", "Feature Importance"))
        importances = pd.Series(model.feature_importances_, index=features)
        fig, ax = plt.subplots()
        importances.sort_values().plot(kind='barh', color='orange', ax=ax)
        st.pyplot(fig)

    st.markdown(_(
        """
        #### Kenapa model ini?
        - **Random Forest**: Cocok untuk data tabular dan menangani non-linearitas dengan baik.
        - **Logistic Regression**: Model baseline yang sederhana dan interpretatif.
        """,
        """
        #### Why this model?
        - **Random Forest**: Great for tabular data and captures non-linearity.
        - **Logistic Regression**: Simple and interpretable baseline model.
        """
    ))

# -------------------- INSIGHT -------------------- #
def insight():
    st.title("📊 " + _("Insight & Rekomendasi", "Insight & Recommendation"))

    st.markdown(_(
        """
        - 📉 Pelanggan baru (tenure < 10 bulan) cenderung lebih mudah churn.
        - 💸 Biaya bulanan tinggi berasosiasi dengan churn lebih besar.
        - 📞 Kontrak jangka panjang mengurangi risiko churn.
        """,
        """
        - 📉 New customers (tenure < 10 months) are more likely to churn.
        - 💸 High monthly charges linked to higher churn.
        - 📞 Long-term contracts reduce churn risk.
        """
    ))

# -------------------- KONTAK -------------------- #
def contact():
    st.title("📞 " + _("Kontak", "Contact"))
    st.markdown("""
    - 📧 harwianbrama02@gmail.com
    - 💼 [LinkedIn](https://www.linkedin.com)
    - 💻 [GitHub](https://github.com/Brama17)
    """)

# -------------------- MAIN APP -------------------- #
df = load_data()
if df is not None:
    page = sidebar()
    if page == "📈 Proyek": project(df)
    elif page == "🤖 Machine Learning": ml(df)
    elif page == "📊 Insight": insight()
    elif page == "📞 Kontak": contact()
    else: about()
else:
    st.error("Gagal memuat data.")
