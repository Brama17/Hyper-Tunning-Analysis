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
    page_icon="\U0001F4CC"
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
    language = st.sidebar.selectbox("Select Language / Pilih Bahasa", ["id", "en"])
    st.session_state.language = language
    st.sidebar.title("\U0001F4C1 Navigasi" if language == "id" else "\U0001F4C1 Navigation")
    options = [
        ("Tentang Saya", "About Me"),
        ("Proyek", "Project"),
        ("Machine Learning", "Machine Learning"),
        ("Insight & Rekomendasi", "Insights & Recommendations"),
        ("Kontak", "Contact")
    ]
    option_labels = [opt[0] if language == "id" else opt[1] for opt in options]
    return st.sidebar.radio("Pilih Halaman / Select Page", option_labels)

# -------------------- TENTANG SAYA -------------------- #
def tentang_saya():
    lang = st.session_state.language
    if lang == "id":
        st.markdown("## \U0001F4CC Tentang Saya")
        st.markdown("### Full Stack Data Science")
        st.markdown("""
        Hai, saya **Harwian Brama Enggar Pramukti**, seorang *Full Stack Data Science* dengan pengalaman:
        - \U0001F4CA Eksplorasi Data & Visualisasi menggunakan Python dan Power BI  
        - \U0001F9F9 Data preprocessing: missing value, outlier, feature engineering  
        - \U0001F9E0 Membangun model ML untuk prediksi churn pelanggan (Telco Churn)  
        - \U0001F680 Deploy model dan dashboard interaktif menggunakan Streamlit  
        - \U0001F527 Tools: pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn
        """)
    else:
        st.markdown("## \U0001F4CC About Me")
        st.markdown("### Full Stack Data Science")
        st.markdown("""
        Hi, I am **Harwian Brama Enggar Pramukti**, a *Full Stack Data Scientist* with experience:
        - \U0001F4CA Data exploration & visualization using Python and Power BI  
        - \U0001F9F9 Data preprocessing: missing values, outliers, feature engineering  
        - \U0001F9E0 Building ML models to predict customer churn (Telco Churn)  
        - \U0001F680 Deploying models and interactive dashboards using Streamlit  
        - \U0001F527 Tools: pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn
        """)

# -------------------- PROYEK -------------------- #
def proyek(df):
    lang = st.session_state.language
    st.title("\U0001F4C8 Proyek" if lang == "id" else "\U0001F4C8 Project")
    st.markdown("#### Telco Customer Churn Analysis")

    st.sidebar.markdown("## \U0001F50D Filter Data")
    min_tenure, max_tenure = st.sidebar.slider("Rentang Tenure", 0, int(df['Tenure'].max()), (0, 72))
    df_filtered = df[(df['Tenure'] >= min_tenure) & (df['Tenure'] <= max_tenure)]

    st.subheader("Distribusi Tenure" if lang == "id" else "Tenure Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df_filtered['Tenure'], bins=30, kde=True, color='skyblue', ax=ax1)
    st.pyplot(fig1)

    st.subheader("Proporsi Churn" if lang == "id" else "Churn Proportion")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_filtered, x='Churn', palette='Set2', ax=ax2)
    ax2.set_xticklabels(['Tidak Churn', 'Churn'] if lang == "id" else ['No Churn', 'Churn'])
    st.pyplot(fig2)

    st.subheader("Monthly Charges vs Churn")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df_filtered, ax=ax3, palette='Set1')
    ax3.set_xticklabels(['Tidak Churn', 'Churn'] if lang == "id" else ['No Churn', 'Churn'])
    st.pyplot(fig3)

# -------------------- MACHINE LEARNING -------------------- #
def machine_learning(df):
    lang = st.session_state.language
    st.title("\U0001F916 Model Machine Learning" if lang == "id" else "\U0001F916 Machine Learning Model")

    st.markdown("Gunakan model untuk memprediksi pelanggan yang akan churn." if lang == "id" else "Use models to predict customers likely to churn.")

    model_choice = st.selectbox("Pilih Model" if lang == "id" else "Select Model", ["Random Forest", "Logistic Regression"])
    features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    X_train, X_test, y_train, y_test = split_data(df, features)

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Evaluasi Model" if lang == "id" else "Model Evaluation")
    st.write(f"**Akurasi:** {accuracy_score(y_test, y_pred):.2f}")
    st.code(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    fig4, ax4 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax4)
    st.pyplot(fig4)

    if model_choice == "Random Forest":
        st.subheader("Feature Importance")
        importance = pd.Series(model.feature_importances_, index=features)
        fig5, ax5 = plt.subplots()
        importance.sort_values().plot(kind='barh', color='orange', ax=ax5)
        st.pyplot(fig5)

# -------------------- INSIGHT & REKOMENDASI -------------------- #
def insight_rekomendasi():
    lang = st.session_state.language
    st.title("\U0001F4CA Insight & Rekomendasi" if lang == "id" else "\U0001F4CA Insights & Recommendations")

    if lang == "id":
        st.markdown("""
        - \U0001F4C9 Pelanggan dengan **Tenure pendek (< 10 bulan)** memiliki tingkat churn yang tinggi.
        - \U0001F4B8 Pelanggan dengan **biaya bulanan tinggi** lebih berisiko churn.
        - \U0001F4E6 Pelanggan dengan **Total Charges rendah** sering kali adalah pelanggan baru.
        
        ### ✅ Rekomendasi Bisnis
        - \U0001F381 Beri loyalty reward untuk pelanggan baru (< 6 bulan).
        - \U0001F4E3 Tawarkan promo khusus untuk pelanggan dengan tagihan tinggi.
        - \U0001F4DE Fokuskan kampanye retensi pada segmen rawan churn.
        """)
    else:
        st.markdown("""
        - \U0001F4C9 Customers with **short tenure (< 10 months)** have a high churn rate.
        - \U0001F4B8 Customers with **high monthly charges** are more likely to churn.
        - \U0001F4E6 Customers with **low total charges** are often new customers.
        
        ### ✅ Business Recommendations
        - \U0001F381 Provide loyalty rewards for new customers (< 6 months).
        - \U0001F4E3 Offer special promos for high-billing customers.
        - \U0001F4DE Focus retention campaigns on high-risk segments.
        """)

# -------------------- KONTAK -------------------- #
def kontak():
    st.title("\U0001F4DE Kontak")
    st.markdown("Silakan hubungi saya melalui informasi berikut:")
    st.markdown("""
    - \U0001F4E7 **Email**: [harwianbrama02@gmail.com](mailto:harwianbrama02@gmail.com)
    - \U0001F4BC **LinkedIn**: [Harwian Brama Enggar P](https://www.linkedin.com)
    - \U0001F4F1 **WhatsApp**: [082141922446](https://wa.me/6282141922446)
    - \U0001F4BB **GitHub**: [GitHub Profil](https://github.com/Brama17)
    """)

# -------------------- MAIN APP -------------------- #
df = load_data()
if df is not None:
    if 'language' not in st.session_state:
        st.session_state.language = "id"
    page = sidebar_navigation()
    if page in ["Tentang Saya", "About Me"]:
        tentang_saya()
    elif page in ["Proyek", "Project"]:
        proyek(df)
    elif page in ["Machine Learning"]:
        machine_learning(df)
    elif page in ["Insight & Rekomendasi", "Insights & Recommendations"]:
        insight_rekomendasi()
    elif page in ["Kontak", "Contact"]:
        kontak()
else:
    st.error("Gagal memuat data.")
