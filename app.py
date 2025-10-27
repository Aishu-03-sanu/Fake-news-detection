# 🧩 --- MUST BE FIRST ---
import streamlit as st
st.set_page_config(
    page_title="📰 News Classifier",
    page_icon="🧠",
    layout="centered"
)

# 📦 --- Import Libraries ---
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 🧠 --- Download NLTK Resources (only first run) ---
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")

# 🧩 --- Load Trained Model and TF-IDF ---
@st.cache_resource
def load_model():
    try:
        tfidf = joblib.load("tfidf_vectorizer.joblib")
        model = joblib.load("news_classifier_model.joblib")
        return tfidf, model
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure 'tfidf_vectorizer.joblib' and 'news_classifier_model.joblib' exist in this folder.")
        st.stop()

tfidf, model = load_model()

# ✨ --- Text Cleaning Function ---
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# 🧭 --- App Header ---
st.title("📰 Fake News Detection App")
st.markdown("""
This web app uses a **Logistic Regression / Naive Bayes** model trained on a labeled news dataset  
to predict whether a news article is **Real** or **Fake**.
""")

# 📝 --- User Input ---
user_input = st.text_area("🗞️ Paste or type a news article below:", height=200, placeholder="Enter full news article here...")

# 🔍 --- Prediction ---
if st.button("🚀 Classify News"):
    if user_input.strip():
        with st.spinner("Analyzing... Please wait..."):
            clean = clean_text(user_input)
            vec = tfidf.transform([clean])
            pred = model.predict(vec)[0]

        # 🎯 Display Result
        if pred == 1:
            st.success("✅ The news article is **REAL**.")
        else:
            st.error("🚨 The news article is **FAKE**.")
    else:
        st.warning("⚠️ Please enter some text before clicking Classify.")

# 📘 --- Sidebar Info ---
st.sidebar.header("ℹ️ About this App")
st.sidebar.write("""
**Fake News Classifier**  
Built using:
- 🧠 Scikit-learn  
- 📚 NLTK for text processing  
- 🌐 Streamlit for deployment  

**Author:** Shrilaxmi  
**Dataset:** Kaggle – Fake and Real News Dataset  
""")

