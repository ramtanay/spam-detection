import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# --- Initialization and Model Loading ---
# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# For newer NLTK versions (≥3.8), include this too
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Initialize Stemmer
ps = PorterStemmer()

# Load Model Artifacts
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ Model files ('vectorizer.pkl' or 'model.pkl') not found.")
    st.stop()

# --- Text Preprocessing Function ---
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# --- Streamlit Web App Layout ---
st.set_page_config(page_title="SMS Spam Detector", page_icon="✉️", layout="centered")

# Title and description
st.title("✉️ SMS Spam Detector")
st.caption("A Machine Learning powered tool to classify messages as Spam or Not Spam (Ham).")
st.markdown("---")

# Sidebar: About and Instructions
with st.sidebar:
    st.header("ℹ️ About this App")
    st.info(
        "This demo uses a pre-trained **Naive Bayes** model and a **TF-IDF vectorizer** "
        "to classify SMS messages as Spam or Ham."
    )
    st.markdown("---")
    st.markdown("**📋 How to Use:**")
    st.markdown("1. Enter a message below.")
    st.markdown("2. Click **Analyze Message**.")
    st.markdown("3. View the prediction instantly!")

# Input Text Area
input_sms = st.text_area(
    "✉️ Enter the SMS or Email message below:",
    height=150,
    placeholder="Type or paste a suspicious message to check for spam..."
)

# Analyze Button
if st.button("🔍 Analyze Message", use_container_width=True):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        # 1. Preprocess the message
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict
        result = model.predict(vector_input)[0]
        
        # 4. Display result
        st.markdown("### 🧾 Prediction Result:")
        if result == 1:
            st.error("🚨 **SPAM DETECTED!**")
            st.write("This message is highly likely to be **Spam**.")
        else:
            st.success("✅ **SAFE (Not Spam / Ham)**")
            st.write("This message appears to be **legitimate (Ham)**.")
            
st.markdown("---")
st.caption("Developed by Ramtanay with ❤️")

