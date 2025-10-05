import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# --- Initialization and Model Loading ---
# Note: You should ensure these NLTK downloads happen before running the app
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Initialize Stemmer
ps = PorterStemmer()

# Load Model Artifacts
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model files ('vectorizer.pkl' or 'model.pkl') not found.")
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

# Set a better title with an emoji
st.title("✉️ SMS Spam Detector")

# Add a subtle subheader/description
st.caption("A Machine Learning powered tool to classify messages as Spam or Not Spam (Ham).")
st.markdown("---") # Visual separator

# Add a sidebar for 'About' and instructions
with st.sidebar:
    st.header("About this App")
    st.info(
        "This application is a simple demonstrator using a pre-trained **Naive Bayes** model "
        "and a **TF-IDF vectorizer** for natural language processing."
    )
    st.markdown("---")
    st.markdown("**How to Use:**")
    st.markdown("1. Enter a full message in the text box.")
    st.markdown("2. Click the 'Analyze Message' button.")
    st.markdown("3. Get the prediction instantly.")


# Text Area for Input
input_sms = st.text_area(
    "Enter the SMS/Email message here:", 
    height=150, 
    placeholder="Type or paste a suspicious message to check for spam..."
)

# Button for Prediction
if st.button("Analyze Message", use_container_width=True):
    if not input_sms.strip():
        st.warning("Please enter a message to analyze.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict
        result = model.predict(vector_input)[0]
        
        # 4. Display Result with improved icons and formatting
        st.markdown("### Prediction Result:")
        if result == 1:
            st.error("🚨 SPAM DETECTED!")
            st.write("This message is highly likely to be a **Spam** message.")
        else:
            st.success("✅ SAFE (Not Spam/Ham)")
            st.write("This message is likely **Not Spam** (Ham).")

st.markdown("---")
