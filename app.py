import pytesseract
from PIL import Image
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def tokenize_text(text):
    return preprocess_text(text).split()

text_data = pd.read_csv("C://Users//Samanth Abbur//pro//stream//DEDuCT_ChemicalBasicInformation.csv")
text_data['processed'] = text_data['Name'].apply(preprocess_text)

vectorizer = TfidfVectorizer(tokenizer=tokenize_text, preprocessor=None)
X = vectorizer.fit_transform(text_data['processed'])

y = text_data['estrogen present']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



import streamlit as st

# Load libraries and data
# Assume you've already imported necessary libraries and loaded your dataset



# Setup Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary


def process_image(image):
    text = pytesseract.image_to_string(image)
    return text

def predict_text(text):
    # Example: Convert text to features, predict using the model
    # Here we should implement text preprocessing and feature extraction
    text_features = [len(text)]  # Example feature
    prediction = model.predict([text_features])
    return prediction

def predict_image(image):
    text = process_image(image)
    prediction = predict_text(text)
    return prediction, text

# Streamlit UI
st.title('Endocrine Disruptors and Estrogen Prediction')

# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
text_input = st.text_area("Or enter text directly:")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    prediction, extracted_text = predict_image(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('Extracted Text:', extracted_text)
    st.write('Prediction:', 'estrogen present' if prediction[0] == 1 else 'Non-Estrogenic')

elif text_input:
    prediction = predict_text(text_input)
    st.write('Input Text:', text_input)
    st.write('Prediction:', 'estrogen present' if prediction[0] == 1 else 'Non-Estrogenic')

