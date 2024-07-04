import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Setup Tesseract path (update this path if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load dataset
df = pd.read_csv('C://Users//amanth Abbur//pro//stream//DEDuCT_ChemicalBasicInformation.csv')

# Ensure the dataset has a 'text_column' and 'estrogenic' column
# Replace 'text_column' with the actual name of the column containing the text data
text_column_name = 'Name'

# Separate features (X) and target variable (y)
X = df[text_column_name]
y = df['estrogen present']

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)



def process_image(image):
    text = pytesseract.image_to_string(image)
    return text

def predict_text(text):
    # Transform the input text to TF-IDF features
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
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
