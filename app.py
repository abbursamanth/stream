import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('C://Users//Samanth Abbur//pro//stream//DEDuCT_ChemicalBasicInformation.csv')

# Separate features (X) and target variable (y)
X = df.drop('estrogenic present', axis=1)  # Assuming 'estrogenic' is the target column
y = df['estrogenic present']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))


import streamlit as st

# Load libraries and data
# Assume you've already imported necessary libraries and loaded your dataset


from PIL import Image
import pytesseract
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

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
    st.write('Prediction:', 'Estrogenic' if prediction[0] == 1 else 'Non-Estrogenic')

elif text_input:
    prediction = predict_text(text_input)
    st.write('Input Text:', text_input)
    st.write('Prediction:', 'Estrogenic' if prediction[0] == 1 else 'Non-Estrogenic')



