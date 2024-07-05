import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


import firebase_admin
from firebase_admin import credentials, firestore

# Setup Tesseract path (update this path if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if not firebase_admin._apps:
    cred = credentials.Certificate('C://Users//Samanth Abbur//pro//stream//estrovision-6085a-firebase-adminsdk-8mdt6-6b38b2c831.json')
    firebase_admin.initialize_app(cred)

# Initialize Firestore DB
db = firestore.client()





def register_user(username, password):
    users_ref = db.collection('users')
    user = users_ref.document(username).get()
    if user.exists:
        return False
    users_ref.document(username).set({
        'password': password
    })
    return True


def check_login(username, password):
    users_ref = db.collection('users')
    user = users_ref.document(username).get()
    if user.exists and user.to_dict()['password'] == password:
        return True
    return False


def login():
    st.session_state.logged_in = True

# Function to logout
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""


# Initialize session state for login and registration
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if 'register' not in st.session_state:
    st.session_state.register = False

# Login and Registration pages
if not st.session_state.logged_in and not st.session_state.register:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.username = username
            login()
            st.success("Login successful")
        else:
            st.error("Invalid username or password")
    if st.button("Register"):
        st.session_state.register = True
elif not st.session_state.logged_in and st.session_state.register:
    st.title("Register")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Create Account"):
        if register_user(new_username, new_password):
            st.session_state.register = False
            st.success("Registration successful. Please login.")
        else:
            st.error("Username already exists. Please choose a different username.")
    if st.button("Back to Login"):
        st.session_state.register = False
else:
    st.sidebar.button("Logout", on_click=logout)
    st.title(f'Welcome, {st.session_state.username}')

    # Streamlit UI for the main application
    st.title('Endocrine Disruptors and Estrogen Prediction')


file_path = r'C:\Users\Samanth Abbur\pro\stream\DEDuCT_ChemicalBasicInformation.csv'


df = pd.read_csv(file_path)

#
text_column_name = 'Name'  # Change this to your actual text column name

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
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))





# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
text_input = st.text_area("Or enter text directly:")




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






if uploaded_file is not None:
    image = Image.open(uploaded_file)
    prediction, extracted_text = predict_image(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('Extracted Text:', extracted_text)
    st.write('Prediction:', 'estrogen present' if prediction[0] == 1 else 'Estrogenic')

elif text_input:
     prediction = predict_text(text_input)
     st.write('Input Text:', text_input)
     st.write('Prediction:', 'estrogen present' if prediction[0] == 1 else 'Estrogenic')
