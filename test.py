from PIL import Image
import pytesseract

def extract_text_from_image(image):
    return pytesseract.image_to_string(Image.open(image))

import requests

def check_ingredients(text):
    # Replace with actual API call and logic
    response = requests.post("LL-K56g8ecLETBnnLKcvt2IUHGPdvOMtwINDL05qDdhwyi98dLqpFxILxAE9XtmB3Zi", data={"text": text})
    return response.json()

def get_health_effects(chemical):
    # Provide detailed health effects of the chemical
    health_effects = {
        "Chemical1": "Effect1",
        "Chemical2": "Effect2",
        # Add more chemicals and their effects
    }
    return health_effects.get(chemical, "No information available")


import streamlit as st

st.title("Health Care with AI")
st.header("Check for Endocrine Disruptors and Estrogens in Product Ingredients")

uploaded_image = st.file_uploader("Upload an image of the ingredient list", type=["png", "jpg", "jpeg"])
input_text = st.text_area("Or enter the ingredient list manually")

if st.button("Check Ingredients"):
    if uploaded_image:
        extracted_text = extract_text_from_image(uploaded_image)
    else:
        extracted_text = input_text

    if extracted_text:
        result = check_ingredients(extracted_text)
        st.write("Analysis Result:")
        for chemical in result.get("chemicals", []):
            st.write(f"{chemical}: {get_health_effects(chemical)}")
    else:
        st.write("Please provide an image or text to analyze.")

print("hello")
