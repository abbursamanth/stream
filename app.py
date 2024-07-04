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

# Main title and header
st.title('Estrogenic Chemical Classification')
st.header('Logistic Regression Model')

# Display dataset if needed
st.subheader('Dataset')
st.write(df)  # Display your dataset in Streamlit if desired

# Train and evaluate the model
st.subheader('Model Evaluation')

# Evaluate model performance
st.write(classification_report(y_test, y_pred))

# Display confusion matrix
st.write('Confusion Matrix:')
st.write(confusion_matrix(y_test, y_pred))

