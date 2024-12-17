import streamlit as st
import joblib
import os
import re

# Load your trained model
model_path = 'C:\\Users\\MITS\\Downloads\\Spam Email Detection\\Spam Email Detection\\model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"Model file not found at {model_path}")
    st.stop()

# Load your vectorizer
vectorizer_path = 'C:\\Users\\MITS\\Downloads\\Spam Email Detection\\Spam Email Detection\\vectorizer.pkl'
if os.path.exists(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
else:
    st.error(f"Vectorizer file not found at {vectorizer_path}")
    st.stop()

def preprocess_text(text):
    """Preprocess the input text by removing unwanted characters and whitespace."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove all non-word characters
    text = text.strip()  # Remove leading and trailing whitespace
    return text

# Streamlit App
st.title("Spam Email Detection")
st.write("Enter text below to predict whether it's Spam or Not Spam.")

# Input text box
email_text = st.text_area("Email Text", placeholder="Enter your email text here...")

# Predict button
if st.button("Predict"):
    if email_text.strip():
        try:
            # Preprocess the input text
            processed_text = preprocess_text(email_text)

            # Vectorize the input text
            text_vectorized = vectorizer.transform([processed_text])

            # Make predictions using the model
            prediction = model.predict(text_vectorized)

            # Map the prediction to a label
            prediction_label = "Spam" if prediction[0] == 1 else "Not Spam"

            # Display the prediction result
            st.success(f"Prediction: {prediction_label}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter some text to make a prediction.")
