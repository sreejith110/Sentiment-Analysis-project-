import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Sentiment Analysis App")

user_input = st.text_area("Enter a review:")

if st.button("Predict"):
    if user_input:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)

        st.success(f"Predicted Sentiment: {prediction[0]}")
    else:
        st.warning("Please enter a review.")