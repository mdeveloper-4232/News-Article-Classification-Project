import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detection App")

news = st.text_area("Enter News Text")

if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter news text")
    else:
        news_vec = vectorizer.transform([news])
        prediction = model.predict(news_vec)

        if prediction[0] == 0:
            st.error(" This News is FAKE")
        else:
            st.success(" This News is REAL")