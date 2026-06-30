import streamlit as st
import pickle

# Load trained model

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Amazon Review Sentiment Analyzer")

st.write("Enter a review below")

review = st.text_area("Review")

if st.button("Analyze"):

    if review.strip():

        prediction = model.predict([review])[0]

        probability = model.predict_proba([review])[0]

        if prediction == 1:
            st.success("Positive Review")
            st.write(
                f"Confidence: {max(probability)*100:.2f}%"
            )
        else:
            st.error("Negative Review")
            st.write(
                f"Confidence: {max(probability)*100:.2f}%"
            )

    else:
        st.warning("Please enter a review")