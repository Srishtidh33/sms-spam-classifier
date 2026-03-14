import streamlit as st
import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="SMS Spam Detector")

st.title("📩 SMS Spam Detector")
st.write("Enter an SMS message and the AI will detect whether it is Spam or Not.")

message = st.text_area("Message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message")

    else:
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]

        if prediction == 1:
            st.error("🚨 Spam Message Detected")
        else:
            st.success("✅ This message is not spam")