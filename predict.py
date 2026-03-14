import pickle

# load saved model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

while True:
    message = input("Enter a message: ")

    if message == "exit":
        break

    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]

    if prediction == 1:
        print("🚨 Spam message")
    else:
        print("✅ Not Spam")