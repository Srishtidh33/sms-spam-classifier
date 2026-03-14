import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from the same folder
df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

# Map labels to 0/1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Text to features
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# Try a custom message
sample = ["Congratulations! You won a free ticket. Call now!"]
print("Prediction (0=ham, 1=spam):", model.predict(vectorizer.transform(sample))[0])