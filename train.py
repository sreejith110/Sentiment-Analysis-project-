import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset
data = {
    "review": [
        "This product is amazing",
        "I love this item",
        "Very good quality",
        "Worst product ever",
        "Very bad experience",
        "I hate this",
        "It is okay",
        "Not bad but not great"
    ],
    "sentiment": [
        "Positive",
        "Positive",
        "Positive",
        "Negative",
        "Negative",
        "Negative",
        "Neutral",
        "Neutral"
    ]
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["review"])

# Train model
model = LogisticRegression()
model.fit(X, df["sentiment"])

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved!")