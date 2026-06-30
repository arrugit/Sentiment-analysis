import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# =====================
# Load Dataset
# =====================

df = pd.read_csv(r"C:\Users\ua\Downloads\archive\Reviews.csv")

# Keep only required columns
df = df[['Text', 'Score']]

# Remove missing reviews
df.dropna(inplace=True)

# =====================
# Convert Score to Sentiment
# =====================

# Positive = 4,5
# Negative = 1,2
# Remove Neutral = 3

df = df[df['Score'] != 3]

df['Sentiment'] = df['Score'].apply(
    lambda x: 1 if x >= 4 else 0
)

# =====================
# Text Cleaning
# =====================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

df['Text'] = df['Text'].apply(clean_text)

# =====================
# Train/Test Split
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    df['Text'],
    df['Sentiment'],
    test_size=0.2,
    random_state=42
)

# =====================
# TF-IDF + BIGRAM + TRIGRAM
# =====================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1,3),     # unigram + bigram + trigram
            max_features=100000,
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

# =====================
# Train
# =====================

model.fit(X_train, y_train)

# =====================
# Evaluate
# =====================

predictions = model.predict(X_test)

print("Accuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# =====================
# Save Model
# =====================

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")