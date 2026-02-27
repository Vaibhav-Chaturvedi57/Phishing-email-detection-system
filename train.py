import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/combined_data.csv")

print("Dataset Loaded:", df.shape)

# -----------------------------
# 2. Clean Text Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["cleaned_text"] = df["text"].apply(clean_text)

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X = df["cleaned_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # IMPORTANT
)

print("Training size:", len(X_train))
print("Testing size:", len(X_test))

# -----------------------------
# 4. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)   # unigrams + bigrams
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# 5. Train Logistic Regression
# -----------------------------
model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 7. Save Model + Vectorizer
# -----------------------------
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("\nModel and Vectorizer saved successfully.")////