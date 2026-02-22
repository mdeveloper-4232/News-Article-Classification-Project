import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data.sample(frac=1, random_state=42)
# 2. Use only news text
X = data["text"]
y = data["label"]
data["text"] = data["text"].str.lower()

# 3. Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# 4. Split into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Check accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# 7. Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")


