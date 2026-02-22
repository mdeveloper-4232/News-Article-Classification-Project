import joblib

# Load saved model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Take user input
news = input("Enter news text: ")

# Convert text
news_vec = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vec)

if prediction[0] == 0:
    print("This news is FAKE")
else:
    print("This news is REAL")