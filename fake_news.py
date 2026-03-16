import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("dataset.csv")

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vector, y)

news = input("Enter news: ")
news_vector = vectorizer.transform([news])

prediction = model.predict(news_vector)

print("Prediction:", prediction[0])