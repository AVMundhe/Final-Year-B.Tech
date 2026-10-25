import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = {
    "text": [
        "I love this product",
        "This is amazing",
        "Very good experience",
        "I hate this product",
        "This is terrible",
        "Very bad experience",
        "The product arrived today",
        "The package was delivered",
        "I received the product",
        
        "मुझे यह उत्पाद पसंद है",
        "यह बहुत अच्छा है",
        "मुझे यह बहुत खराब लगा",
        "यह बहुत बुरा उत्पाद है",
        "उत्पाद आज पहुंचा",
        
        "मला हे उत्पादन आवडते",
        "हा खूप चांगला अनुभव आहे",
        "मला हे उत्पादन आवडले नाही",
        "हा खूप खराब अनुभव आहे",
        "उत्पादन आज मिळाले"
    ],

    "label": [
        2, 2, 2,
        0, 0, 0,
        1, 1, 1,
        
        2, 2,
        0, 0,
        1,
        
        2, 2,
        0, 0,
        1
    ]
}

df = pd.DataFrame(data)

print(df)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

print("Predictions:")
print(predictions)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

