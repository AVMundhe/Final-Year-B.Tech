import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# =========================
# Load Models and Vectorizers
# =========================

emotion_model = joblib.load("models/emotion_model.joblib")
emotion_tfidf = joblib.load("models/emotion_tfidf.joblib")

depression_model = joblib.load("models/depression_model.joblib")
depression_tfidf = joblib.load("models/depression_tfidf.joblib")


# =========================
# Stopwords
# =========================

stop_words = set(stopwords.words("english"))

# Preserve important negation words
negation_words = {"no", "not", "nor", "never"}
stop_words = stop_words - negation_words


# =========================
# Contractions
# =========================

CONTRACTIONS = {
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "he's": "he is",
    "he'll": "he will",
    "he'd": "he would",
    "she's": "she is",
    "she'll": "she will",
    "she'd": "she would",
    "it's": "it is",
    "it'll": "it will",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'd": "they would",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "mustn't": "must not",
    "needn't": "need not"
}


def expand_contractions(text):
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    return text


# =========================
# Text Cleaning
# =========================

def clean_text(text):

    text = text.replace("’", "'").replace("‘", "'")

    text = re.sub(
        r"(https?://\S+|www\.\S+|\S+://\S+)",
        "",
        text
    )

    text = text.lower()

    text = expand_contractions(text)

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# Preprocessing
# =========================

def preprocess_text(text):

    text = clean_text(text)

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# =========================
# Streamlit UI
# =========================

st.title("Depression & Emotion Detection")

st.write(
    "Enter a text to analyze its predicted emotion "
    "and whether it is depression-related."
)

text = st.text_area(
    "Enter your text:",
    height=150,
    placeholder="Type something here..."
)


# =========================
# Prediction
# =========================

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        processed_text = preprocess_text(text)

        # Emotion prediction
        emotion_features = emotion_tfidf.transform(
            [processed_text]
        )

        emotion_prediction = emotion_model.predict(
            emotion_features
        )[0]

        emotion_labels = {
            0: "Sadness",
            1: "Joy",
            2: "Love",
            3: "Anger",
            4: "Fear",
            5: "Surprise"
        }

        predicted_emotion = emotion_labels[
            emotion_prediction
        ]


        # Depression prediction
        depression_features = depression_tfidf.transform(
            [processed_text]
        )

        depression_prediction = depression_model.predict(
            depression_features
        )[0]


        # Display results
        st.subheader("Results")

        st.write(
            f"**Predicted Emotion:** {predicted_emotion}"
        )

        if depression_prediction == 1:
            st.write(
                "**Depression-related text:** Yes"
            )
        else:
            st.write(
                "**Depression-related text:** No"
            )