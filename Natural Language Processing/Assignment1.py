# ==========================================================
# TEXT PREPROCESSING PIPELINE USING NLTK
# Author : Aditya Mundhe
# Concepts Covered:
# 1. Whitespace Tokenization
# 2. Word Tokenization (Rule-Based)
# 3. Sentence Tokenization
# 4. Stop Word Removal
# 5. Porter Stemming
# 6. Snowball Stemming
# 7. Lemmatization
# 8. POS Tagging
# ==========================================================

# -----------------------------
# Import Required Libraries
# -----------------------------

import nltk

# Tokenizers
from nltk.tokenize import (
    WhitespaceTokenizer,
    word_tokenize,
    sent_tokenize
)

# Stop Words
from nltk.corpus import stopwords

# Stemmers
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer

# Lemmatizer
from nltk.stem import WordNetLemmatizer

# POS Tagging
from nltk import pos_tag


# -----------------------------
# Download Required Resources
# (Only required the first time)
# -----------------------------

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


# -----------------------------
# Take Input from User
# -----------------------------

text = input("\nEnter any sentence:\n")


# ==========================================================
# 1. WHITESPACE TOKENIZATION
# ==========================================================

print("\n==============================")
print("1. WHITESPACE TOKENIZATION")
print("==============================")

# Create Whitespace Tokenizer Object
whitespace = WhitespaceTokenizer()

# Tokenize the sentence
whitespace_tokens = whitespace.tokenize(text)

print(whitespace_tokens)


# ==========================================================
# 2. RULE BASED TOKENIZATION
# ==========================================================

print("\n==============================")
print("2. WORD TOKENIZATION")
print("==============================")

# Rule based tokenizer
word_tokens = word_tokenize(text)

print(word_tokens)


# ==========================================================
# 3. SENTENCE TOKENIZATION
# ==========================================================

print("\n==============================")
print("3. SENTENCE TOKENIZATION")
print("==============================")

sentences = sent_tokenize(text)

print(sentences)


# ==========================================================
# 4. STOP WORD REMOVAL
# ==========================================================

print("\n==============================")
print("4. STOP WORD REMOVAL")
print("==============================")

# Load English Stop Words
stop_words = set(stopwords.words("english"))

filtered_words = []

# Remove stop words
for word in word_tokens:

    # Convert to lowercase before comparison
    if word.lower() not in stop_words:

        filtered_words.append(word)

print(filtered_words)


# ==========================================================
# 5. PORTER STEMMER
# ==========================================================

print("\n==============================")
print("5. PORTER STEMMER")
print("==============================")

porter = PorterStemmer()

porter_output = []

for word in filtered_words:

    porter_output.append(porter.stem(word))

print(porter_output)


# Optional : SnowballStemmer!
# # ==========================================================
# # 6. SNOWBALL STEMMER
# # ==========================================================

# print("\n==============================")
# print("6. SNOWBALL STEMMER")
# print("==============================")

# snowball = SnowballStemmer("english")

# snowball_output = []

# for word in filtered_words:

#     snowball_output.append(snowball.stem(word))

# print(snowball_output)


# ==========================================================
# 7. LEMMATIZATION
# ==========================================================

print("\n==============================")
print("7. LEMMATIZATION")
print("==============================")

lemmatizer = WordNetLemmatizer()

lemma_output = []

for word in filtered_words:

    lemma_output.append(lemmatizer.lemmatize(word))

print(lemma_output)


# ==========================================================
# 8. POS TAGGING
# ==========================================================

print("\n==============================")
print("8. POS TAGGING")
print("==============================")

# POS Tagging is performed on word tokens
pos_output = pos_tag(word_tokens)

print(pos_output)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n=========================================")
print("TEXT PREPROCESSING PIPELINE COMPLETED")
print("=========================================")

print("\nOriginal Text:")
print(text)

print("\nWhitespace Tokens:")
print(whitespace_tokens)

print("\nWord Tokens:")
print(word_tokens)

print("\nSentence Tokens:")
print(sentences)

print("\nAfter Stop Word Removal:")
print(filtered_words)

print("\nPorter Stemmer Output:")
print(porter_output)

print("\nSnowball Stemmer Output:")
print(snowball_output)

print("\nLemmatization Output:")
print(lemma_output)

print("\nPOS Tagged Output:")
print(pos_output)