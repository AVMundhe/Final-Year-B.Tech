#Simplified Code
# # # ==========================================================
# # BAG OF WORDS IMPLEMENTATION FROM SCRATCH
# # (Improved Version using Regex)
# # ==========================================================

# import re

# # -----------------------------
# # Step 1 : Input Documents
# # -----------------------------

# documents = [
#     "I love NLP!!",
#     "I love Python.",
#     "Python is Amazing!!!"
# ]

# # -----------------------------
# # Step 2 : Tokenization Function
# # Lowercase + Remove Punctuation + Tokenize
# # -----------------------------

# def tokenize(text):

#     # Convert text into lowercase
#     text = text.lower()

#     # Keep only complete alphabetic words
#     words = re.findall(r"[a-z]+", text)

#     return words


# # -----------------------------
# # Step 3 : Build Vocabulary
# # -----------------------------

# vocabulary = {}

# index = 0

# for doc in documents:

#     words = tokenize(doc)

#     for word in words:

#         if word not in vocabulary:

#             vocabulary[word] = index
#             index += 1

# print("Vocabulary")
# print(vocabulary)


# # -----------------------------
# # Step 4 : Create Bag of Words
# # -----------------------------

# bow_matrix = []

# for doc in documents:

#     # Tokenize current document
#     words = tokenize(doc)

#     # Create vector filled with zeros
#     vector = [0] * len(vocabulary)

#     # Count frequency
#     for word in words:

#         position = vocabulary[word]

#         vector[position] += 1

#     bow_matrix.append(vector)


# # -----------------------------
# # Step 5 : Display Result
# # -----------------------------

# print("\nBag of Words Matrix\n")

# for i in range(len(documents)):

#     print("Document", i + 1)
#     print(documents[i])
#     print(bow_matrix[i])
#     print()

import re
from collections import Counter

# Function to tokenize text
def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


# Function to build vocabulary
def build_vocab(corpus):

    vocab = set()

    for doc in corpus:
        vocab.update(tokenize(doc))

    return sorted(vocab)


# Function to create Bag of Words Vector
def bow_vector(doc, vocab):

    counts = Counter(tokenize(doc))

    return [counts.get(word, 0) for word in vocab]


# -----------------------------
# Main Program
# -----------------------------

corpus = [
    "I love NLP!!",
    "I love Python.",
    "Python is Amazing!!!"
]

# Build Vocabulary
vocab = build_vocab(corpus)

print("Vocabulary:")
print(vocab)

print("\nBag of Words Matrix:")

for doc in corpus:
    print(doc)
    print(bow_vector(doc, vocab))
    
#*----------------------------------------------------------------------------*
#Implemetation using Scikit 
from sklearn.feature_extraction.text import CountVectorizer

corpuss = [
    "I love NLP!!",
    "I love Python.",
    "Python is Amazing!!!"
]

vectorizer = CountVectorizer()

bow_matrix = vectorizer.fit_transform(corpuss)

print("Vocabulary")
print(vectorizer.vocabulary_)

print("\nBag of Words Matrix")
print(bow_matrix.toarray())