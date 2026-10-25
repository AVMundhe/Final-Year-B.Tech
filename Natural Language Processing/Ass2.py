import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "machine learning is useful",
    "machine learning is powerful",
    "deep learning is useful"
]

#BoW from Scratch
vocablury = []
for doc in documents:
    words = doc.lower().split()
    for word in words:
        if word not in vocablury:
            vocablury.append(word)

bow = []
for doc in documents:
    words = doc.lower().split()
    vector = []
    for word in vocablury:
        count = words.count(word)
        vector.append(count)
    bow.append(vector)

#Using Scikit learn
vectorizer = CountVectorizer()
x_bow = vectorizer.fit_transform(documents)
print("\nSklearn Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nSklearn Bag-of-Words:")
print(x_bow.toarray())

#TF-IDF from scratch

def calculate_tf(word,doc):
    words = doc.lower().split()
    word_count = words.count(word)
    total_words = len(words)
    return (word_count/total_words)

def caluclate_idf(word,documents):
    word_doc = 0
    for doc in documents:
        words = doc.lower().split()
        if word in words:
            word_doc = word_doc+1
    return math.log(len(documents)/word_doc)

def calculate_tf_idf(word,doc,documents):
    tf = calculate_tf(word,doc)
    idf = caluclate_idf(word,documents)
    return (tf*idf)
    

tfidf = []
for doc in documents:
    vector = []
    for word in vocablury:
        value = calculate_tf_idf(word,doc,documents)
        vector.append(value)
    tfidf.append(vector)

print("\nTF-IDF from scratch:")

for vector in tfidf_vectors:
    print(vector)

#using scikit learn
tfidf_vectorizer = TfidfVectorizer()
x_tfidf = tfidf_vectorizer.fit_transform(documents)

print("\nSklearn Features:")
print(tfidf_vectorizer.get_feature_names_out())

print("\nSklearn TF-IDF:")
print(X_tfidf.toarray())

#Cosine similarity
similarity_matrix = cosine_similarity(x_tfidf)
print("\nCosine Similarity:")
print(similarity_matrix)



