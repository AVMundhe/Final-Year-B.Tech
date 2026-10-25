# ==========================================
# WORD2VEC
# ==========================================

from gensim.models import Word2Vec

sentences = [
    ["machine", "learning", "is", "useful"],
    ["machine", "learning", "is", "powerful"],
    ["deep", "learning", "is", "useful"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["neural", "networks", "are", "powerful"],
    ["python", "is", "useful", "for", "machine", "learning"],
    ["data", "science", "uses", "machine", "learning"]
]


# Train Word2Vec

model = Word2Vec(
    sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1
)


# Vocabulary

print("Vocabulary:")
print(model.wv.index_to_key)


# Word vector

print("\nVector for machine:")
print(model.wv["machine"])


# Similarity

print("\nSimilarity:")
print(
    model.wv.similarity(
        "machine",
        "learning"
    )
)


# Most similar words

print("\nMost Similar Words:")

print(
    model.wv.most_similar(
        "learning",
        topn=3
    )
)


# ==========================================
# PCA
# ==========================================

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

words = model.wv.index_to_key

vectors = []

for word in words:

    vectors.append(
        model.wv[word]
    )


pca = PCA(n_components=2)

pca_vectors = pca.fit_transform(vectors)


plt.figure(figsize=(8, 6))

for i in range(len(words)):

    x = pca_vectors[i][0]
    y = pca_vectors[i][1]

    plt.scatter(x, y)

    plt.text(
        x,
        y,
        words[i]
    )

plt.title("Word2Vec using PCA")

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show()


# ==========================================
# t-SNE
# ==========================================

from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=3
)

tsne_vectors = tsne.fit_transform(vectors)


plt.figure(figsize=(8, 6))

for i in range(len(words)):

    x = tsne_vectors[i][0]
    y = tsne_vectors[i][1]

    plt.scatter(x, y)

    plt.text(
        x,
        y,
        words[i]
    )

plt.title("Word2Vec using t-SNE")

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.show()


# ==========================================
# GLOVE
# ==========================================

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


glove = {}

with open(
    "glove.6B.100d.txt",
    encoding="utf8"
) as file:

    for line in file:

        values = line.split()

        word = values[0]

        vector = np.asarray(
            values[1:],
            dtype="float32"
        )

        glove[word] = vector


# Check vector

print("\nGloVe Vector for king:")
print(glove["king"])


# ==========================================
# GLOVE ANALOGY
# king - man + woman = queen
# ==========================================

result = (
    glove["king"]
    - glove["man"]
    + glove["woman"]
)


similar_words = []

for word, vector in glove.items():

    similarity = cosine_similarity(
        result.reshape(1, -1),
        vector.reshape(1, -1)
    )[0][0]

    similar_words.append(
        (word, similarity)
    )


similar_words.sort(
    key=lambda x: x[1],
    reverse=True
)


print("\nGloVe Analogy Results:")

print(similar_words[:10])