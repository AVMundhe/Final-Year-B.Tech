import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

glove = {}
with open("glove.6B.100d.txt", encoding="utf8") as file:
    for line in file:
        values = line.split()
        if len(values) < 2:
            continue
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        glove[word] = vector

words = ["king", "queen", "man", "woman", "computer", "programming"]

for word in words:
    if word in glove:
        print(word)
        print("vector:", glove[word])
        print("Dimensions:", len(glove[word]))

print()
result = glove["king"] - glove["man"] + glove["woman"]
print("King-Man+Woman = ",result)
print()

similarity1 = cosine_similarity(
    glove["computer"].reshape(1, -1),
    glove["programming"].reshape(1, -1)
)
print("Similarity1:", similarity1)

similarity2 = cosine_similarity(
    glove["king"].reshape(1, -1),
    glove["queen"].reshape(1, -1)
)
print("Similarity2:", similarity2)