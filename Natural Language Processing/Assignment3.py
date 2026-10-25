from gensim.models import Word2Vec

corpus = [
    ["smartphone","has", "good", "camera"],
    ["smartphone", "has", "large", "screen"],
    ["laptop", "has", "good", "battery"],
    ["laptop", "has", "large", "screen"]
]
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    window=4,
    min_count=1,
    workers=4,
    epochs=10
)

vector = model.wv["smartphone"]

print(vector)
print("Vector Size : ",len(vector))

similar_words = model.wv.most_similar("smartphone",topn=3)
print(similar_words)

