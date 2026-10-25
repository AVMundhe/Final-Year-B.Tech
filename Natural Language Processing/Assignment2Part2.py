import re
import math

documents = [
    "I love NLP!!",
    "I love Python.",
    "Python is Amazing!!!"
]

def tokenize(doc):
    words =re.findall(r"[a-z]+",doc.lower())
    return words

vocablury = {}
index = 0
for doc in documents:
    words = tokenize(doc)
    for word in words:
        if word not in vocablury:
            vocablury[word]=index
            index+=1

print(vocablury)

#Finding DF
freq_document = {}
for voc in vocablury:
    freq_document[voc]=0

for doc in documents:
    words = set(tokenize(doc))
    for word in words:
        freq_document[word]+=1
print("\nThe Document Frequecy:")
print(freq_document)

#Finding IDF
idf = {}
total_doc = len(documents)
for word in freq_document:
    word_freq = freq_document[word]
    idf[word]=math.log(total_doc/word_freq)

print("\nThe Inverse Document Frequecy:")
print(idf)