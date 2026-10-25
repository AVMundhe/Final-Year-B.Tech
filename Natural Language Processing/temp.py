# import nltk
# from nltk.tokenize import sent_tokenize,word_tokenize
# nltk.download('punkt')  #cut the sentence
# nltk.download('punkt_tab')  #contains rule of grammar
# nltk.download('stopwords')  #it tells which word to give importance to
# nltk.download('wordnet') #dictionary of english word
# nltk.download('omi-1.4')  #translate wordnet words into other language
# nltk.download('averaged_perceptron_tagger') #part of speech tagger tag each word as pronoun,noun etc



# text=("माझे नाव हर्ष आहे")

# sentences=sent_tokenize(text)
# print("Sentences: ",sentences)


# tokens=word_tokenize(text)
# print("Tokens: ",tokens)
# ---------------------------------------------------------
# Assignment 2 

import re

from collections import Counter

#1. Tokenizer: Lowercase + extract alphabetic words def tokenize(text):

return re.findall(r"[a-z]+", text.lower())

#2. Build vocabulary across the whole corpus

def build vocab(corpus):

vocab= set()

for doc in corpus:

vocab.update(tokenize(doc)) return sorted(vocab)

#3. Represent one document as a count vector def bow vector(doc, vocab): counts Counter(tokenize(doc)) return [counts.get(w, 0) for w in vocab]

vocab = build vocab(corpus)

bow matrix = [bow vector(d, vocab) for d in corpus]