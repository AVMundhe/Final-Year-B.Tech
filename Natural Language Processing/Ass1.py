import nltk
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

text = """
Students are studying Natural Language Processing.
They are building useful NLP applications!
"""

#whitespace word_tokenize

whitespace=text.split()

#Word word_tokenize
tokens = word_tokenize(text)

#stop word removal
stopword = set(stopwords.words("english"))

filtere = []
for word in tokens:
    if word.lower() not in stopword:
        filtere.append(word)

#Stemming
stemmer = PorterStemmer()
stem_words = []
for word in filtere:
    stem_word = stemmer.stem(word)
    stem_words.append(stem_word)

#Lemmatization
lemmitizer = WordNetLemmatizer()
lemmatized_words = []

for word in filtere:

    lemmatized_word = lemmitizer.lemmatize(word)

    lemmatized_words.append(lemmatized_word)

#POS Tagging
pos_tags = pos_tag(tokens)

#Using SpaCy
spacy_pos = []

for token in doc:

    spacy_pos.append((token.text, token.pos_))

print("\nspaCy POS Tags:")
print(spacy_pos)


# spaCy Lemmas

spacy_lemmas = []

for token in doc:

    spacy_lemmas.append((token.text, token.lemma_))

print("\nspaCy Lemmas:")
print(spacy_lemmas)
