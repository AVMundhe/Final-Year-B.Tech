import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

text = "Aditya visited Pune and joined Microsoft."

doc = nlp(text)

# Print entities
for ent in doc.ents:
    print(ent.text, "→", ent.label_)

# Visualize entities
displacy.serve(doc, style="ent")