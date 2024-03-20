import spacy
from spacy import displacy
import timeit

ner = spacy.load("en_core_web_trf")

text_eng = "Hi there, my name is Terence Collin, I live at Nogent-sur-Seine, I'm French, I'm 33 years old and I'm working at Bluesoft as an IA "\
       "consultant. I studied Arts at university in St-Denis for 4 years. I never did any formation in the first-aid field"
text_fr = "Bonjour, je m’appelle Terence Collin, j’habite à Nogent-sur-Seine, je suis français, j’ai 33 ans et je\
    travaille a Bluesoft comme consultant en IA. J’ai étudié les Arts plastiques à l’université de St-Denis pendant\
    4 ans. Je n’ai jamais effectué de formation dans le domaine du secourisme"

text = text_fr

print(text + "\n")

start = timeit.default_timer()

print("NER with Spacy nlp:")
text_ner = ner(text)

for ent in text_ner.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

stop = timeit.default_timer()

print("\n")

print('Time: ', stop - start)

# print("\n")

# print(spacy.explain("MISC"))
# print(spacy.explain("PER"))
# print(spacy.explain("LOC"))
# print(spacy.explain("0RG"))
#
# displacy.serve(text_ner, style="ent")

