from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import timeit

tokenizer = AutoTokenizer.from_pretrained('Jean-Baptiste/roberta-large-ner-english')
model = AutoModelForTokenClassification.from_pretrained('Jean-Baptiste/roberta-large-ner-english')

ner = pipeline('ner', model=model, tokenizer=tokenizer)

text_eng = "Hi there, my name is Terence Collin, I live at Nogent-sur-Seine, I'm French, I'm 33 years old and I'm working at Bluesoft as an AI "\
       "consultant. I studied Arts at university in St-Denis for 4 years. I never did any formation in the first-aid field"
text_fr = "Bonjour, je m’appelle Terence Collin, j’habite à Nogent-sur-Seine, je suis français, j’ai 33 ans et je\
    travaille a Bluesoft comme consultant en IA. J’ai étudié les Arts plastiques à l’université de St-Denis pendant\
    4 ans. Je n’ai jamais effectué de formation dans le domaine du secourisme"

text = text_eng

start = timeit.default_timer()

ner_list = ner(text)

for ent in ner_list:
    print(ent)

print("\n")

stop = timeit.default_timer()
print('Time: ', stop - start)