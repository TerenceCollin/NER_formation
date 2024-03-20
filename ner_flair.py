from flair.data import Sentence
from flair.models import SequenceTagger
import timeit

# load tagger
tagger = SequenceTagger.load("flair/ner-english-large")

# make example sentence
sentence = Sentence("Hi there, my name is Terence Collin, I live at Nogent-sur-Seine, I'm French, I'm 33 years old "\
                    "and I'm working at Bluesoft as an AI consultant. I studied Arts at university in St-Denis "\
                    "for 4 years. I never did any formation in the first-aid field")

start = timeit.default_timer()

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print('The following NER tags are found:')
# iterate over entities and print
for entity in sentence.get_spans('ner'):
    print(entity)

print("\n")

stop = timeit.default_timer()
print('Time: ', stop - start)