# Parts of Speech analysis using spacy and example text
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy import displacy

def print_sentence(s):
    for e in s:
        print(f'{e.text:10} {e.pos_:10} {e.tag_:10} {spacy.explain(e.tag_)}')

# Return POS counts with names
def counts(doc):
    stats = doc.count_by(spacy.attrs.POS)    
    stats = [ (k,doc.vocab[k].text,v) for k, v in stats.items()]
    return stats

# Count % of given POS
def percentage(doc,pos):
    stats = doc.count_by(spacy.attrs.POS)    
    return round(stats[pos] / len(list(doc)),2)


# Return sentences with named entities and the rest
def sentences_with_ents(doc):
    result = []
    result_no = []
    for s in doc.sents:        
        if len(list(filter(lambda e: e.start >= s.start and e.end <= s.end,doc.ents))) > 0:
            result.append(s)
        else:
            result_no.append(s)
    return result, result_no

# ------------ #

with open('../TextFiles/peterrabbit.txt') as f:
    doc = nlp(f.read())

sents = list(doc.sents)

print('Second sentence')
print_sentence(sents[2])


print('Parts of speech stats:')
for e in counts(doc):
    print(f'id: {e[0]:5} {e[1]:10} {e[2]:5}')


print(f'% of  NOUNS: {percentage(doc,92)}')

print('Dep graph for 2nd sentence')
#displacy.serve(sents[2],style='dep',auto_select_port=True )


for e in doc.ents[:2]:
    print(f'{e.text}  : {e.label_} : {spacy.explain(e.label_)}')


list_of_sents, no_ents = sentences_with_ents(doc)

print(f'Count of sentences with named entities: {len(list_of_sents)}')
print(f'Count of sentences with NO named entities: {len(no_ents)}')

# Alternatively 
list_of_sents = [nlp(s.text) for s in doc.sents]
list_ners =  [s for s in list_of_sents if s.ents]
print(len(list_ners))


print(len([s for s in doc.sents if s.ents]))

# #displacy.render(no_ents[2],style='ent')
# #displacy.render(list_of_sents[0],style='ent')