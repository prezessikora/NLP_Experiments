import spacy
from spacy.language import Language # for the custom pipeline stage

nlp = spacy.load('en_core_web_sm')

# print the document tokens
def print_doc(doc):
    for s in doc.sents: 
        print(s)
    for token in doc:
        print(f'{token.text:10} {token.is_sent_start:10}')


# custom sentence split on ;
@Language.component("my_component")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i+1].is_sent_start = True
            print(doc[token.i+1])
    return doc

doc = nlp('Sentence one; Sentence two; Leaders are good. - Peter Drucker')
print(nlp.pipe_names)
print_doc(doc)

print("-- With custom sentence split --")

nlp.add_pipe('my_component',before='parser')
print(nlp.pipe_names)
doc = nlp('Sentence one; Sentence two; Leaders are good. - Peter Drucker')

print_doc(doc)


for s in doc.sents: 
    print(s)