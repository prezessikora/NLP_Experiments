import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')

def show_lemmas(text):
    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.dep_:<{22}} {token.lemma_}')

def print_matches(found_matches):
    sents = [sent for sent in doc.sents] # to get the whole sentences

    for match_id, start, end in found_matches:
        string_id = nlp.vocab.strings[match_id]  # get string representation
        span = doc[start:end]                    # get the matched span        
        
        print("Match: ",match_id, string_id, start, end, span.text)
        print(f'Context: {doc[start-5:end+5]}')

        for s in sents:
            if end < s.end:
                print(f'Full sentence: {s}')
                break

with open('owlcreek.txt') as f:
    doc = nlp(f.read())
    print(len(doc))
    # for token in doc:
    #     print(token.text,token.pos_,token.dep_,sep=' : ')
    
    sents = [sent for sent in doc.sents]
    print(f'Sentences count: {len(sents)}')
    for s in sents:
        print(f'{s.start} : {s.end}')
    print(sents[1])
    show_lemmas(sents[1])

    # Search for pattern  "swimming vigorously"

    p1 = [{'LOWER':'swimming'},{'IS_SPACE': True},{'LOWER':'vigorously'}]
    matcher = Matcher(nlp.vocab)
    matcher.add('P',[p1])

    found_matches = matcher(doc)
    print_matches(found_matches)

    

