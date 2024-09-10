
import spacy
nlp = spacy.load('en_core_web_lg')

def printT(tokens):
    print('='*10)
    for token1 in tokens:
        for token2 in tokens:
            print(f'{token1.text:10} {token2.text:10}, {token1.similarity(token2)}')


printT(nlp(u'lion cat pet'))
printT(nlp(u'like love hate'))

king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

import numpy as np

def similar(v1,v2):
    return np.abs( np.sqrt(np.sum(v1 ** 2)) - np.sqrt(np.sum(v2 ** 2)))

from scipy import spatial
cosine_similarity = lambda v1, v2: 1 - spatial.distance.cosine(v1,v2)

new_vector = king - man + woman

all = []

# Watch out nlp.vocab contains just a cache of recenly used tokens
for id in nlp.vocab.vectors:
    w = nlp.vocab[id]
    if w.has_vector and w.is_lower and w.is_alpha:
        all.append((w,cosine_similarity(new_vector,w.vector)))

all = sorted(all,key=lambda item: -item[1])

l = ( [(w[0].text,w[1]) for w in all[:20]])
for e in l:
    print(f'{e[0]:15} {round(e[1],4):5}')
