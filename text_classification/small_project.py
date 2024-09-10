import spacy
import numpy as np
from scipy import spatial 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nlp = spacy.load('en_core_web_lg')

def cosine_similarity(v1,v2):
    return 1 - spatial.distance.cosine(v1, v2)



def sentiment_of_text(text):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
                                 
    if sentiment['compound'] > 0:
        return 'Pos'
    elif sentiment['compound'] < 0:
        return 'Neg'
    else:
        return 'Neu'

def word_vector_add(w1, w2):
    v1 = nlp.vocab[w1].vector
    v2 = nlp.vocab[w2].vector

    r = v1 - v2
    all = []
    for id in nlp.vocab.vectors:        
        w = nlp.vocab[id]
        if w.has_vector and w.is_lower and w.is_alpha:
            all.append((w,cosine_similarity(r,w.vector)))    
    similar_words = sorted(all,key=lambda e: -e[1])
    print(f'> TOP SIMILAR WORDS of {w1} + {w2} :')
    for r in similar_words[:5]:
        print(f'{r[0].text:10} {r[1]:5}')



print(sentiment_of_text('This was good movie.'))
print(sentiment_of_text('This was terrible week. I hate to spend time like that. '))
word_vector_add('car','fast')
word_vector_add('wolf','dog')



