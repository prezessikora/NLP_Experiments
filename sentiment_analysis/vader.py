import nltk 
nltk.download('vader_lexicon')

import pandas as pd

df = pd.read_csv('../TextFiles/amazonreviews.tsv',sep='\t')
print(df.head())
df.dropna(inplace=True)
blanks = []
for i,l,review in df.itertuples():
    if type(review) == str and review == '':
        blanks.append(i)
df.drop(blanks,inplace=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

sentiment = sid.polarity_scores((df.iloc[0])['review'])

df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda score: score['compound'])
df['label_vader'] = df['compound'].apply(lambda score: 'pos' if score > 0.0 else 'neg')

from sklearn.metrics import accuracy_score, confusion_matrix
acuracy = accuracy_score(df['label'],df['label_vader'])
print(acuracy)
cm = confusion_matrix(df['label'],df['label_vader'])
print(cm)

