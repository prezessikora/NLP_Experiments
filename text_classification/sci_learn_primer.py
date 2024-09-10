#Just some intro 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../TextFiles/smsspamcollection.tsv',sep='\t')

print(df.head())

#count empty values
print(df.isnull().sum())

#some other interesing dataframe info

print(len(df))

print(df['label'].unique())

print(df['label'].value_counts())

#lets fit a random model



# just grab length and punct features that are numbers

X = df[['length','punct']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

from sklearn import metrics

predict = lr.predict(X_test)

print(metrics.confusion_matrix(y_test,predict,labels=['ham','spam']))
print(pd.DataFrame(metrics.confusion_matrix(y_test,predict,labels=['ham','spam']),index=['ham','spam'], columns=['ham','spam']))
print(metrics.classification_report(y_test,predict))
print(metrics.accuracy_score(y_test,predict))


X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


X_train_counts = cv.fit_transform(X_train)

print(X_train_counts.shape)
print(X_train.shape)

assert X_train_counts.shape[0] == X_train.shape[0]

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_counts)

assert X_train_tfidf.shape[0] == X_train.shape[0]