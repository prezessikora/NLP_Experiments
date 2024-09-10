import numpy as np 
import pandas as pd

df = pd.read_csv('moviereviews2.tsv',sep='\t')

print(df.head())

print('Checking and removing blanks ..')

print(df.isnull().sum())

print(len(df))

df.dropna(inplace=True)

for id,label,review in df.itertuples():
    if review.isspace():
        df.drop(id)

print(len(df))

print('Peek look at data ..')

print(df['label'].value_counts())

from sklearn.model_selection import train_test_split

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

assert(len(X_train) + len(X_test) == len(df))

# Create pipeline

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

pipeline = Pipeline([('tfidf',TfidfVectorizer()),('svc',LinearSVC())])

pipeline.fit(X_train,y_train)

predict = pipeline.predict(X_test)

# Calculate and print model metrics

from sklearn import metrics

print(metrics.confusion_matrix(y_test,predict))

labels = sorted(df.label.unique().tolist())

cm_df = pd.DataFrame(metrics.confusion_matrix(y_test,predict),index=['neg','pos'], columns=['neg','pos'])

print('==== Confusion Matrix === ')
print(cm_df)

print('==== Classification Report === ')
print(metrics.classification_report(y_test,predict))

