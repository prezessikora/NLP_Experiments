# An example using NNMF Non-negtive matrix factorization on a colletion of quora questions categorizing them

# Number of expected components (categories/topcis) to identify
k = 20


import pandas as pd
df = pd.read_csv('quora_questions.csv')
print('The dataset head: ')
print(df.head())

# Peform TFIDF vectorizaion on the documents

from sklearn.feature_extraction.text import TfidfVectorizer

# max_df the word can not exceed % of questions  = remove common words
# min_df the word has to be in at least % of questions = remove specific words


tfidf = TfidfVectorizer(max_df=0.99,min_df=5,stop_words='english')

documents = tfidf.fit_transform(df['Question'])
print(f'Questions encoded on vector lenght: {documents.shape[1]}')

# tfidf.get_feature_names_out() contains the encoded words



# NMF

from sklearn.decomposition import NMF
nmf = NMF(n_components=k,random_state=42)

result = nmf.fit_transform(documents)
# Now result contains a vector for each document of probable topics

# nmf.components_ contains the probability of word in topic, lets display top 15 wrods for each topic

for topic_index in range(len(nmf.components_)):    
    print(f'-NEW TOPIC- {topic_index}')
    topic_words = nmf.components_[topic_index]    
    top_words = map(lambda word_index: tfidf.get_feature_names_out()[word_index],topic_words.argsort()[-15:])
    print(list(top_words))


# Add topic id and name to the df

df['Topic'] = result.argmax(axis=1)

id_name = {i: f'topic {i}' for i in range(k)}
id_name[8] = 'weight loss'

# Add the topic label

df['Topic'] = df['Topic'].map(id_name)

print(df.head())