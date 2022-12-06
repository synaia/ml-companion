import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining, the weather is sweet',
                 'and one and one is two'
                 ])
bag = count.fit_transform(docs)
print(count.vocabulary_)

# Inverse Document Frequency
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
print('\n')
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
