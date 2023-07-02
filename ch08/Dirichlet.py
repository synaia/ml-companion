import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv('../dataset/imdb/movie_data_cleaned.csv', encoding='utf-8')
count = CountVectorizer(stop_words='english', max_df=0.1, max_features=5000)
X = count.fit_transform(df['review'].values)

lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')
X_topics = lda.fit_transform(X)

n_top_words = 5
feature_names = count.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_, 1):
    print(f'Topic {topic_idx}')
    print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

horror = X_topics[:, 5].argsort()[::-1]
for iter_idx, movie_idx in enumerate(horror[:3]):
    print(f'\nHorror movie # {(iter_idx + 1)}:')
    print(df['review'][movie_idx][:300], '...')


print()