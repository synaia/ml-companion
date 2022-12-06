import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from ch08.cleaning_data import tokenizer, tokenizer_porter

df = pd.read_csv('../dataset/imdb/movie_data_cleaned.csv', encoding='utf-8')
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

stop = stopwords.words('english')

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
small_param_grid = [
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
    },
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer],
        'vect__use_idf': [False],
        'vect__norm': [None],
        'clf__penalty': ['l2'],   # 'l1'
        'clf__C': [1.0, 10.0]
    }
]
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(solver='liblinear'))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)
print(f'\nBest parameter set: {gs_lr_tfidf.best_params_}')

print(f'CV Accuracy: {gs_lr_tfidf.best_score_:0.3f}')
clf = gs_lr_tfidf.best_estimator_
print(f'Test Accuracy: {clf.score(X_test, y_test)}')
