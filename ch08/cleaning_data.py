import re
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text


porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


df = pd.read_csv('../dataset/imdb/movie_data.csv', encoding='utf-8')
df['review'] = df['review'].apply(preprocessor)



print()