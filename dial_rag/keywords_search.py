
from typing import List

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


LANG = "english"

stemmer = SnowballStemmer(LANG)
stopwords_list = stopwords.words(LANG)


def keywords_preprocess(text: str) -> List[str]:
    return [
        stemmer.stem(t.lower())
        for t in word_tokenize(text)
        if t not in stopwords_list
    ]
