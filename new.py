#pip3 install pickle-mixin
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle

#nltk.download("punkt")


dataset = pd.read_csv('dataset.csv')
dataset.head()
dataset.sort_values("Body", inplace = True)
dataset = dataset.drop(columns=["B","id"])

dataset.drop_duplicates(subset ="Body",keep = False, inplace = True)

def optimizasyon(dataset):
    dataset = dataset.dropna()

    stop_words = set(stopwords.words('turkish'))
    noktalamaIsaretleri = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
    stop_words.update(noktalamaIsaretleri)

    for ind in dataset.index:
        body = dataset['Body'][ind]
        body = body.lower()
        body = re.sub(r'http\S+', '', body)
        body = re.sub('\[[^]]*\]', '', body)
        body = (" ").join([word for word in body.split() if not word in stop_words])
        body = "".join([char for char in body if not char in noktalamaIsaretleri])
        dataset['Body'][ind] = body
    return dataset


dataset = optimizasyon(dataset)


pat1 = '@[^ ]+'
pat2 = 'http[^ ]+'
pat3 = 'www.[^ ]+'
pat4 = '#[^ ]+'
pat5 = '[0-9]'

combined_pat = '|'.join((pat1, pat2, pat3, pat4, pat5))


for ind in dataset.index:
    t = dataset['Body'][ind]
    t = t.lower()
    stripped = re.sub(combined_pat, '', t)
    tokens = word_tokenize(stripped)
    words = [x for x  in tokens if len(x) > 1]
    sentences = " ".join(words)
    negations = re.sub("n't", "not", sentences)
    dataset['Body'][ind] = negations

x = dataset['Body']
y = dataset['Label']


tv = TfidfVectorizer(stop_words='english', binary=False, ngram_range=(1,3))
x_tv = tv.fit_transform(x)
x_train_tv, x_test_tv, y_train_tv, y_test_tv = train_test_split(x_tv, y, test_size=0.2, random_state=0)

log_tv = LogisticRegression() 
log_tv.fit(x_train_tv,y_train_tv)

y_pred_tv = log_tv.predict(x_test_tv)
print(confusion_matrix(y_test_tv,y_pred_tv))
print(classification_report(y_test_tv,y_pred_tv))