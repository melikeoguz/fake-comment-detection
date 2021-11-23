#pip3 install keras
#pip3 install tensorflow
#pip3 install nltk
#pip3 install gensim

from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import gensim

import nltk
from nltk.corpus import stopwords
import re

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import os

dataset = pd.read_csv('dataset.csv')
dataset.head()
dataset.sort_values("Body", inplace = True)
dataset = dataset.drop(columns="B")

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


X = dataset.loc[:,"Body"]
y = dataset.loc[:,"Label"]

print(X)


print(y)

X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size = 0.2, random_state = 28) 

# Word2Vec yani dökümanı vektöre çevirecek gensim modeli için gerekli parametreler
maxmesafe = 2 #Bir cümle içindeki mevcut ve tahmin edilen kelime arasındaki maksimum mesafe
minfrekans = 1 #Toplam sıklığı bundan daha düşük olan kelimeleri göz ardı eder
vektor_boyut = 200 #Öznitelik vektörlerinin boyutluluğu
maxlen = 1000 #Bir  metninin maksimum uzunluğu


X_egitim_splited = [metin.split() for metin in X_egitim]
w2v_model = gensim.models.Word2Vec(sentences = X_egitim_splited, vector_size=vektor_boyut, window = maxmesafe,  min_count = minfrekans)                                          
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_egitim_splited)
X_egitim_tok = tokenizer.texts_to_sequences(X_egitim_splited)
kelime_index = tokenizer.word_index
kelime_sayi = len(kelime_index) + 1
X_egitim_tok_pad = pad_sequences(X_egitim_tok, maxlen=maxlen)


print('Sözlük boyutu: ', kelime_sayi)

matris = np.zeros((kelime_sayi, vektor_boyut))
for kelime, i in kelime_index.items():
    matris[i] = w2v_model.wv[kelime]

model = Sequential()
model.add(Embedding(matris.shape[0], 
                    output_dim=matris.shape[1],
                    weights=[matris], 
                    input_length=maxlen, 
                    trainable=False))
model.add(LSTM(units=32))   
model.add(Dense(1, activation='sigmoid'))   # Aktivasyon fonksiyonu olarak "Sigmoid" i seçiyoruz
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])    # Optimizer parametresi olarak "adam", loss için ise "binary_crossentropy" seçiyoruz.

model.summary()


model.fit(X_egitim_tok_pad, y_egitim, validation_split=0.2, epochs=30, batch_size = 64, verbose = 1)

model.save('egitilmis_model.h5')

print("Model eğitildi ve kayıt edildi !")