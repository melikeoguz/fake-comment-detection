import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer




def optimizasyon(metin):

    stop_words = set(stopwords.words('turkish'))
    noktalamaIsaretleri = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '–', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
    stop_words.update(noktalamaIsaretleri)

    body = metin
    body = body.lower()
    body = re.sub(r'http\S+', '', body)
    body = re.sub('\[[^]]*\]', '', body)
    body = (" ").join([word for word in body.split() if not word in stop_words])
    body = "".join([char for char in body if not char in noktalamaIsaretleri])
    return body


LogisticRegressionModel = pickle.load(open("egitilmis_model", 'rb'))

tfIdf = pickle.load(open("vektorlestirici", 'rb'))



def tahminEt(testData):
    testData = optimizasyon(testData)
    vektorlestirilmis_test_verisi = tfIdf.transform([testData])


    print(vektorlestirilmis_test_verisi)
    #4 yıldız çünkü jelatini açılmıştı jelatininin açılması dışında her şey yolundaydı
    tahminSonuc = LogisticRegressionModel.predict(vektorlestirilmis_test_verisi)

    print(tahminSonuc)



inp = input("Yorum giriniz")


tahminEt(inp)