#import libraries
import joblib
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
import demoji
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI

#initializing the app
app = FastAPI()



model = joblib.load('tweet_classifier.joblib')
vectorizer=joblib.load('count_vectorizer.pkl')
#preprocessing data 
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


def classify_message(model, message):
    message = preprocessor(message)
    message = vectorizer.transform([message])
    message = message.toarray()  # Convert sparse matrix to dense array
    label = model.predict(message)
    label=int(label)
    #spam_prob = model.predict_proba(message)
    return {'label': label}

#defining simple Get request 

@app.get('/spam_detection_query/')
async def detect_spam_query(message: str):
	return classify_message(model, message)

@app.get('/spam_detection_path/{message}')
async def detect_spam_path(message: str):
	return classify_message(model, message)


@app.get('/')
def get_root():
    return {'message': 'Welcome to the tweet classifier API'}