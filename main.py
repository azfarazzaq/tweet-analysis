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


#call the saved model and vectorizer 
model = joblib.load('tweet_classifier.joblib')
vectorizer=joblib.load('count_vectorizer.pkl')
#preprocessing the text data (function)
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

#define a function to send the preprocessed data in to the model to predict outcome 

def classify_message(model, message):
    message = preprocessor(message)
    message = vectorizer.transform([message])
    message = message.toarray()  # Convert to  array
    label = model.predict(message) #obtain the relevant label 
    label=int(label) #convert in to integer form 
    tweet_prob = model.predict_proba(message) # predicts the probability for each class
    if label==1:
          label= "negative tweet"
          probability= tweet_prob[0][1]
    else:
          label= "positive tweet"
          probability=tweet_prob[0][0]
    return {'label': label,'probability': probability}#probability of getting the predicted output 
#Defining the tweet Detection GET Request
# can supply the inputs of a machine learning model to a GET request using query parameters or path variables

#Using Query Parameters 
#eg: 127.0.0.1.8000/tweet_detection_query/?message=’i love pancakes’ 
@app.get('/tweet_detection_query/')
async def detect_tweet_query(message: str):
	return classify_message(model, message)

#using path variable 
# eg: 127.0.0.1.8000/tweet_detection_query/i hate usa
@app.get('/tweet_detection_path/{message}')
async def detect_tweet_path(message: str):
	return classify_message(model, message)

# define the output for a simple get request. will return a JSON output with a welcome message.
@app.get('/')
def get_root():
    return {'message': 'Welcome to the tweet classifier API'}