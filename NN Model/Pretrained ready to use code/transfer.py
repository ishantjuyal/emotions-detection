from keras.models import model_from_json
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions
import pandas as pd
import numpy as np
import pickle
import nltk
import re

def load_model():
    # Load the saved model

    # load json and create model
    file = open('model.json', 'r')
    model_json = file.read()
    file.close()
    model = model_from_json(model_json)
    # load weights
    model.load_weights("model.h5")

    # Code is then compiled with appropriate loss and optimizer based on type of output
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    return(model)

model = load_model()
max_sequence_len = 60

def preprocess_sentence(sentence):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    filter_sentence = ''

    # Contractions will convert short form of the words into their original form
    sentence = contractions.fix(sentence)
    
    # Cleaning the sentence with regex
    sentence = re.sub(r'[^\w\s]', '', sentence)
    
    # Tokenization here means that words in a sentence are separated and stored in a list
    words = nltk.word_tokenize(sentence)

    # Removal of stopwords
    words = [w for w in words if not w in stop_words]
    
    # Lemmatization is done so that the words are converted to their original form
    for words in words:
        filter_sentence = filter_sentence  + ' ' + str(lemmatizer.lemmatize(words)).lower()
    
    # This final preprocessed sentence is then returned so that model can predict using this preprocessed sentence as input
    return(filter_sentence)


def detect(text):
    # The input sentence is first preprocessed using the function below
    text = preprocess_sentence(text)

    # loading tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # The preprocessed sentence is now tokenized and converted into array of numbers
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # The below two steps are done in place of padding.
    zeros = max(0, max_sequence_len - len(token_list))
    token_list = [0]*zeros + token_list
    
    # The token list is now reshaped so that it can be used as an input for the model 
    token_list = np.array(token_list).reshape(1, max_sequence_len)

    # The model predicts the most likely output of the input as an integer
    index = np.argmax(model.predict(token_list), axis = -1)[0]
    
    # The list given below is such that the output given by the model is the index at which the final emotion is present
    emotions = ['happy', 'sadness', 'fear', 'anger', 'love', 'surprise']
    emotion = emotions[index]
    return("The sentence has emotion: " + emotion)



