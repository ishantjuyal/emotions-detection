from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from ktrain import text
import pandas as pd
import numpy as np
import ktrain
import nltk
import re

"""
If you have any problem in importing ktrain, install ktrain first using
'pip install ktrain' in the command prompt, or '!pip install' ktrain in notebooks
"""

def get_predictor(url):
    # Read data from CSV and save it to dataframe "train"
    train = pd.read_csv(url)

    # Converting train into a data type that would be useful for ktrain
    (X_train, y_train), (X_test, y_test), preprocess = text.texts_from_df(train_df= train, 
                                                                      text_column = 'Text',
                                                                      label_columns = 'Emotion',
                                                                      maxlen = 60,
                                                                      preprocess_mode = 'bert')
    
    # We actually want to train the model on the whole data, so we will just concatenate
    # train and test data to convert all the data to appropriate data type
    X_new = [[],[]]
    X_new[0] = np.concatenate((X_train[0], X_test[0]))
    X_new[1] = np.concatenate((X_train[1], X_test[1]))
    y_new = np.concatenate((y_train, y_test))
    
    # Building the model using BERT
    model = text.text_classifier(name= 'bert',
                             train_data= (X_new, y_new),
                             preproc = preprocess)
    # Preparing learner and then training it over the complete dataset for 1 epoch
    learner = ktrain.get_learner(model = model,
                             train_data = (X_new, y_new),
                             val_data = (X_test, y_test),
                             batch_size = 32)
    learner.fit_onecycle(lr = 2e-5, epochs = 1)

    # Saving the predicting part of the model to predictor
    predictor = ktrain.get_predictor(learner.model, preproc= preprocess)
    return(predictor)

predictor = get_predictor("https://raw.githubusercontent.com/ishantjuyal/Emotions-Detection/main/Data/Emotion_final.csv")


def detect(sentence):
    data = [sentence]
    emotion = predictor.predict(data)
    return('You are feeling ' + emotion[0])


# Saving the predictor for later use
predictor.save('my_predictor')

"""
# reload the predictor
reloaded_predictor = ktrain.load_predictor('my_predictor')
"""
