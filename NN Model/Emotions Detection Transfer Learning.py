from keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import contractions
import pandas as pd
import numpy as np
import re
import nltk

text_dict = {'text':[], 'happy':[], 'sadness':[], 'fear':[], 'anger':[], 'love':[], 'surprise':[]}
emotions = ['happy', 'sadness', 'anger', 'fear', 'love', 'surprise']
def read_data(url):
    # Reading data from CSV file and saving it into dataframe "train"
    train = pd.read_csv(url)

    # Changing the data into dictionary for better handling
    for index, row in train.iterrows():
        sentence = row['Text']
        emotion = row['Emotion']
        text_dict['text'].append(sentence)
        # The loop below gives value 1 to the correct emotion and 0 to others making
        # a one hot vector for the emotions.
        for e in emotions:
            if emotion == e:
                text_dict[emotion].append(1)
            else:
                text_dict[e].append(0)
    # All the data was being stored in a dictionary till now, converting that to dataframe again
    train = pd.DataFrame(text_dict)
    return(train)

def preprocess_dataframe(df):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    for index, row in df.iterrows():
        filter_sentence = ''
        sentence = row['text']
        
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
        
        # The preprocessed sentence now replaces the original sentence
        df.loc[index, 'text'] = filter_sentence
    df = df[['text', 'happy', 'sadness', 'fear', 'anger', 'love', 'surprise']]
    return(df)

def tokenize(train):
    Y = train[['happy', 'sadness', 'fear', 'anger', 'love', 'surprise']]
    Y = np.array(Y)
    X = np.array(train['text'])
    
    # Tokenizer is used to count all unique words and to label them with a certain number
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)

    # After converting all texts from X to sequences, they ae stored in X_new
    X_new = []
    for line in X:
        token_list = tokenizer.texts_to_sequences([line])[0]
        X_new.append(token_list)
    max_sequence_len = max(max([len(x) for x in X_new]), 60)
    
    # Padding is done in order to make all the inputs of same size
    input_sequences = np.array(pad_sequences(X_new, maxlen=max_sequence_len, padding='pre'))
    total_words = len(tokenizer.word_index) + 1

    # Final tokenized inputs are stored in X which is then returned
    X = input_sequences
    return(X, Y, total_words, tokenizer, max_sequence_len)

def build_model():
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

def train_model(model, X, Y, epochs):
    model.fit(X, Y, epochs = 10)
    return(model)

def give_model(url):
    train = read_data(url)
    train = preprocess_dataframe(train)
    X, Y, total_words, tokenizer, max_sequence_len = tokenize(train)
    model = build_model()
    epochs = 30
    model = train_model(model, X, Y, epochs)
    return(model, tokenizer, max_sequence_len)


model, tokenizer, max_sequence_len = give_model("https://raw.githubusercontent.com/ishantjuyal/Emotions-Detection/main/Data/Emotion_final.csv")


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
    return(emotion)

print(detect("I am very sad"))

# # Save the trained model so that it can be used later

# # Serialize to JSON
# json_file = model.to_json()
# with open("model.json", "w") as file:
#    file.write(json_file)
# # Serialize weights to HDF5
# model.save_weights("model.h5")


