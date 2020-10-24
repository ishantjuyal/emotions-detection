#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import re
import nltk


# In[2]:


text_dict = {'text':[], 'emotion':[]}


# In[ ]:


# def get_csv_data(url):
#     df = pd.read_csv(url)
#     emotion_dict = {"happy": 1, "sad":2, "surprise":3, "fear":4, "disgust":5, "anger":6, "shame":7, "love":8, "neutral":9, "sadness":2, "joy":1}
#     for index, row in df.iterrows():
#         text = row["text"]
#         emotion = row["emotion"]
#         text_dict['text'].append(text)
#         text_dict['emotion'].append(emotion_dict[emotion])


# In[3]:


def get_data(url):
    f = open(url, 'r', encoding = "UTF-8")
    text = f.read()
    text_array = text.split('\n')
    emotion_dict = {"happy": 1, "sad":2, "surprise":3, "fear":4, "disgust":5, "anger":6, "shame":7, "love":8}
    for sentence in text_array[:-1]:
        start_index = sentence.index('>')
        end_index = sentence[start_index:].index("<")
        text = sentence[start_index + 1:end_index]
        emotion = sentence[1:start_index]
        text_dict['text'].append(text)
        text_dict['emotion'].append(emotion_dict[emotion])


# In[4]:


def get_data_2(url):
    f = open(url, 'r', encoding = "UTF-8")
    text = f.read()
    text_array = text.split('\n')
    emotion_dict = {"anger": 6, "joy":1, "fear":4, "sadness":2, "love":8, "surprise":3}
    for sentence in text_array:
        if ';' in sentence:
            a = sentence.split(';')
            text_dict['text'].append(a[0])
            text_dict['emotion'].append(emotion_dict[a[1]])


# In[5]:


get_data("No Cause.txt")
get_data_2("train.txt")
get_data_2("test.txt")
get_data_2("val.txt")


# In[6]:


train = pd.DataFrame(text_dict)
print(train.shape)


# In[7]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


# In[8]:


def preprocess(df):
    for index, row in df.iterrows():
        filter_sentence = ''
        sentence = row['text']
    
        # Cleaning the sentence with regex
        sentence = re.sub(r'[^\w\s]', '', sentence)

        # Tokenization
        words = nltk.word_tokenize(sentence)

        # Stopwords removal
        words = [w for w in words if not w in stop_words]
        
        for words in words:
            filter_sentence = filter_sentence  + ' ' + str(lemmatizer.lemmatize(words)).lower()
        
        df.loc[index, 'text'] = filter_sentence
    df = df[['text', 'emotion']]
    return(df)


# In[9]:


train.head()


# In[10]:


train = preprocess(train)
train.head()


# In[11]:


X = train['text']
Y = train['emotion']


# In[12]:


from sklearn.pipeline import Pipeline
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[13]:


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', ensemble.RandomForestClassifier()),
])


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)


# In[15]:


pipeline.fit(X_train, y_train)


# In[16]:


y_pred = pipeline.predict(X_test)


# In[17]:


y_test = np.array(y_test)


# In[18]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[19]:


def preprocess_sentence(sentence):
    
    sentence = re.sub(r'[^\w\s]', '', sentence)
    
    words = nltk.word_tokenize(sentence)
    
    words = [w for w in words if not w in stop_words]
    
    filter_sentence = ''
    for words in words:
        filter_sentence = filter_sentence  + ' ' + str(lemmatizer.lemmatize(words)).lower()
        
    return(filter_sentence)


# In[20]:


def detect(text):
    text = preprocess_sentence(text)
    emotion_label = pipeline.predict([text])
    label_to_emotion = {1: "happy", 2: "sad", 3: "surprised", 4: "fear", 5: "disgust", 6: "anger", 7: "shame", 8: "love", 9:"neutral"}
    return(label_to_emotion[emotion_label[0]])


# In[21]:


detect("It's okay to cry sometimes. I don't consider it as a sign of problem now")


# In[26]:


detect("It's cherry blossom season in Japan and I get out of work early and it's so sunny and cool, nothing can stop me today!")


# In[27]:


detect("My girlfriend broke up with me yesterday. I knew it was coming and thought I was ready but nope, I've been a disaster.")


# In[28]:


detect("I got in trouble at work and I feel this way. like i'm going to lose my job tomorrow")


# In[29]:


detect("My friend was murdered this weekend by another (now former) friend of mine.")


# In[30]:


detect("Mediocre. I'm overweight, jobless, and unmotivated. I don't want to do something, I want to want to do something.")


# In[31]:


detect("I've masturbated six times today, to horse porn. How do you think I'm feeling.")


# In[32]:


detect("I feel tired, sore, and lonely. I just wan't somebody to hold...")


# In[33]:


detect("My body keeps failing worse and worse so that makes me pretty sad, but I had the most delicious milk today and the weather was nice, and that makes me feel happy.")


# In[34]:


detect("Not great. I feel suicidal and I have no one to talk to.")


# In[35]:


detect("I took a look at my life today in nearly all aspects I could think of - relationships, career aspirations, schoolwork, friends, and my health. I came up disappointed on all fronts. So there's that.")


# In[36]:


detect("Happy, optimistic, inspired, enthusiastic, upbeat, silly, joyful!!")


# In[38]:


detect("I applied for a job and the interviewer was pretty nice, then I got a nice compliment here. So, its getting better")


# In[ ]:




