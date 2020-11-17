# Emotions-Detection

### Dataset

The Data used to build this model contains some sentences along with the labels of emotions that are expressed through those sentences.

Three models were made using 3 different techniques:

1. Count Vectorizer and TF-IDF Vectorizer were used to convert each sentence into a vector with length equal to the vocabulary of the corpus. The Vectors were then fed to a Random Forest Classifier Model which classified a given text as one of the emotion. 
2. In the Neural Network Model, after tokenization, the text was converted to a sequence with padding done to make all the inputs of same length. The Inputs were then fed into a simple Neural Network Model after being converted into word embeddings using the Embedding() layer. 
3. The pretrained BERT model was used to train the classifier on the training data. This model was the best performing out of all the three models.

### Preprocessing

1. Removal of all the symbols using regex
2. Removal of stopwords
3. Lemmatization of all the words left after removal of stopwords.
4. Converting all letters to lowercase before tokenization. 

### Performance

Random Forest Classifier: 87.32%
Neural Network Classifier: 85.98%
BERT Training Model: 91%

### References:

[Dataset](https://www.kaggle.com/c/sa-emotions)

[Blog on Emotion Detection and Sentiment Analysis](https://medium.com/neuronio/from-sentiment-analysis-to-emotion-recognition-a-nlp-story-bcc9d6ff61ae)

[BERT Model Sentiment Analysis Video Tutorial](https://www.youtube.com/watch?v=8N-nM3QW7O0)
