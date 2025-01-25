# CODTECH_DS_04
COMPANY   : CODTECH IT SOLUTIONS 
NAME      : DEEPSIKA A
INTERN ID : CT08FOT
DOMAIN    : DATA SCIENCE
DURATION  : 4 WEEKS
MENTOR    : NEELA SANTOSH
Create a natural language processor application to analyze and process text data using techniques such as tokenization, stemming, lemmatization, and stop-word removal for text preprocessing.
Implement advanced NLP models like word embeddings (e.g., Word2Vec, GloVe),

import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import gensim.downloader as api
import spacy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Tokenization
def tokenize_text(text):
    return word_tokenize(text)

# Remove stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

# Stemming
def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

# Lemmatization
def lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]
    # Sample dataset
data = [
    "I love programming with Python!",
    "Natural Language Processing is amazing.",
    "Text data can be processed efficiently with NLP techniques.",
    "Word embeddings are a great way to represent words in NLP tasks.",
    "I enjoy solving real-world problems using machine learning."
]

# Preprocess the text: tokenize, remove stopwords, apply stemming, and lemmatization
processed_data = []

for sentence in data:
    tokens = tokenize_text(sentence)  # Tokenization
    tokens = remove_stopwords(tokens)  # Remove stopwords
    tokens = stemming(tokens)  # Stemming (optional, can also apply lemmatization)
    tokens = lemmatization(tokens)  # Lemmatization (optional, can be used instead of stemming)
    
    processed_data.append(" ".join(tokens))

print(processed_data)
OUTPUT:['love program python', 'natur languag process amaz', 'text data process effici nlp techniqu', 'word embedd great way represent word nlp task', 'enjoy solv real-world problem use machin learn']


