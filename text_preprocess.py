import pandas as pd
import re
import nltk
import spacy

spacy_en_lemma = spacy.load('en_core_web_sm')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

nltk.download('stopwords')
nltk.download('punkt_tab')


def to_lowercase(text):
    return text.lower()


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)


stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])

def tokenize(text):
    return word_tokenize(text)


def stemming(text):
    tokens = word_tokenize(text)
    return ' '.join([stemmer.stem(word) for word in tokens])


def lemmatize(text):
    token_list = spacy_en_lemma(text)
    return ' '.join([token.lemma_ for token in token_list])


def preprocess_text(text, lemma=True):
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = lemmatize(text) if lemma else stemming(text)
    return text



# Example text dataset
data = {'important_info': [
    "This is a simple example of text preprocessing!",
    "NLP is awesome; you can extract meaning from text.",
    "Text cleaning is the first step in any NLP pipeline."
]}

df = pd.DataFrame(data)

df['cleaned_text'] = df['important_info'].apply(preprocess_text)
print(df[['important_info', 'cleaned_text']])

df.to_csv("processed_data.csv")

