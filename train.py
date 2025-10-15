from nltk.stem.snowball import SnowballStemmer
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
import re
from tqdm import tqdm
import os
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


nlp = spacy.load('ru_core_news_sm')
stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer("russian")


dataset = load_dataset('0x7o/taiga', split='train')
if config['dataset']['DEBUG']:   
    dataset = dataset[:2000]['text']
else:
    dataset = dataset['text']

print(len(dataset))


def preprocessing(raw_text:str):
    raw_text = raw_text.lower()
    raw_text = re.sub(r'[^\w\s]|\d', '', raw_text) 
    raw_text = raw_text.replace('\n', ' ')
    raw_text = re.sub(r'[\s]+', ' ', raw_text)
    tokens = [stemmer.stem(i) for i in raw_text.split() if i not in stop_words]
    return tokens


