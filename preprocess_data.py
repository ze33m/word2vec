from datasets import load_dataset
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import spacy
from tqdm import tqdm


nlp = spacy.load('ru_core_news_sm')
dataset = load_dataset('0x7o/taiga', split='train')

